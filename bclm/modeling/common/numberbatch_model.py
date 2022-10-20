import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from bclm.data_processing import numberbatch, bgulex
from bclm.data_processing import hebma

import pygtrie
import re


class MorphEmbeddingModel(nn.Module):

    def __init__(self, vocab: dict[str:int], embedding: nn.Embedding,
                 postag_vocab: dict[str:int], postag_embedding: nn.Embedding,
                 ma: hebma.HebrewMorphAnalyzer, trie: pygtrie.Trie):
        super(MorphEmbeddingModel, self).__init__()
        self.embedding = embedding
        self.postag_embedding = postag_embedding
        self._vocab = vocab
        self._postag_vocab = postag_vocab
        self._ma = ma
        self._trie = trie

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    @property
    def vocab(self) -> dict[str:int]:
        return self._vocab

    @property
    def postag_vocab(self) -> dict[str:int]:
        return self._postag_vocab

    def forward(self, input_seq: tuple[torch.Tensor]):
        word_emb = self.embedding(input_seq[0])
        morph_emb = self.embedding(input_seq[1][:, :, :-1]).mean(dim=(1, 2, 3))
        tag_emb = self.postag_embedding(input_seq[1][:, :, :, -1]).mean(dim=(1, 2))
        return torch.stack([word_emb, morph_emb, tag_emb]).mean(dim=0)

    def to_data_sample(self, words_seq: list[str]) -> (pd.DataFrame, pd.DataFrame):
        # Numberbatch phrases data frame
        words_data = _to_words_data(words_seq, self._trie)
        words_data['word_id'] = np.vectorize(self.get_vocab_id)(words_data['word'])
        words_data['phrase_id'] = np.vectorize(self.get_vocab_id)(words_data['phrase'])

        # Morphological data frame
        morph_data = _to_morph_data(words_seq, self._ma)
        morph_data['word_id'] = np.vectorize(self.get_vocab_id)(morph_data['word'])
        morph_data['form_id'] = np.vectorize(self.get_vocab_id)(morph_data['form'])
        morph_data['lemma_id'] = np.vectorize(self.get_vocab_id)(morph_data['lemma'])
        morph_data['postag_id'] = np.vectorize(self.get_postag_vocab_id)(morph_data['postag'])
        return words_data, morph_data

    def get_vocab_id(self, word: str) -> int:
        return self.vocab.get(word, 0)

    def get_postag_vocab_id(self, tag: str) -> int:
        return self.postag_vocab.get(tag, 0)


# Word tensor shape: [num-words x 2] (word, nb phrase)
# Morph tensor shape: [num-words x max-num-analyses x max-num-morphemes x 3] (form, lemma, postag)
# TODO: Expand morph tensor to 5-dim batch of samples
# TODO: [num-samples x max-sample-len x max-num-analyses x max-analysis-len x 3] (form, lemma, postag)
def to_input_sample(data_sample: tuple[pd.DataFrame]) -> (torch.Tensor, torch.Tensor):
    # Word data sample
    word_sample = torch.tensor(data_sample[0]['phrase_id'].values)

    # Morph data sample
    # TODO: check if all these groupby operations can be sped up by setting an explicit index on the morph dataframe
    word_indices, uniq_word_indices = data_sample[1].word_index.factorize()
    analysis_indices = data_sample[1].groupby(['word_index']).cumcount().values
    morph_indices = data_sample[1].groupby(['word_index', 'analysis_index']).cumcount().values
    morph_values = data_sample[1].loc[:, 'form_id':]
    morph_sample_shape = (word_indices.max() + 1, analysis_indices.max() + 1, morph_indices.max() + 1,
                          morph_values.shape[1])
    morph_sample = torch.zeros(morph_sample_shape, dtype=torch.int)
    morph_sample[word_indices, analysis_indices, morph_indices] = torch.tensor(morph_values.values, dtype=torch.int)

    return word_sample, morph_sample


# Match numerical templates, e.g. "## ##" represents any two 2-digit numbers
def _normalize(word: str):
    if word.isdigit() and len(word) > 1:
        return re.sub(".", "#", word)
    return word


def _index_words(words: list[str], vocab: dict[str:int], trie: pygtrie.Trie) -> torch.Tensor:
    indices = torch.zeros(len(words), dtype=torch.int)
    words = [_normalize(word) for word in words]
    i = 0
    while i < len(words):
        longest_prefix = trie.longest_prefix(' '.join(words[i:]))
        if longest_prefix.value:
            indices[i] = vocab.get(longest_prefix.key)
            parts = longest_prefix.key.split(' ')
            i += len(parts)
        else:
            i += 1
    return indices


def _to_words_data(input_word_seq: list[str], trie: pygtrie.Trie) -> pd.DataFrame:
    phrases = [None] * len(input_word_seq)
    norm_words = [_normalize(word) for word in input_word_seq]
    i = 0
    while i < len(norm_words):
        longest_prefix = trie.longest_prefix(' '.join(norm_words[i:]))
        if longest_prefix.value:
            phrases[i] = longest_prefix.key
            parts = longest_prefix.key.split(' ')
            i += len(parts)
        else:
            i += 1
    return pd.DataFrame(list(zip(range(len(input_word_seq)), input_word_seq, phrases)),
                        columns=['word_index', 'word', 'phrase'])


# TODO: Consider setting explicit index: [word_index, analysis_index, morph_index]
def _to_morph_data(input_word_seq: list[str], ma: hebma.HebrewMorphAnalyzer) -> pd.DataFrame:
    words, forms, lemmas, postags = [], [], [], []
    word_indices, analysis_indices, morpheme_indices = [], [], []
    for i, word in enumerate(input_word_seq):
        for j, a in enumerate(_analyze_expand(word, ma)):
            word_forms = _get_forms(a)
            word_lemmas = _get_lemmas(a)
            word_postags = _get_postags(a)
            word_indices.extend((i for _ in word_forms))
            analysis_indices.extend((j for _ in word_forms))
            morpheme_indices.extend(range(len(word_forms)))
            words.extend((word for _ in word_forms))
            forms.extend(word_forms)
            lemmas.extend(word_lemmas)
            postags.extend(word_postags)
    return pd.DataFrame(list(zip(word_indices, analysis_indices, morpheme_indices, words, forms, lemmas, postags)),
                        columns=['word_index', 'analysis_index', 'morph_index', 'word', 'form', 'lemma', 'postag'])


def _index_morph(analyses: list[list[tuple[str]]], vocab: dict[str:int]) -> torch.Tensor:
    # word_analyses = [[[vocab.get(value) for value in analysis if value in vocab]
    word_analyses = [[[vocab.get(value, 0) for value in analysis] for analysis in word_analyses]
                     for word_analyses in analyses]
    seq_lens = [[len(a) for a in analyses] for analyses in word_analyses]
    max_len = max([v for lens in seq_lens for v in lens])
    max_num = max([len(analyses) for analyses in word_analyses])
    indices = torch.zeros((len(seq_lens), max_num, max_len), dtype=torch.int)
    for i, analyses in enumerate(word_analyses):
        for j, analysis in enumerate(analyses):
            for k, value in enumerate(analysis):
                indices[i][j][k] = value
    return indices


def _read_words(p: Path) -> list[str]:
    with p.open() as f:
        return [line.strip() for line in f.readlines()]


def _get_morph_analyses(words: list[str], ma: hebma.HebrewMorphAnalyzer) -> (
        list[list[str]],
        list[list[str]],
        list[list[str]]):
    forms, lemmas, postags = [], [], []
    for word in words:
        word_forms, word_lemmas, word_postags = [], [], []
        for a in _analyze_expand(word, ma):
            word_forms.append(_get_forms(a))
            word_lemmas.append(_get_lemmas(a))
            word_postags.append(_get_postags(a))
        forms.append(word_forms)
        lemmas.append(word_lemmas)
        postags.append(word_postags)
    return forms, lemmas, postags


def _analyze_expand(word: str, ma: hebma.HebrewMorphAnalyzer) -> list[hebma.Analysis]:
    return [ma.expand_suffixes(a) for a in ma.analyze_word(word)]


def _get_forms(analysis: hebma.Analysis) -> tuple[str]:
    if not analysis.suffixes:
        return tuple(analysis.forms)
    forms = [prefix.form for prefix in analysis.prefixes]
    forms.append(analysis.base.lemma)
    if len(analysis.suffixes) == 2:
        forms.append(analysis.suffixes[0].lemma)
    forms.append(analysis.suffixes[-1].form)
    return tuple(forms)


def _get_lemmas(analysis: hebma.Analysis) -> tuple[str]:
    lemmas = [prefix.form for prefix in analysis.prefixes]
    lemmas.append(analysis.base.lemma)
    lemmas.extend([suffix.lemma for suffix in analysis.suffixes])
    return tuple(lemmas)


def _get_postags(analysis: hebma.Analysis) -> tuple[str]:
    return tuple(analysis.cpostags)


# Look for forms and lemmas that are not in the numberbatch list of words (entries)
# Compute the missing form/lemma embedding vectors by averaging the associated word vectors
def _build_morph_vocab(entries: list[str], weights: torch.Tensor,
                       ma: hebma.HebrewMorphAnalyzer) -> (list[str], torch.Tensor):
    vocab = set(entries)
    morph2vec = defaultdict(list)
    for i, entry in enumerate(entries):
        words = set(entry.split())
        for word in words:
            for a in ma.analyze_word(word):
                if a.base.form is None:
                    print(f'build morph vocab missing form: {word}')
                    continue
                if a.base.form not in vocab:
                    morph2vec[a.base.form].append(i)
                if a.base.lemma is None:
                    # print(f'build morph vocab missing lemma: {word}')
                    continue
                if a.base.lemma not in vocab:
                    morph2vec[a.base.lemma].append(i)
    morphs = list(morph2vec.keys())
    morph_weights = [weights[morph2vec[m]].mean(dim=0) for m in morphs]
    return morphs, torch.stack(morph_weights)


def _build_tag_to_vec(tags: list[str], data: pd.DataFrame, morph: str, vocab: dict[str:int], weights: torch.Tensor):
    tag2vec = {}
    for tag in tags:
        tag_morph_value = set(data[data.cpostag == tag][morph])
        tag_morph_indices = [vocab[value] for value in tag_morph_value if value in vocab]
        if tag_morph_indices:
            tag2vec[tag] = weights[tag_morph_indices].mean(dim=0)
    return tag2vec


# Build embeddings based on the lemmas associated with each POS tag
# Add random vectors for punctuations
def _build_postag_vocab(vocab: dict[str:int], weights: torch.Tensor,
                        ma: hebma.HebrewMorphAnalyzer) -> (list[str], torch.Tensor):
    prefix2vec = _build_tag_to_vec(ma.prefix_tags, ma.preflex, 'form', vocab, weights)

    # Note: ma.lex_tags doesn't include punctuations and suffixes
    tag2vec = _build_tag_to_vec(ma.lex_tags, ma.lex, 'lemma', vocab, weights)

    # TODO: as long as hebma doesn't map suffix tags to pronouns we need to add them to the tag vocab
    sufflex = ma.lex[ma.lex.cpostag.isin(ma.suffix_tags)].drop_duplicates().reset_index(drop=True)
    suffix_morphemes = [ma.expand_suffix(s) for s in bgulex.data_to_morphemes(sufflex)]
    sufflex['form'] = [m[-1].form if m else '_' for m in suffix_morphemes]
    # sufflex['lemma'] = [m[-1].lemma if m else '_' for m in suffix_morphemes]
    # sufflex['fpostag'] = [m[-1].fpostag.value if m else '_' for m in suffix_morphemes]
    suffix2vec = _build_tag_to_vec(ma.suffix_tags, sufflex, 'form', vocab, weights)

    tag2vec.update(prefix2vec)
    tag2vec.update(suffix2vec)
    tags = list(tag2vec.keys())
    tag_weights = [tag2vec[tag] for tag in tags]
    tags.extend(ma.punct_tags)
    tag_weights.extend([torch.rand(weights.shape[1], dtype=weights.dtype) for _ in ma.punct_tags])
    return tags, torch.stack(tag_weights)


def _build_morph_embedding(entries: list[str], weights: torch.Tensor,
                           ma: hebma.HebrewMorphAnalyzer) -> (dict[str:int], nn.Embedding):
    morphs, morph_weights = _build_morph_vocab(entries, weights, ma)
    entries = entries + morphs
    weights = torch.cat([weights, morph_weights])
    return _build_embedding(entries, weights)


def _build_postag_embedding(vocab: dict[str:int], weights: torch.Tensor,
                            ma: hebma.HebrewMorphAnalyzer) -> (dict[str:int], nn.Embedding):
    tag_entries, tag_weights = _build_postag_vocab(vocab, weights, ma)
    return _build_embedding(tag_entries, tag_weights)


def _build_embedding(entries: list[str], weights: torch.Tensor) -> (dict, nn.Embedding):
    pad_vector = torch.zeros(weights.shape[1], dtype=weights.dtype)
    embedding = nn.Embedding.from_pretrained(torch.cat([pad_vector.unsqueeze(0), weights]), padding_idx=0)
    vocab = {entry: i for i, entry in enumerate(['<pad>'] + entries)}
    return vocab, embedding


def create_morph_emb_model(lexicon_root_path: Path, nb_langs: list[str]) -> MorphEmbeddingModel:
    logging.info(f'Loading Numberbatch Morph Model')
    nbm = numberbatch.Numberbatch(nb_langs)
    ma = hebma.HebrewMorphAnalyzer(*(bgulex.load(lexicon_root_path)))
    entries = [word.replace('_', ' ') for word in nbm.words]
    weights = torch.tensor(nbm.vectors)
    vocab, embedding = _build_morph_embedding(entries, weights, ma)
    postag_vocab, postag_embedding = _build_postag_embedding(vocab, embedding.weight, ma)
    trie = pygtrie.StringTrie.fromkeys(vocab.keys(), value=True, separator=' ')
    return MorphEmbeddingModel(vocab, embedding, postag_vocab, postag_embedding, ma, trie)


# TODO: Save interm/processed data (embedding entries with lemmas and forms as well as tag embedding entries)
def save_morph_emb_model():
    pass


# TODO: Load interm/processed data (embedding entries with lemmas and forms as well as tag embedding entries)
def load_morph_emb_model():
    pass


if __name__ == '__main__':
    # he_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    # emb_model = create_morph_emb_model(he_root_path, ['he'])
    # torch.save(emb_model, 'he_nb_morph_emb_model.pt')
    emb_model = torch.load('he_nb_morph_emb_model.pt')
    print(emb_model)
    # print(list(emb_model.parameters()))
    sample_sentence = _read_words(Path('words.txt'))
    sample_data = emb_model.to_data_sample(sample_sentence)
    sample_input = to_input_sample(sample_data)
    for word_vec in emb_model(sample_input):
        print(word_vec)
