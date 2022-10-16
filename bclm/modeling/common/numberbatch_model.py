from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from bclm.data_processing import numberbatch as nb, bgulex
from bclm.data_processing import hebma

import pygtrie
import re


class MorphEmbeddingModel(nn.Module):

    def __init__(self, words: list[str], word_weights: torch.Tensor, postags: list[str], postag_weights: torch.Tensor):

        super().__init__()
        pad_vector = torch.zeros(word_weights.shape[1], dtype=word_weights.dtype)
        self._word_embedding = nn.Embedding.from_pretrained(torch.cat([pad_vector.unsqueeze(0), word_weights]),
                                                            padding_idx=0)
        self._word_vocab = {word.replace('_', ' '): i+1 for i, word in enumerate(words)}

        pad_vector = torch.zeros(postag_weights.shape[1], dtype=postag_weights.dtype)
        self._postag_embedding = nn.Embedding.from_pretrained(torch.cat([pad_vector.unsqueeze(0), postag_weights]),
                                                              padding_idx=0)
        self._postag_vocab = {postag: i + 1 for i, postag in enumerate(postags)}

    @property
    def embedding(self) -> nn.Embedding:
        return self._word_embedding

    @property
    def vocab(self) -> dict[str:int]:
        return self._word_vocab

    @property
    def postag_embedding(self) -> nn.Embedding:
        return self._postag_embedding

    @property
    def postag_vocab(self) -> dict[str:int]:
        return self._postag_vocab

    def embed_words(self, words: list[str], ma: hebma.HebrewMorphAnalyzer, trie: pygtrie.Trie) -> torch.Tensor:
        forms, lemmas, postags = _get_morph_analyses(words, ma)
        word_vectors = self._embed_words(words, trie)
        form_vectors = _embed_morph_values(forms, self.vocab, self.embedding)
        lemma_vectors = _embed_morph_values(lemmas, self.vocab, self.embedding)
        postag_vectors = _embed_morph_values(postags, self.postag_vocab, self.postag_embedding)
        return torch.mean(torch.stack([word_vectors, form_vectors, lemma_vectors, postag_vectors]), dim=0)

    def _embed_words(self, words: list[str], trie: pygtrie.Trie) -> torch.Tensor:
        emb_input = torch.zeros(len(words), dtype=torch.int)
        words = [_normalize(word) for word in words]
        i = 0
        while i < len(words):
            longest_prefix = trie.longest_prefix(' '.join(words[i:]))
            if longest_prefix.value:
                emb_input[i] = self.vocab.get(longest_prefix.key)
                parts = longest_prefix.key.split(' ')
                i += len(parts)
            else:
                i += 1
        return self.embedding(emb_input)


# Match number templates, e.g. ##_## represents any 2 2-digit numbers
def _normalize(word: str):
    if word.isdigit() and len(word) > 1:
        return re.sub(".", "#", word)
    return word


def _embed_morph_values(word_analyses: list[list[tuple[str]]], vocab: dict[str:int],
                        embedding: nn.Embedding) -> torch.Tensor:
    word_analyses = [[[vocab.get(value) for value in analysis if value in vocab]
                      for analysis in list(set(analyses))] for analyses in word_analyses]
    seq_lens = [[len(a) for a in analyses] for analyses in word_analyses]
    max_len = max([v for lens in seq_lens for v in lens])
    max_num = max([len(analyses) for analyses in word_analyses])
    emb_input = torch.zeros((len(seq_lens), max_num, max_len), dtype=torch.int)
    for i, analyses in enumerate(word_analyses):
        for j, analysis in enumerate(analyses):
            for k, value in enumerate(analysis):
                emb_input[i][j][k] = value
    emb_vectors = embedding(emb_input)
    return torch.mean(emb_vectors, dim=(1, 2))


def _read_words(p: Path) -> list[str]:
    with p.open() as f:
        return [line.strip() for line in f.readlines()]


def _get_morph_analyses(words: list[str], ma: hebma.HebrewMorphAnalyzer) -> (
        list[list[list[str]]],
        list[list[list[str]]],
        list[list[list[str]]]):
    forms, lemmas, postags = [], [], []
    for word in words:
        word_forms, word_lemmas, word_postags = [], [], []
        for a in _analyze_expand(word, ma):
            word_forms.append(_get_expanded_forms(a))
            word_lemmas.append(_get_expanded_lemmas(a))
            word_postags.append(tuple(a.cpostags))
        forms.append(word_forms)
        lemmas.append(word_lemmas)
        postags.append(word_postags)
    return forms, lemmas, postags


def _analyze_expand(word: str, ma: hebma.HebrewMorphAnalyzer) -> list[hebma.Analysis]:
    return [ma.expand_suffixes(a) for a in ma.analyze_word(word)]


def _get_expanded_forms(analysis: hebma.Analysis) -> tuple[str]:
    expanded_forms = []
    for prefix in analysis.prefixes:
        expanded_forms.append(prefix.form)
    expanded_forms.append(analysis.base.lemma)
    if len(analysis.suffixes) == 2:
        expanded_forms.append(analysis.suffixes[0].form)
    return tuple(expanded_forms)


def _get_expanded_lemmas(analysis: hebma.Analysis) -> tuple[str]:
    expanded_lemmas = []
    for prefix in analysis.prefixes:
        expanded_lemmas.append(prefix.form)
    expanded_lemmas.append(analysis.base.lemma)
    if len(analysis.suffixes) == 1:
        expanded_lemmas.append(analysis.suffixes[0].form)
    elif len(analysis.suffixes) == 2:
        expanded_lemmas.append(analysis.suffixes[0].lemma)
        expanded_lemmas.append(analysis.suffixes[1].form)
    return tuple(expanded_lemmas)


# Look for forms and lemmas that are not in the numberbatch list of words (entries)
# Compute the missing form/lemma embedding vectors by averaging the associated word vectors
def _build_morph_vocab(entries: list[str], weights: torch.Tensor,
                       ma: hebma.HebrewMorphAnalyzer) -> (list[str], list[torch.Tensor]):
    entries_vocab = {e: i for i, e in enumerate(entries)}
    morph2vec = defaultdict(list)
    for i, entry in enumerate(entries):
        words = set(entry.split('_'))
        for word in words:
            for a in ma.analyze_word(word):
                if a.base.form is None:
                    print(f'build morph vocab missing form: {word}')
                    continue
                if a.base.form not in entries_vocab:
                    morph2vec[a.base.form].append(i)
                if a.base.lemma is None:
                    # print(f'build morph vocab missing lemma: {word}')
                    continue
                if a.base.lemma not in entries_vocab:
                    morph2vec[a.base.lemma].append(i)
    morphs = list(morph2vec.keys())
    morph_weights = [weights[morph2vec[m]].mean(dim=0) for m in morphs]
    return morphs, morph_weights


# Build embeddings based on the lemmas associated with each POS tag
# Add random vectors for punctuations
def _build_postag_vocab(entries: list[str], weights: torch.Tensor,
                        ma: hebma.HebrewMorphAnalyzer) -> (list[str], list[torch.Tensor]):
    entries_vocab = {e: i for i, e in enumerate(entries)}
    tag2vec = {}
    for tag in ma.lex_tags:
        tag_lemmas = set(ma.lex[ma.lex.cpostag == tag].lemma)
        tag_lemma_indices = [entries_vocab[lemma] for lemma in tag_lemmas if lemma in entries_vocab]
        if tag_lemma_indices:
            tag2vec[tag] = weights[tag_lemma_indices].mean(dim=0)
    prefix2vec = {}
    for prefix_tag in ma.prefix_tags:
        prefix_forms = set(ma.preflex[ma.preflex.cpostag == prefix_tag].form)
        prefix_form_indices = [entries_vocab[form] for form in prefix_forms if form in entries_vocab]
        if prefix_form_indices:
            prefix2vec[prefix_tag] = weights[prefix_form_indices].mean(dim=0)
    tag2vec.update(prefix2vec)
    tags = list(tag2vec.keys())
    tag_weights = [tag2vec[tag] for tag in tags]
    tags.extend(ma.punct_tags)
    tag_weights.extend([torch.rand(weights.shape[1], dtype=weights.dtype) for _ in ma.punct_tags])
    return tags, tag_weights


if __name__ == '__main__':
    root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    heb_ma = hebma.HebrewMorphAnalyzer(*(bgulex.load(root_path)))

    nbm = nb.Numberbatch(['he'])
    nb_word_weights = torch.FloatTensor(nbm.vectors)

    nb_tags, nb_tag_weights = _build_postag_vocab(nbm.words, nb_word_weights, heb_ma)
    nb_morphs, nb_morph_weights = _build_morph_vocab(nbm.words, nb_word_weights, heb_ma)
    nb_words = nbm.words + nb_morphs
    nb_word_weights = torch.cat([nb_word_weights, torch.stack(nb_morph_weights)])
    emb_model = MorphEmbeddingModel(nb_words, nb_word_weights, nb_tags, torch.stack(nb_tag_weights))
    torch.save(emb_model, 'nb_morph_emb_model.pt')
    emb_model = torch.load('nb_morph_emb_model.pt')
    print(emb_model.embedding)
    print(emb_model.postag_embedding)
    nb_trie = pygtrie.StringTrie.fromkeys(emb_model.vocab.keys(), value=True, separator=' ')
    sample_sentence = _read_words(Path('words.txt'))
    for word_vec in emb_model.embed_words(sample_sentence, heb_ma, nb_trie):
        print(word_vec)
