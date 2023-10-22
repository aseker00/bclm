import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from bclm.data import bgulex
from bclm.data import hebma
from bclm.model.common.token_embedding_model import TokenEmbeddingModel


class MorphEmbeddingModel(nn.Module):

    def __init__(self, token_emb: TokenEmbeddingModel, postag_emb: TokenEmbeddingModel, ma: hebma.HebrewMorphAnalyzer):
        super(MorphEmbeddingModel, self).__init__()
        self.surface_embedding = token_emb
        self.postag_embedding = postag_emb
        self._ma = ma

    @property
    def embedding_dim(self):
        return self.surface_embedding.embedding_dim

    @property
    def vocab(self) -> dict[str:int]:
        return self.surface_embedding.vocab

    @property
    def postag_vocab(self) -> dict[str:int]:
        return self.postag_embedding.vocab

    def forward(self, input_seq: tuple[torch.Tensor]):
        surface_emb = self.surface_embedding(input_seq[:, :, :-1]).mean(dim=(1, 2, 3))
        tag_emb = self.postag_embedding(input_seq[:, :, :, -1]).mean(dim=(1, 2))
        return torch.stack([surface_emb, tag_emb]).mean(dim=0)

    def to_morph_data_sample(self, words_seq: list[str]) -> pd.DataFrame:
        morph_data = self._to_morph_data(words_seq)
        morph_data['word_id'] = np.vectorize(self.get_vocab_id)(morph_data['word'])
        morph_data['form_id'] = np.vectorize(self.get_vocab_id)(morph_data['form'])
        morph_data['lemma_id'] = np.vectorize(self.get_vocab_id)(morph_data['lemma'])
        morph_data['postag_id'] = np.vectorize(self.get_postag_vocab_id)(morph_data['postag'])
        return morph_data

    # TODO: Consider setting explicit index: [word_index, analysis_index, morph_index]
    def _to_morph_data(self, input_word_seq: list[str]) -> pd.DataFrame:
        words, forms, lemmas, postags = [], [], [], []
        word_indices, analysis_indices, morpheme_indices = [], [], []
        for i, word in enumerate(input_word_seq):
            analyses = self._analyze_word(word)
            for j, a in enumerate(analyses):
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

    def get_vocab_id(self, word: str) -> int:
        return self.vocab.get(word, 0)

    def get_postag_vocab_id(self, tag: str) -> int:
        return self.postag_vocab.get(tag, 0)

    def _analyze_word(self, word: str) -> list[hebma.Analysis]:
        return _analyze_expand(word, self._ma)

    def _analyze_word_find_longest_lemma(self, word: str) -> list[hebma.Analysis]:
        analyses = _analyze_expand(word, self._ma)
        lemma_analyses = _find_all(analyses, self._vocab)
        if lemma_analyses:
            analyses = _sort_analyses(lemma_analyses)[0]
        return analyses


# Morph tensor shape: [num-words x max-num-analyses x max-num-morphemes x 3] (form, lemma, postag)
# TODO: Expand morph tensor to 5-dim batch of samples
# TODO: [num-samples x max-sample-len x max-num-analyses x max-analysis-len x 3] (form, lemma, postag)
def to_morph_input_sample(morph_data_sample: pd.DataFrame) -> torch.Tensor:
    # TODO: check if all these groupby operations can be sped up by setting an explicit index on the morph dataframe
    word_indices, uniq_word_indices = morph_data_sample.word_index.factorize()
    analysis_indices = morph_data_sample.groupby(['word_index']).cumcount().values
    morph_indices = morph_data_sample.groupby(['word_index', 'analysis_index']).cumcount().values
    morph_values = morph_data_sample.loc[:, 'form_id':]
    num_words = word_indices.max() + 1
    max_num_analyses = analysis_indices.max() + 1
    max_num_morphemes = morph_indices.max() + 1
    morph_sample_shape = (num_words, max_num_analyses, max_num_morphemes, morph_values.shape[1])
    morph_sample = torch.zeros(morph_sample_shape, dtype=torch.int)
    morph_sample[word_indices, analysis_indices, morph_indices] = torch.tensor(morph_values.values, dtype=torch.int)
    return morph_sample


def _analyze_expand(word: str, ma: hebma.HebrewMorphAnalyzer) -> list[hebma.Analysis]:
    return [ma.expand_suffixes(a) for a in ma.analyze_word(word)]


def _sort_analyses(analyses: list[hebma.Analysis]) -> list[list[hebma.Analysis]]:
    sorted_analyses = sorted(analyses, key=lambda a: len(a.morphemes))
    return [list(group) for (item, group) in groupby(sorted_analyses, key=lambda a: len(a.morphemes))]


def _find_all(analyses: list[hebma.Analysis], vocab: dict[str:int]) -> list[hebma.Analysis]:
    found = []
    for i, analysis in enumerate(analyses):
        word_lemmas = _get_lemmas(analysis)
        if all([lemma in vocab for lemma in word_lemmas]):
            found.append(analysis)
    return found


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


# Look for forms and lemmas that are not in the list of words (entries)
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


def _build_tag_to_vec(tags: list[str], data: pd.DataFrame, morph: str,
                      word_emb: TokenEmbeddingModel) -> dict[str:torch.Tensor]:
    tag2vec = {}
    for tag in tags:
        tag_morph_value = set(data[data.cpostag == tag][morph])
        tag_morph_indices = [word_emb.vocab[value] for value in tag_morph_value if value in word_emb.vocab]
        if tag_morph_indices:
            tag2vec[tag] = word_emb.embedding.weight[tag_morph_indices].mean(dim=0)
    return tag2vec


# Build embeddings based on the lemmas associated with each POS tag
# Add random vectors for punctuations
def _build_postag_vocab(word_emb: TokenEmbeddingModel, ma: hebma.HebrewMorphAnalyzer) -> (list[str], torch.Tensor):
    # Note: ma.lex_tags doesn't include punctuations and suffixes
    tag2vec = _build_tag_to_vec(ma.lex_tags, ma.lex, 'lemma', word_emb)

    # Preflex entries have empty lemmas, so use the form value to look for embedding vectors
    prefix2vec = _build_tag_to_vec(ma.prefix_tags, ma.preflex, 'form', word_emb)
    tag2vec.update(prefix2vec)

    # Setup sufflex dataframe with form values (suffix entries in lex dataframe have empty forms and lemmas).
    sufflex = ma.lex[ma.lex.cpostag.isin(ma.suffix_tags)].drop_duplicates().reset_index(drop=True)
    suffix_morphemes = [ma.expand_suffix(s) for s in bgulex.data_to_morphemes(sufflex)]
    sufflex['form'] = [m[-1].form if m else '_' for m in suffix_morphemes]
    # sufflex['lemma'] = [m[-1].lemma if m else '_' for m in suffix_morphemes]
    # sufflex['fpostag'] = [m[-1].fpostag.value if m else '_' for m in suffix_morphemes]

    # TODO: as long as hebma doesn't map suffix tags to pronouns we need to add them to the tag vocab
    suffix2vec = _build_tag_to_vec(ma.suffix_tags, sufflex, 'form', word_emb)
    tag2vec.update(suffix2vec)

    tags = list(tag2vec.keys())
    tag_weights = [tag2vec[tag] for tag in tags]
    tags.extend(ma.punct_tags)
    tag_weights.extend([torch.rand(word_emb.embedding_dim, dtype=word_emb.embedding.weight.dtype)
                        for _ in ma.punct_tags])
    return tags, torch.stack(tag_weights)


def _build_morph_embedding(entries: list[str], weights: torch.Tensor,
                           ma: hebma.HebrewMorphAnalyzer) -> TokenEmbeddingModel:
    morphs, morph_weights = _build_morph_vocab(entries, weights, ma)
    entries = entries + morphs
    weights = torch.cat([weights, morph_weights])
    return TokenEmbeddingModel(*(_pad_embedding(entries, weights)))


def _build_postag_embedding(word_emb: TokenEmbeddingModel, ma: hebma.HebrewMorphAnalyzer) -> TokenEmbeddingModel:
    tag_entries, tag_weights = _build_postag_vocab(word_emb, ma)
    return TokenEmbeddingModel(*(_pad_embedding(tag_entries, tag_weights)))


def _pad_embedding(entries: list[str], weights: torch.Tensor) -> (list[str], torch.Tensor):
    pad_vector = torch.zeros(weights.shape[1], dtype=weights.dtype)
    return ['<pad>'] + entries, torch.cat([pad_vector.unsqueeze(0), weights])


def create_morph_emb_model(words: list[str], weights: torch.Tensor,
                           ma: hebma.HebrewMorphAnalyzer) -> MorphEmbeddingModel:
    logging.info(f'Loading Morph Model')
    word_emb_model = _build_morph_embedding(words, weights, ma)
    postag_emb_model = _build_postag_embedding(word_emb_model, ma)
    return MorphEmbeddingModel(word_emb_model, postag_emb_model, ma)


# TODO: Save interm/processed data (embedding entries with lemmas and forms as well as tag embedding entries)
def save_morph_emb_model(emb_model: MorphEmbeddingModel, dest_path: Path):
    pass


# TODO: Load interm/processed data (embedding entries with lemmas and forms as well as tag embedding entries)
def load_morph_emb_model(src_path: Path) -> MorphEmbeddingModel:
    pass
