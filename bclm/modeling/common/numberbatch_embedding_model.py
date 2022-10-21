import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from bclm.data_processing import numberbatch, bgulex
from bclm.data_processing import hebma
from bclm.modeling.common.morph_embedding_model import MorphEmbeddingModel
from bclm.modeling.common.morph_embedding_model import to_morph_input_sample
from bclm.modeling.common.morph_embedding_model import create_morph_emb_model
from bclm.modeling.common.token_embedding_model import TokenEmbeddingModel

import pygtrie
import re


class NumberbatchEmbeddingModel(nn.Module):

    def __init__(self, word_emb: TokenEmbeddingModel, morph_emb: MorphEmbeddingModel, trie: pygtrie.Trie):
        super().__init__()
        self._word_emb = word_emb
        self._morph_emb = morph_emb
        self._trie = trie

    @property
    def embedding_dim(self):
        return self._word_emb.embedding_dim

    @property
    def vocab(self) -> dict[str:int]:
        return self._word_emb.vocab

    @property
    def postag_vocab(self) -> dict[str:int]:
        return self._morph_emb.postag_vocab

    def forward(self, input_seq: tuple[torch.Tensor]):
        phrase_emb = self._word_emb(input_seq[0])
        morph_emb = self._morph_emb(input_seq[1])
        return torch.stack([phrase_emb, morph_emb]).mean(dim=0)

    def to_data_sample(self, words_seq: list[str]) -> (pd.DataFrame, pd.DataFrame):
        # Numberbatch phrases
        phrases_data = _to_phrases_data(words_seq, self._trie)
        phrases_data['word_id'] = np.vectorize(self.get_vocab_id)(phrases_data['word'])
        phrases_data['phrase_id'] = np.vectorize(self.get_vocab_id)(phrases_data['phrase'])

        # Morphological dataframe
        morph_data = self._morph_emb.to_morph_data_sample(words_seq)
        return phrases_data, morph_data

    def get_vocab_id(self, word: str) -> int:
        return self.vocab.get(word, 0)


# TODO: Expand to batch of samples
def to_input_sample(data_sample: tuple[pd.DataFrame]) -> (torch.Tensor, torch.Tensor):
    phrase_input = torch.tensor(data_sample[0]['phrase_id'].values)
    morph_input = to_morph_input_sample(data_sample[1])
    return phrase_input, morph_input


# Match numerical templates, e.g. "## ##" represents any two 2-digit numbers
def _normalize(word: str):
    if word.isdigit() and len(word) > 1:
        return re.sub(".", "#", word)
    return word


def _to_phrases_data(input_word_seq: list[str], trie: pygtrie.Trie) -> pd.DataFrame:
    word_indices = range(len(input_word_seq))
    norm_words = [_normalize(word) for word in input_word_seq]
    phrases = [trie.longest_prefix(' '.join(norm_words[i:])).key for i in range(len(norm_words))]
    return pd.DataFrame(list(zip(word_indices, input_word_seq, phrases)), columns=['word_index', 'word', 'phrase'])


def _read_words(p: Path) -> list[str]:
    with p.open() as f:
        return [line.strip() for line in f.readlines()]


def create_nb_emb_model(lexicon_root_path: Path, nb_langs: list[str]) -> NumberbatchEmbeddingModel:
    logging.info(f'Loading Numberbatch model')
    nbm = numberbatch.Numberbatch(nb_langs)
    ma = hebma.HebrewMorphAnalyzer(*(bgulex.load(lexicon_root_path)))
    entries = [word.replace('_', ' ') for word in nbm.words]
    weights = torch.tensor(nbm.vectors)
    word_emb_model = TokenEmbeddingModel(entries, weights)
    morph_emb_model = create_morph_emb_model(entries, weights, ma)
    nb_trie = pygtrie.StringTrie.fromkeys(entries, value=True, separator=' ')
    return NumberbatchEmbeddingModel(word_emb_model, morph_emb_model, nb_trie)


# TODO: Save interm/processed data (embedding entries with lemmas and forms as well as tag embedding entries)
def save_nb_emb_model(emb_model: NumberbatchEmbeddingModel, dest_path: Path):
    pass


# TODO: Load interm/processed data (embedding entries with lemmas and forms as well as tag embedding entries)
def load_nb_emb_model(src_path: Path) -> NumberbatchEmbeddingModel:
    pass


if __name__ == '__main__':
    he_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    emb_model = create_nb_emb_model(he_root_path, ['he'])
    torch.save(emb_model, 'nb_he_morph_emb_model.pt')
    emb_model = torch.load('nb_he_morph_emb_model.pt')
    print(emb_model)
    sample_sentence = _read_words(Path('words.txt'))
    sample_data = emb_model.to_data_sample(sample_sentence)
    print(sample_data)
    sample_input = to_input_sample(sample_data)
    for word_vec in emb_model(sample_input):
        print(word_vec)
