from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bclm.data_processing import numberbatch as nb, hebtagset, bgulex
from bclm.data_processing import hebma


class HebrewEmbeddingModel(nn.Module):

    def __init__(self, words: list, weights: torch.Tensor, pos2vec: dict[hebtagset.POSTag:list[int]]):
        super().__init__()
        pad_vector = torch.zeros(weights.shape[1], dtype=weights.dtype)
        self.embedding = nn.Embedding.from_pretrained(torch.cat([pad_vector.unsqueeze(0), weights]), padding_idx=0)
        self.vocab = {word: i+1 for i, word in enumerate(words)}
        self.pos2vec = pos2vec

    def embed_words(self, words: list[str], ma: hebma.HebrewMorphAnalyzer) -> torch.Tensor:
        forms, lemmas = _get_morph_analyses(words, ma)
        word_vectors = self._embed_words(words)
        form_vectors = self._embed_morph_values(forms)
        lemma_vectors = self._embed_morph_values(lemmas)
        return torch.mean(torch.stack([word_vectors, form_vectors, lemma_vectors]), dim=0)

    def _embed_words(self, words: list[str]) -> torch.Tensor:
        emb_input = torch.tensor([self.vocab.get(word, 0) for word in words])
        # return self.embedding(emb_input), emb_input > 0
        return self.embedding(emb_input)

    def _embed_morph_values(self, word_analyses: list[list[list[str]]]) -> torch.Tensor:
        word_indices = [[torch.tensor([self.vocab.get(value, 0) for value in analysis]) for analysis in analyses]
                        if analyses else [torch.tensor([0])] for analyses in word_analyses]
        seq_lens = [[len(a) for a in analyses] for analyses in word_indices]
        max_len = max([v for lens in seq_lens for v in lens])
        max_num = max([len(analyses) for analyses in word_indices])
        emb_input = [torch.stack([F.pad(a, (0, max_len-len(a))) for a in analyses], dim=0) for analyses in word_indices]
        emb_input = torch.stack([F.pad(analyses, (0, 0, 0, max_num - len(analyses))) for analyses in emb_input])
        emb_vectors = self.embedding(emb_input)
        return torch.mean(emb_vectors, dim=(1, 2))


def _read_words(p: Path) -> list[str]:
    with p.open() as f:
        return [line.strip() for line in f.readlines()]


def _get_morph_analyses(words: list[str], ma: hebma.HebrewMorphAnalyzer) -> (list[list[list[str]]],
                                                                             list[list[list[str]]]):
    forms, lemmas = [], []
    for word in words:
        word_forms, word_lemmas = _analyze_expand(word, ma)
        forms.append(word_forms)
        lemmas.append(word_lemmas)
    return forms, lemmas


def _analyze_expand(word: str, ma: hebma.HebrewMorphAnalyzer) -> (list[list[str]],
                                                                  list[list[str]]):
    word_forms, word_lemmas = [], []
    for a in ma.analyze_word(word):
        a = ma.expand_suffixes(a)
        word_forms.append(_get_expanded_forms(a))
        word_lemmas.append(_get_expanded_lemmas(a))
    return word_forms, word_lemmas


def _get_expanded_forms(analysis: hebma.Analysis) -> list[str]:
    expanded_forms = []
    for prefix in analysis.prefixes:
        expanded_forms.append(prefix.form)
    expanded_forms.append(analysis.base.lemma)
    if analysis.suffixes:
        expanded_forms.append(analysis.suffixes[0].form)
    return expanded_forms


def _get_expanded_lemmas(analysis: hebma.Analysis) -> list[str]:
    expanded_lemmas = []
    for prefix in analysis.prefixes:
        expanded_lemmas.append(prefix.form)
    expanded_lemmas.append(analysis.base.lemma)
    if analysis.suffixes:
        expanded_lemmas.append(analysis.suffixes[0].lemma)
        expanded_lemmas.append(analysis.suffixes[1].form)
    return expanded_lemmas


def _build_morph_vocab(words: list, weights: torch.Tensor,
                       ma: hebma.HebrewMorphAnalyzer) -> (list[str], list[torch.Tensor],
                                                          dict[hebtagset.POSTag:list[int]]):
    words_vocab = set(words)
    morph2vec = defaultdict(list)
    pos2vec = defaultdict(list)
    for i, word in enumerate(words):
        word_postags = set()
        for a in ma.analyze_word(word):
            if a.base.form not in words_vocab:
                morph2vec[a.base.form].append(i)
            if a.base.lemma not in words_vocab:
                morph2vec[a.base.lemma].append(i)
            word_postags.add(a.base.cpostag)
        if len(word_postags) == 1:
            pos2vec[list(word_postags)[0]].append(i)
    words = list(morph2vec.keys())
    word_weights = [torch.mean(torch.stack([weights[i] for i in morph2vec[morph]]), dim=0) for morph in words]
    # postag_weights = {t: torch.mean(torch.stack([word_weights[i] for i in pos2vec[t]]), dim=0) for t in pos2vec}
    return words, word_weights, pos2vec


if __name__ == '__main__':
    root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    heb_ma = hebma.HebrewMorphAnalyzer(*(bgulex.load(root_path)))

    nbm = nb.Numberbatch(['he'])
    # word_weights = torch.FloatTensor(nbm.vectors)
    # morph_words, morph_weights, morph_pos2vec = _build_morph_vocab(nbm.words, word_weights, heb_ma)

    # emb_words = nbm.words + morph_words
    # emb_weights = torch.cat([word_weights, torch.stack(morph_weights)])
    # emb_model = HebrewEmbeddingModel(emb_words, emb_weights, morph_pos2vec)
    # torch.save(emb_model, 'nb_emb_model.pt')

    emb_model = torch.load('nb_emb_model.pt')
    print(emb_model.embedding)
    sample_sentence = _read_words(Path('words.txt'))
    for t in emb_model.embed_words(sample_sentence, heb_ma):
        print(t)
