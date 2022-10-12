from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bclm.data_processing import numberbatch as nb, bgulex
from bclm.data_processing import hebma


class MorphEmbeddingModel(nn.Module):

    def __init__(self, words: list, word_weights: torch.Tensor, postags: list, postag_weights: torch.Tensor):
        super().__init__()
        pad_vector = torch.zeros(word_weights.shape[1], dtype=word_weights.dtype)
        self.word_embedding = nn.Embedding.from_pretrained(torch.cat([pad_vector.unsqueeze(0), word_weights]),
                                                           padding_idx=0)
        self.word_vocab = {word: i+1 for i, word in enumerate(words)}

        pad_vector = torch.zeros(postag_weights.shape[1], dtype=postag_weights.dtype)
        self.postag_embedding = nn.Embedding.from_pretrained(torch.cat([pad_vector.unsqueeze(0), postag_weights]),
                                                             padding_idx=0)
        self.postag_vocab = {postag: i + 1 for i, postag in enumerate(postags)}

    def embed_words(self, words: list[str], ma: hebma.HebrewMorphAnalyzer) -> torch.Tensor:
        forms, lemmas, postags = _get_morph_analyses(words, ma)
        word_vectors = self._embed_words(words)
        form_vectors = _embed_morph_values(forms, self.word_vocab, self.word_embedding)
        lemma_vectors = _embed_morph_values(lemmas, self.word_vocab, self.word_embedding)
        postag_vectors = _embed_morph_values(postags, self.postag_vocab, self.postag_embedding)
        return torch.mean(torch.stack([word_vectors, form_vectors, lemma_vectors, postag_vectors]), dim=0)

    def _embed_words(self, words: list[str]) -> torch.Tensor:
        emb_input = torch.tensor([self.word_vocab.get(word, 0) for word in words])
        # return self.embedding(emb_input), emb_input > 0
        return self.word_embedding(emb_input)


def _embed_morph_values(word_analyses: list[list[list[str]]], vocab: dict[str:int],
                        embedding: nn.Embedding) -> torch.Tensor:
    word_indices = [[torch.tensor([vocab.get(value, 0) for value in analysis]) for analysis in analyses]
                    if analyses else [torch.tensor([0])] for analyses in word_analyses]
    seq_lens = [[len(a) for a in analyses] for analyses in word_indices]
    max_len = max([v for lens in seq_lens for v in lens])
    max_num = max([len(analyses) for analyses in word_indices])
    emb_input = [torch.stack([F.pad(a, (0, max_len-len(a))) for a in analyses], dim=0) for analyses in word_indices]
    emb_input = torch.stack([F.pad(analyses, (0, 0, 0, max_num - len(analyses))) for analyses in emb_input])
    emb_vectors = embedding(emb_input)
    return torch.mean(emb_vectors, dim=(1, 2))


def _read_words(p: Path) -> list[str]:
    with p.open() as f:
        return [line.strip() for line in f.readlines()]


def _get_morph_analyses(words: list[str], ma: hebma.HebrewMorphAnalyzer) -> (list[list[list[str]]],
                                                                             list[list[list[str]]],
                                                                             list[list[list[str]]]):
    forms, lemmas, postags = [], [], []
    for word in words:
        word_forms, word_lemmas, word_postags = _analyze_expand(word, ma)
        forms.append(word_forms)
        lemmas.append(word_lemmas)
        postags.append(word_postags)
    return forms, lemmas, postags


def _analyze_expand(word: str, ma: hebma.HebrewMorphAnalyzer) -> (list[list[str]],
                                                                  list[list[str]],
                                                                  list[list[str]]):
    word_forms, word_lemmas, word_postags = [], [], []
    for a in ma.analyze_word(word):
        a = ma.expand_suffixes(a)
        word_forms.append(_get_expanded_forms(a))
        word_lemmas.append(_get_expanded_lemmas(a))
        word_postags.append(_get_expanded_postags(a))
    return word_forms, word_lemmas, word_postags


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


def _get_expanded_postags(analysis: hebma.Analysis) -> list[str]:
    expanded_postags = []
    for prefix in analysis.prefixes:
        expanded_postags.append(prefix.cpostag.value)
    expanded_postags.append(analysis.base.cpostag.value)
    if analysis.suffixes:
        expanded_postags.append(analysis.suffixes[0].cpostag.value)
        expanded_postags.append(analysis.suffixes[1].cpostag.value)
    return expanded_postags


def _build_morph_vocab(words: list, weights: torch.Tensor,
                       ma: hebma.HebrewMorphAnalyzer) -> ((list[str], list[torch.Tensor]),
                                                          (list[str], list[torch.Tensor])):
    words_vocab = set(words)
    morph2vec = defaultdict(list)
    pos2vec = defaultdict(list)
    for i, word in enumerate(words):
        for a in ma.analyze_word(word):
            if a.base.form not in words_vocab:
                morph2vec[a.base.form].append(i)
            if a.base.lemma not in words_vocab:
                morph2vec[a.base.lemma].append(i)
            pos2vec[a.base.cpostag.value].append(i)
    morph_words = list(morph2vec.keys())
    postags = list(pos2vec.keys())
    morph_weights = [torch.mean(torch.stack([weights[i] for i in morph2vec[morph]]), dim=0) for morph in morph_words]
    postag_weights = [torch.mean(torch.stack([weights[i] for i in pos2vec[postag]]), dim=0) for postag in postags]
    return (morph_words, morph_weights), (postags, postag_weights)


if __name__ == '__main__':
    root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    heb_ma = hebma.HebrewMorphAnalyzer(*(bgulex.load(root_path)))

    nbm = nb.Numberbatch(['he'])
    nb_word_weights = torch.FloatTensor(nbm.vectors)

    (nb_morph_words, nb_morph_weights), (nb_postags, nb_postag_weights) = _build_morph_vocab(nbm.words,
                                                                                             nb_word_weights, heb_ma)
    emb_words = nbm.words + nb_morph_words
    emb_weights = torch.cat([nb_word_weights, torch.stack(nb_morph_weights)])
    emb_model = MorphEmbeddingModel(emb_words, emb_weights, nb_postags, torch.stack(nb_postag_weights))
    torch.save(emb_model, 'nb_emb_model.pt')

    emb_model = torch.load('nb_emb_model.pt')
    print(emb_model.word_embedding)
    print(emb_model.postag_embedding)
    sample_sentence = _read_words(Path('words.txt'))
    for t in emb_model.embed_words(sample_sentence, heb_ma):
        print(t)