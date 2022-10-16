from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from bclm.data_processing import numberbatch as nb, bgulex
from bclm.data_processing import hebma

import pygtrie


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

    def embed_words(self, words: list[str], ma: hebma.HebrewMorphAnalyzer, tree: pygtrie.CharTrie) -> torch.Tensor:
        forms, lemmas, postags = _get_morph_analyses(words, ma)
        word_vectors = self._embed_words(words, tree)
        form_vectors = _embed_morph_values(forms, self._word_vocab, self.word_embedding)
        lemma_vectors = _embed_morph_values(lemmas, self._word_vocab, self.word_embedding)
        postag_vectors = _embed_morph_values(postags, self._postag_vocab, self.postag_embedding)
        return torch.mean(torch.stack([word_vectors, form_vectors, lemma_vectors, postag_vectors]), dim=0)

    def _embed_words(self, words: list[str], tree: pygtrie.CharTrie) -> torch.Tensor:
        i = 0
        emb_input = torch.zeros(len(words), dtype=torch.int)
        while i < len(words):
            longest_prefix = tree.longest_prefix(' '.join(words[i:]))
            if longest_prefix.value:
                parts = longest_prefix.key.split(' ')
                emb_input[i] = self.vocab.get(longest_prefix.key)
                i += len(parts)
            else:
                i += 1
        return self.word_embedding(emb_input)


# TODO: Match to number templates, e.g. ##_## can represent any 2 2-digit numbers
def _lookup_numbers():
    pass


def _embed_morph_values(word_analyses: list[list[list[str]]], vocab: dict[str:int], embedding: nn.Embedding) -> (
        torch.Tensor):
    word_indices = [[torch.tensor([vocab.get(value, 0) for value in analysis], dtype=torch.int)
                     for analysis in analyses] if analyses else [torch.tensor([0], dtype=torch.int)]
                    for analyses in word_analyses]
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


def _get_morph_analyses(words: list[str], ma: hebma.HebrewMorphAnalyzer) -> (
        list[list[list[str]]],
        list[list[list[str]]],
        list[list[list[str]]]):
    forms, lemmas, postags = [], [], []
    for word in words:
        word_forms, word_lemmas, word_postags = _analyze_expand(word, ma)
        forms.append(word_forms)
        lemmas.append(word_lemmas)
        postags.append(word_postags)
    return forms, lemmas, postags


def _analyze_expand(word: str, ma: hebma.HebrewMorphAnalyzer) -> (list[list[str]], list[list[str]], list[list[str]]):
    word_forms, word_lemmas, word_postags = [], [], []
    for a in ma.analyze_word(word):
        a = ma.expand_suffixes(a)
        word_forms.append(_get_expanded_forms(a))
        word_lemmas.append(_get_expanded_lemmas(a))
        word_postags.append(a.cpostags)
    return word_forms, word_lemmas, word_postags


def _get_expanded_forms(analysis: hebma.Analysis) -> list[str]:
    expanded_forms = []
    for prefix in analysis.prefixes:
        expanded_forms.append(prefix.form)
    expanded_forms.append(analysis.base.lemma)
    if len(analysis.suffixes) == 2:
        expanded_forms.append(analysis.suffixes[0].form)
    return expanded_forms


def _get_expanded_lemmas(analysis: hebma.Analysis) -> list[str]:
    expanded_lemmas = []
    for prefix in analysis.prefixes:
        expanded_lemmas.append(prefix.form)
    expanded_lemmas.append(analysis.base.lemma)
    if len(analysis.suffixes) == 1:
        expanded_lemmas.append(analysis.suffixes[0].form)
    elif len(analysis.suffixes) == 2:
        expanded_lemmas.append(analysis.suffixes[0].lemma)
        expanded_lemmas.append(analysis.suffixes[1].form)
    return expanded_lemmas


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
                elif a.base.form not in entries_vocab:
                    morph2vec[a.base.form].append(i)
                if a.base.lemma is None:
                    print(f'build morph vocab missing lemma: {word}')
                elif a.base.lemma not in entries_vocab:
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
    for tag in ma.postags:
        tag_lemmas = set(ma.lex[ma.lex.cpostag == tag].lemma)
        tag_lemma_indices = [entries_vocab[lemma] for lemma in tag_lemmas if lemma in entries_vocab]
        if tag_lemma_indices:
            tag2vec[tag] = weights[tag_lemma_indices].mean(dim=0)
    prefix2vec = {}
    for prefix in ma.prefixes:
        prefix_forms = set(ma.preflex[ma.preflex.cpostag == prefix].form)
        prefix_form_indices = [entries_vocab[form] for form in prefix_forms if form in entries_vocab]
        if prefix_form_indices:
            prefix2vec[prefix] = weights[prefix_form_indices].mean(dim=0)
    tag2vec.update(prefix2vec)
    tags = list(tag2vec.keys())
    tag_weights = [tag2vec[tag] for tag in tags]
    tags.extend(ma.punctuations)
    tag_weights.extend([torch.rand(weights.shape[1], dtype=weights.dtype) for _ in ma.punctuations])
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
    tree = pygtrie.StringTrie.fromkeys(emb_model.vocab.keys(), value=True, separator=' ')
    sample_sentence = _read_words(Path('words.txt'))
    for word_vec in emb_model.embed_words(sample_sentence, heb_ma, tree):
        print(word_vec)
