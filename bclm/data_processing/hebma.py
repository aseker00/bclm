import pandas as pd
from collections import defaultdict
from bclm.data_processing import hebtb, bgulex, hebtagset
from bclm.data_processing.format import conllx
from pathlib import Path
import re


class Analysis:

    def __init__(self, word: str, prefixes: list[conllx.Morpheme], base: conllx.Morpheme,
                 suffixes: list[conllx.Morpheme]):
        self._word = word
        self._prefixes = prefixes
        self._base = base
        self._suffixes = suffixes
        self._morphemes = None

    @property
    def word(self) -> str:
        return self._word

    @property
    def prefixes(self) -> list[conllx.Morpheme]:
        return self._prefixes

    @property
    def base(self) -> conllx.Morpheme:
        return self._base

    @property
    def suffixes(self) -> list[conllx.Morpheme]:
        return self._suffixes

    @property
    def morphemes(self) -> list[conllx.Morpheme]:
        if not self._morphemes:
            self._morphemes = [m for m in self.prefixes + [self.base] + self.suffixes if m]
        return self._morphemes

    @property
    def forms(self) -> list[str]:
        return [m.form for m in self.morphemes if m.form]

    @property
    def lemmas(self) -> list[str]:
        return [m.lemma for m in self.morphemes if m.lemma]

    @property
    def cpostags(self) -> list[str]:
        return [m.cpostag.value for m in self.morphemes if m.cpostag]

    @property
    def feats(self) -> list[str]:
        return [hebtagset.format_parsed_features(m.feats) for m in self.morphemes if m.feats]

    def __len__(self):
        return len(self.morphemes)

    class Builder:

        def __init__(self, orig: 'Analysis' = None):
            if orig is not None:
                self._word = orig.word
                self._prefixes = orig.prefixes
                self._base = orig.base
                self._suffixes = orig.suffixes

            else:
                self._word = None
                self._prefixes = []
                self._base = None
                self._suffixes = []

        @property
        def word(self) -> str:
            return self._word

        @word.setter
        def word(self, value: str):
            self._word = value

        def word_is(self, value: str) -> 'Analysis.Builder':
            self.word = value
            return self

        @property
        def prefixes(self) -> list[conllx.Morpheme]:
            return self._prefixes

        @prefixes.setter
        def prefixes(self, value: list[conllx.Morpheme]):
            self._prefixes = value

        def prefixes_is(self, value: list[conllx.Morpheme]) -> 'Analysis.Builder':
            self.prefixes = value
            return self

        @property
        def base(self) -> conllx.Morpheme:
            return self._base

        @base.setter
        def base(self, value: conllx.Morpheme):
            self._base = value

        def base_is(self, value: conllx.Morpheme) -> 'Analysis.Builder':
            self.base = value
            return self

        @property
        def suffixes(self) -> list[conllx.Morpheme]:
            return self._suffixes

        @suffixes.setter
        def suffixes(self, value: list[conllx.Morpheme]):
            self._suffixes = value

        def suffixes_is(self, value: list[conllx.Morpheme]) -> 'Analysis.Builder':
            self.suffixes = value
            return self

        def build(self) -> 'Analysis':
            return Analysis(self.word, self.prefixes, self.base, self.suffixes)


class HebrewMorphAnalyzer:

    def __init__(self, preflex: pd.DataFrame, lex: pd.DataFrame):
        self.preflex, self.lex = preflex, lex

        # Prefix vocabulary
        self.preflex_vocab = set(preflex.index.get_level_values(0).values)

        # Prefix POS tags
        self.prefix_tags = set(preflex[preflex['cpostag'] != '_'].cpostag)

        # Lexical vocabulary
        self.lex_vocab = set(lex.index.get_level_values(0).values)

        # Lexical POS tags
        self.lex_tags = set(lex[lex['cpostag'] != '_'].cpostag)

        # Punctuation POS tags
        self.punct_tags = set(t for t in self.lex_tags if t.startswith('yy'))

        # Suffix POS tags
        self.suffix_tags = set(t for t in self.lex_tags if t.startswith('S_'))

        # Lexical POS tags
        self.lex_tags = self.lex_tags.difference(self.punct_tags).difference(self.suffix_tags)

        # Suffix mapping: break up suffixes into constituent morphemes
        self._suffix_lemmas = self._get_suffix_lemmas()
        self._suffix_morphemes = self._get_suffix_morphemes()

        # Default templates - used as default for OOV words
        self._default_noun_analyses = _prepare_default_analyses(['NN', 'NNT', 'NNP'])
        self._default_numeral_analyses = _prepare_default_analyses(['CD', 'CDT'])

        self._cache = {}

        pattern = r"^[+-]?((\d+(\.\d+)?)|(\.\d+))$"
        self.numeric_regexp = re.compile(pattern)

    # Get all possible morphological analyses of the given word according to the given lexicon
    def analyze_word(self, word: str) -> list[Analysis]:
        if word not in self._cache:
            lex_analyses = self._get_lex_entries(word, True)
            preflex_analyses = self._combine_prefixes(word)
            if preflex_analyses:
                valid_preflex_analyses = _filter_duplicate_prefixes(preflex_analyses, lex_analyses)
                lex_analyses.extend(valid_preflex_analyses)
            self._cache[word] = lex_analyses
        return self._cache[word]

    def expand_suffixes(self, analysis: Analysis) -> Analysis:
        expanded_analysis_builder = Analysis.Builder().word_is(analysis.word)
        prefixes = _to_conllx_prefixes(analysis)
        if prefixes:
            expanded_analysis_builder.prefixes_is(prefixes)
        base_builder = conllx.Morpheme.Builder(analysis.base)
        base_builder.fpostag_is(analysis.base.cpostag)
        if analysis.suffixes:
            expanded_analysis_builder.suffixes_is(self.expand_suffix(analysis.suffixes[0]))
        expanded_analysis_builder.base_is(base_builder.build())
        return expanded_analysis_builder.build()

    def expand_suffix(self, suffix: conllx.Morpheme) -> list[conllx.Morpheme]:
        feats = hebtagset.format_parsed_features(suffix.feats)
        return self._suffix_morphemes[suffix.cpostag.value].get(feats, [])

    def _get_suffix_lemmas(self) -> dict[str:str]:
        possessive = self.lex[self.lex.cpostag == 'POS'].copy()  # S_PP
        s_pp_lemma = possessive['lemma'].unique()[0]
        accusative = self.lex[self.lex.cpostag == 'AT'].copy()  # S_ANP
        s_anp_lemma = accusative['lemma'].unique()[0]
        pronouns = self.lex[self.lex.cpostag == 'PRP-PERS'].copy()  # S_PRN
        pronoun_lemmas = pronouns.groupby('lemma')['lemma']
        pronoun_lemmas = pronoun_lemmas.count().reset_index(name='count').sort_values('count', ascending=False)
        s_prn_lemma = pronoun_lemmas.iloc[0].lemma
        return {'S_PRN': s_prn_lemma, 'S_PP': s_pp_lemma, 'S_ANP': s_anp_lemma}

    # TODO: Map suffix tags to pronoun tags (S_PRN, S_PP, S_ANP -> PRP_PERS)?
    def _get_suffix_morphemes(self) -> dict[str:dict[str:list[conllx.Morpheme]]]:
        s_prn_morphemes = self._get_s_prn_morphemes()
        s_pp_morphemes = self._get_s_pp_morphemes(s_prn_morphemes)
        s_anp_morphemes = self._get_s_anp_morphemes(s_prn_morphemes)
        return {'S_PRN': {k: [s_prn_morphemes[k]] for k in s_prn_morphemes},
                'S_PP': {k: s_pp_morphemes[k].morphemes for k in s_pp_morphemes},
                'S_ANP': {k: s_anp_morphemes[k].morphemes for k in s_anp_morphemes}}

    def _extract_suffix_analysis(self, data: pd.DataFrame, prp: dict[str:conllx.Morpheme]) -> dict[str:Analysis]:
        extracted = defaultdict(set)
        for t in data.itertuples():
            word, i = t[0][:-1]
            analysis = self._get_lex_entries(word, False)[i]
            if not analysis.suffixes:
                continue

            # Extracted suffix key
            suffix = analysis.suffixes[0]
            feats = hebtagset.format_parsed_features(suffix.feats)

            # Construct extracted suffix value (cpostag, features ,lemma, form)
            prp_suffix_builder = conllx.Morpheme.Builder(suffix)
            prp_feats = hebtagset.format_parsed_features(suffix.feats)
            prp_suffix = prp[prp_feats]
            prp_suffix_builder.form_is(prp_suffix.form)
            prp_suffix_builder.lemma_is(prp_suffix.lemma)
            prp_suffix_builder.cpostag_is(prp_suffix.cpostag)
            prp_suffix_builder.fpostag_is(prp_suffix.fpostag)
            prp_suffix = prp_suffix_builder.build()
            prp_analysis = Analysis.Builder(analysis).suffixes_is([prp_suffix]).build()
            extracted[feats].add(prp_analysis)

        return {k: sorted(extracted[k], key=lambda x: len(x.base.form))[0] for k in extracted}

    def _get_s_pp_morphemes(self, prp: dict[str:conllx.Morpheme]) -> dict[str:Analysis]:
        possessives = self.lex[self.lex.cpostag == 'POS'].copy()
        possessives.loc[:, 'cpostag'] = 'S_PP'
        return self._extract_suffix_analysis(possessives, prp)

    def _get_s_anp_morphemes(self, prp: dict[str:conllx.Morpheme]) -> dict[str:Analysis]:
        accusatives = self.lex[self.lex.cpostag == 'AT'].copy()
        accusatives.loc[:, 'cpostag'] = 'S_ANP'
        return self._extract_suffix_analysis(accusatives, prp)

    def _get_s_prn_morphemes(self) -> dict[str:conllx.Morpheme]:
        s_prn_lemma = self._suffix_lemmas['S_PRN']
        pronouns = self.lex[(self.lex.cpostag == 'PRP-PERS') & (self.lex.lemma == s_prn_lemma)].copy()
        pronouns.loc[:, 'cpostag'] = 'S_PRN'
        pronouns.loc[:, 'fpostag'] = 'PRP-PERS'
        prp_morphemes = bgulex.data_to_morphemes(pronouns)
        extracted = defaultdict(set)
        for prp in prp_morphemes:
            feats = hebtagset.format_parsed_features(prp.feats)
            extracted[feats].add(prp)
        return {k: sorted(extracted[k], key=lambda x: len(x.form))[0] for k in extracted}

    def _get_lex_entries(self, word: str, with_nn_defaults: bool) -> list[Analysis]:
        data = _get_lexical_data_entry(word, self.lex_vocab, self.lex)
        lex_analyses = _lex_data_entry_to_analyses(word, data)
        if not lex_analyses:
            if self.is_numeral(word):
                lex_analyses = _get_default_analyses(word, self._default_numeral_analyses)
            elif with_nn_defaults:
                lex_analyses = _get_default_analyses(word, self._default_noun_analyses)
        return lex_analyses

    def _get_preflex_entries(self, prefix: str) -> list[Analysis]:
        data = _get_lexical_data_entry(prefix, self.preflex_vocab, self.preflex)
        return _preflex_data_to_analyses(data)

    # Break down the word into all possible prefixes and remainders
    def _combine_prefixes(self, word: str) -> list[Analysis]:
        analyses_combinations = []

        # Try to break the word down into prefix + remainder
        for i in range(1, len(word)):

            # If there is no prefix - we're done
            prefix = word[:i]
            preflex_analyses = self._get_preflex_entries(prefix)
            if len(preflex_analyses) == 0:
                break

            # If no remainder - skip it
            remainder = word[i:]
            remainder_analyses = self._get_lex_entries(remainder, False)
            if len(remainder_analyses) == 0:
                continue

            # Build different combinations of analyses
            for preflex_analysis in preflex_analyses:
                for remainder_analysis in remainder_analyses:

                    # Combine the prefix with the base and suffix remainder analysis
                    analysis_builder = Analysis.Builder(remainder_analysis)
                    analysis_builder.word_is(word)
                    analysis_builder.prefixes_is(preflex_analysis.prefixes)
                    analyses_combinations.append(analysis_builder.build())
        return analyses_combinations

    def is_numeral(self, word: str) -> bool:
        return bool(self.numeric_regexp.search(word))


def _get_default_analyses(word: str, defaults: list[Analysis]) -> list[Analysis]:
    analyses = []
    for a in defaults:
        # m = conllx.Morpheme.Builder(a.base).form_is(word).lemma_is(word).build()
        m = conllx.Morpheme.Builder(a.base).form_is(word).build()
        analyses.append(Analysis.Builder(a).word_is(word).base_is(m).build())
    return analyses


def _prepare_default_analyses(tags: list) -> list[Analysis]:
    analyses = []
    for tag in tags:
        tag = hebtagset.get_postag(tag)
        m = conllx.Morpheme.Builder().cpostag_is(tag).fpostag_is(tag).build()
        analyses.append(Analysis.Builder().base_is(m).build())
    return analyses


def _get_lexical_data_entry(word: str, vocab: set[str], lexicon: pd.DataFrame) -> pd.DataFrame:
    if word in vocab:
        return lexicon.loc[word].copy()
    return pd.DataFrame()


def _lex_data_entry_to_analyses(word: str, data: pd.DataFrame) -> list[Analysis]:
    analyses = []
    # Group by analysis index, level=0 is the word level (0=word, 1=analysis, 2=morpheme)
    for _, analysis_data in data.groupby(level=0):
        analysis = _lex_data_entry_to_analysis(word, analysis_data)
        analyses.append(analysis)
    return analyses


def _lex_data_entry_to_analysis(word: str, analysis_data: pd.DataFrame) -> Analysis:
    prefix, base, suffix = bgulex.data_to_morphemes(analysis_data)
    analysis_builder = Analysis.Builder().word_is(word).base_is(base)
    if prefix:
        analysis_builder.prefixes_is([prefix])
    if suffix:
        analysis_builder.suffixes_is([suffix])
    return analysis_builder.build()


def _to_conllx_prefixes(analysis: Analysis) -> list[conllx.Morpheme]:
    prefixes = []
    for prefix in analysis.prefixes:
        prefix_builder = conllx.Morpheme.Builder(prefix)
        prefix_builder.lemma_is(prefix.form)
        prefix_builder.fpostag_is(prefix.cpostag)
        prefixes.append(prefix_builder.build())
    return prefixes


def _to_conllx_base(analysis: Analysis) -> conllx.Morpheme:
    base = analysis.base
    base_builder = conllx.Morpheme.Builder(base)
    if analysis.suffixes:
        suffix = analysis.suffixes[0]
        base_builder.fpostag_is(suffix.cpostag)
        feats = {f: base.feats[f] for f in base.feats}
        for f in suffix.feats:
            feats[f'suf_{f}'] = suffix.feats[f]
        base_builder.feats_is(feats)
    else:
        base_builder.fpostag_is(base.cpostag)
    return base_builder.build()


def _to_conllx_analysis(analysis: Analysis) -> Analysis:
    analysis_builder = Analysis.Builder().word_is(analysis.word)
    prefixes = _to_conllx_prefixes(analysis)
    if prefixes:
        analysis_builder.prefixes_is(prefixes)
    analysis_builder.base_is(_to_conllx_base(analysis))
    return analysis_builder.build()


def _preflex_data_to_analyses(data: pd.DataFrame) -> list[Analysis]:
    analyses = []
    # Group by analysis index, level=0 is the word level (1=analysis, 2=morpheme)
    for _, pref_data in data.groupby(level=0):
        prefixes = bgulex.data_to_morphemes(pref_data)
        analysis = Analysis.Builder().prefixes_is(prefixes).build()
        analyses.append(analysis)
    return analyses


def _filter_duplicate_prefixes(preflex_combinations: list[Analysis], lex_analyses: list[Analysis]) -> list[Analysis]:
    filtered = []
    lex_analyses_with_prefix = {b.prefixes[0].form: b for b in lex_analyses if len(b.prefixes) > 0}
    for a in preflex_combinations:
        if ''.join([p.form for p in a.prefixes]) in lex_analyses_with_prefix:
            continue
        filtered.append(a)
    return filtered


def _read_word(p: Path) -> str:
    with p.open() as f:
        return f.readline()


# Transform list of analyses into conllx format
def _conllx_format_analyses(analyses: list[Analysis]) -> list[list[str]]:
    conllx_analyses = [_to_conllx_analysis(analysis) for analysis in analyses]
    sorted_analyses = list(reversed(sorted(conllx_analyses, key=len)))
    max_len = len(sorted_analyses[0])

    # Construct lattice nodes
    # A node represents a partial sequence of surface forms consumed in the path leading to it
    nodes = {}
    for analysis in sorted_analyses:
        forms = []
        if tuple(forms) not in nodes:
            nodes[tuple(forms)] = len(nodes)
        for form in analysis.forms[:-1]:
            forms.append(form)
            if tuple(forms) not in nodes:
                nodes[tuple(forms)] = len(nodes)
        forms.append(analysis.forms[-1])
        if tuple(forms) not in nodes:
            nodes[tuple(forms)] = max_len

    # Construct lattice edges
    # An edge connects a pair of nodes labeled by the morpheme being consumed in the path
    edges = defaultdict(set)
    for analysis in sorted_analyses:
        forms = []
        from_node = nodes[tuple(forms)]
        for m in analysis.morphemes:
            forms.append(m.form)
            to_node = nodes[tuple(forms)]
            edge = (from_node, to_node, m.form)
            edges[edge].add(m)
            from_node = to_node

    return [[str(v) for v in edge[:-1]] + hebtb.format_morpheme(m) for edge in edges for m in edges[edge]]


if __name__ == '__main__':
    root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    heb_ma = HebrewMorphAnalyzer(*(bgulex.load(root_path)))
    word_to_analyze = _read_word(Path('word.txt'))
    word_analyses = heb_ma.analyze_word(word_to_analyze)
    lines = ['\t'.join(a) for a in _conllx_format_analyses(word_analyses)]
    print('\n'.join(lines))
