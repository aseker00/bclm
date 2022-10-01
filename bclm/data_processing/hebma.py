import pandas as pd
from collections import defaultdict
from bclm.data_processing import hebtb, bgulex
from bclm.data_processing.format import conllx
from pathlib import Path


class HebrewMorphAnalyzer:

    def __init__(self, root_path: Path):
        self.preflex_df, self.lex_df = bgulex.load(root_path)

    # Get all possible morphological analyses of the given word according to the given lexicon
    def analyze(self, word: str) -> list[list[conllx.Morpheme]]:
        lex_analyses = _get_lex_entries(self.lex_df, word)
        preflex_analyses = _analyze_combine(self.lex_df, self.preflex_df, word)
        valid_preflex_analyses = _remove_redundant_combinations(preflex_analyses, lex_analyses)
        lex_analyses.extend(valid_preflex_analyses)
        return lex_analyses


def _lex_data_to_conll_analysis(analysis_data: pd.DataFrame) -> list[conllx.Morpheme]:
    pref, base, suff = bgulex.data_to_morphemes(analysis_data)
    builder = conllx.Morpheme.Builder(base).fpostag_is(suff.cpostag)
    feats = base.feats
    if suff.feats:
        for f in suff.feats:
            feats[f'suf_{f}'] = suff.feats[f]
        builder.feats_is(feats)
    base = builder.build()

    # Filter out empty prefixes
    if pref.cpostag is not None:
        return [pref, base]
    return [base]


def _get_lex_entries(lex_df: pd.DataFrame, word: str) -> list[list[conllx.Morpheme]]:

    # Verify that the word is valid (with a lex entry), if not - skip it
    try:
        analyses_data = lex_df.loc[word]
    except KeyError:
        return []

    analyses = []
    # Group by analysis index
    for _, analysis_data in analyses_data.groupby(level=0):
        analysis = _lex_data_to_conll_analysis(analysis_data)
        analyses.append(analysis)
    return analyses


def _get_preflex_entries(preflex_df: pd.DataFrame, prefix: str) -> list[list[conllx.Morpheme]]:

    # Verify that the prefix is valid (with a preflex entry), if not - skip it
    try:
        prefixes_data = preflex_df.loc[prefix]
    except KeyError:
        return []

    analyses = []
    # Group by analysis index
    for _, pref_data in prefixes_data.groupby(level=0):
        prefixes = bgulex.data_to_morphemes(pref_data)
        analyses.append(prefixes)
    return analyses


# Break down the word into all possible prefixes and reminders
def _analyze_combine(lex_df: pd.DataFrame, preflex_df: pd.DataFrame, word: str) -> list[list[conllx.Morpheme]]:
    analyses_combinations = []

    # Try to break the word down into prefix + remainder
    for i in range(1, len(word)):
        prefix = word[:i]
        remainder = word[i:]

        # If the remainder or prefixes are empty - skip this one
        remainder_analyses = _get_lex_entries(lex_df, remainder)
        if len(remainder_analyses) == 0:
            continue
        preflex_analyses = _get_preflex_entries(preflex_df, prefix)
        if len(preflex_analyses) == 0:
            continue

        # Build different combinations of analyses
        for preflex_analysis in preflex_analyses:
            for remainder_analysis in remainder_analyses:
                # Combine the prefix with the base and suffix remainder analysis
                analysis = preflex_analysis + remainder_analysis
                analyses_combinations.append(analysis)
    return analyses_combinations


def _remove_redundant_combinations(preflex_combinations: list[list[conllx.Morpheme]],
                                   lex_analyses: list[list[conllx.Morpheme]]):
    valid_combinations = []
    lex_analyses_with_prefix = {b[0].form: b for b in lex_analyses if b[0].cpostag is not None}
    for a in preflex_combinations:
        if a[0].form in lex_analyses_with_prefix:
            continue
        valid_combinations.append(a)
    return valid_combinations


def _load_word(p: Path) -> str:
    with p.open() as f:
        return f.readline()


# Transform list of analyses into conllx format
def _format_analyses(analyses: list[list[conllx.Morpheme]]) -> list[list[str]]:
    sorted_analyses = list(reversed(sorted(analyses, key=len)))
    max_len = len(sorted_analyses[0])

    # Construct lattice nodes
    # A node represents a partial sequence of surface forms consumed in the path leading to it
    nodes = {}
    for analysis in sorted_analyses:
        forms = []
        if tuple(forms) not in nodes:
            nodes[tuple(forms)] = len(nodes)
        for m in analysis[:-1]:
            forms.append(m.form)
            if tuple(forms) not in nodes:
                nodes[tuple(forms)] = len(nodes)
        forms.append(analysis[-1].form)
        if tuple(forms) not in nodes:
            nodes[tuple(forms)] = max_len

    # Construct lattice edges
    # An edge connects a pair of nodes labeled by the morpheme being consumed in the path
    edges = defaultdict(set)
    for analysis in sorted_analyses:
        forms = []
        from_node = nodes[tuple(forms)]
        for m in analysis:
            forms.append(m.form)
            to_node = nodes[tuple(forms)]
            edge = (from_node, to_node, m.form)
            edges[edge].add(m)
            from_node = to_node

    return [[str(v) for v in edge[:-1]] + hebtb.format_morpheme(m) for edge in edges for m in edges[edge]]


if __name__ == '__main__':
    lexicon_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    ma = HebrewMorphAnalyzer(lexicon_root_path)
    word_to_analyze = _load_word(Path('word.txt'))
    word_analyses = ma.analyze(word_to_analyze)
    lines = ['\t'.join(a) for a in _format_analyses(word_analyses)]
    print('\n'.join(lines))
