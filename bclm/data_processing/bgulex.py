import pandas as pd
from collections import defaultdict
from bclm.data_processing import hebtb, heb_tagset
from bclm.data_processing.format import conllx
from pathlib import Path
import random

# The list of Lexical categories is found in Yoav's thesis (Table 4.4):
# https://www.cs.bgu.ac.il/~elhadad/nlpproj/pub/yoav-phd.pdf


# Get the first value as the POS tag and return the rest of the features
# Handle no tag cases, e.g.: '', -F-S-2
# Handle multi-value tags cases, e.g. PRP-DEM, PRP-PERS, PRP-IMP, PRP-REF, CC-COORD, CC-REL, CC-SUB
def _extract_postag(feats_str: str) -> (str, list[str]):
    feats = feats_str.split('-')

    postag_index = 0
    postag = '-'.join(feats[:postag_index + 1])

    # Handle multi-value POS tags
    # Look and see if this is the case (i.e. join 2 or more values as the POS tag)
    while postag_index < len(feats) and postag in heb_tagset.postag_values:
        if postag_index > 0:
            print(f'multipart POS tag in {feats_str}: {postag}')
        postag_index += 1
        postag = '-'.join(feats[:postag_index + 1])

    # No tag cases, e.g.: '', -MF-P-3, -F-S-2, -MF-P-2, -M-S-2, -F-S-3, -MF-S-1, -M-P-2, -F-P-2
    if postag_index == 0:
        print(f'missing POS tag in {feats_str}')
        return None, feats[postag_index + 1:]

    # Use the last postag index
    postag = '-'.join(feats[:postag_index])
    return postag, feats[postag_index:]


# Each morpheme is represented as a '-' seperated list of values (pos tag, morph features)
def _parse_lex_feats(feat_str: str) -> dict:
    result = {}

    # The prefix part of the analysis could be (and often is) empty
    if len(feat_str) == 0:
        return result
    # feats = feat_str.split('-')

    # Get POS tag and the rest of the features
    postag, feats = _extract_postag(feat_str)
    result['pos'] = postag
    for feat in feats:

        # Special case treatment - Personal Pronouns
        # PRP types: Demonstrative, Personal, Impersonal, Reference
        # Sometime the PRP type (DEM, PERS, IMP, REF) is not positioned after the PRP pos tag (this case is handled by
        # _extract_postag) but further down the feat_str
        # In this case add this value to the POS tag:
        if feat == 'DEM' or feat == 'PERS' or feat == 'IMP' or feat == 'REF':
            postag = f"{result['pos']}-{feat}"
            if postag in heb_tagset.postag_values:
                result['pos'] = postag
                continue

        # Short version lexicon feature values, e.g. M, F, MF, S, 1, 2, A, etc.
        # Map the lexicon-style feature values into conll features
        feat_name = heb_tagset.get_morph_feature_name_by_value(feat)
        feature = heb_tagset.get_morph_feature_by_name(feat_name)
        result[feat_name] = feature.get_value(feat).value

    return result


# Parse suffixes
def _parse_suffix(feat_str: str, base: dict) -> dict:
    # Handle Special case: S_PP -> NN_S_PP, CD_N_PP, BN_N_PP
    if 'S_PP' in feat_str:
        feat_str = feat_str.replace('S_PP', f"{base['pos']}_S_PP")
    return _parse_lex_feats(feat_str)


# Build a morpheme
def _to_morpheme(form: str, lemma: str = None, postag: str = None, feats: dict = None) -> conllx.Morpheme:
    builder = conllx.Morpheme.Builder(form=form)
    if lemma and lemma != 'unspecified':
        builder.lemma_is(lemma)
    if postag:
        builder.cpostag_is(postag)
    if feats:
        builder.feats_is(feats)
    return builder.build()


# A preflex entry is formatted differently from the base lexicon
# A prefix entry can actually be made up of several prefixes (e.g. and when -> vekshe) seperated by a '+'
def _parse_preflex_feat(feat_str: str) -> list:
    return feat_str.split('+')


# Each analysis is represented as a ':' seperated list of morphemes (prefix, base, suffix)
def _parse_features(feats_str: str) -> (dict, dict, dict):
    parts = feats_str.split(':')
    pref = _parse_lex_feats(parts[0])
    base = _parse_lex_feats(parts[1])
    if len(base) == 0:
        print(f'WARNING: base should not be empty: {feats_str}')
    suff = _parse_suffix(parts[2], base)
    return pref, base, suff


# Preflex entries are also represented as a ':' seperated list of morphemes except that the list always contains
# just a single part
def _parse_preflex_features(feats_str: str) -> list:
    return _parse_preflex_feat(feats_str.split(':')[0])


# Each entry in the lexicon is formatted as a word form followed by a list of possible analyses
# Each analysis is represented as a ':' seperated list of morphemes (prefix, base, suffix)
# Each morpheme is represented as a '-' seperated list of values (lemma, POS tag, features)
def _parse_lex_entry(entry: str) -> (str, list[list[conllx.Morpheme]]):
    analyses = []
    entry_parts = entry.split(' ')
    form = entry_parts[0]
    lemmas = entry_parts[2::2]
    feats = entry_parts[1::2]
    for lemma, feat in zip(lemmas, feats):
        analysis = []
        pref, base, suff = _parse_features(feat)
        postag = pref.pop('pos', None)
        analysis.append(_to_morpheme(form, None, postag, pref))  # prefix
        postag = base.pop('pos', None)
        analysis.append(_to_morpheme(form, lemma, postag, base))  # base
        postag = suff.pop('pos', None)
        analysis.append(_to_morpheme(form, None, postag, suff))  # suffix
        analyses.append(analysis)
    return form, analyses


# A preflex entry is formatted differently from the base lexicon
# A prefix entry can actually be made up of several prefixes (e.g. and when -> vekshe)
# The word forms are seperated by a '^' and the POS tags are seperated by a '+'
def _parse_preflex_entry(entry: str) -> (str, list):
    analyses = []
    pref_entry_parts = entry.split(' ')
    prefix = pref_entry_parts[0]
    feat_seqs = pref_entry_parts[2::2]
    form_seqs = pref_entry_parts[1::2]
    builder = conllx.Morpheme.Builder(prefix)
    for forms_str, feats_str in zip(form_seqs, feat_seqs):
        analysis = []
        postags = _parse_preflex_features(feats_str)
        forms = forms_str.split('^')
        for form, postag in zip(forms, postags):
            builder.form_is(form)
            builder.cpostag_is(postag)
            analysis.append(builder.build())
        analyses.append(analysis)
    return prefix, analyses


# Normalize POS tags and morph features (turning lexicon style short feature values into treebank conll style values)
def _norm_morpheme(morpheme: conllx.Morpheme) -> conllx.Morpheme:
    return conllx.Morpheme.Builder()\
        .form_is(morpheme.form)\
        .lemma_is(None if not morpheme.lemma else morpheme.lemma)\
        .cpostag_is(None if not morpheme.cpostag else heb_tagset.get_postag(morpheme.cpostag)) \
        .fpostag_is(None if not morpheme.fpostag else heb_tagset.get_postag(morpheme.fpostag)) \
        .feats_is(None if not morpheme.feats else heb_tagset.parse_features(morpheme.feats))\
        .build()


# Normalize morpheme feature values and format as conll rows
def _format_morpheme(morpheme: conllx.Morpheme) -> list[str]:
    parsed_morpheme = _norm_morpheme(morpheme)
    return hebtb.format_morpheme(parsed_morpheme)


# Transform single lexical analysis (prefix, base, suffix) into conll format
def _format_lex_analysis(word: str, analysis_index: int, analysis: list[conllx.Morpheme]) -> list[list]:
    result = []
    prefix = analysis[0]
    base = analysis[1]
    suffix = analysis[2]
    result.append([word, analysis_index, 0] + _format_morpheme(prefix))
    result.append([word, analysis_index, 1] + _format_morpheme(base))
    result.append([word, analysis_index, 2] + _format_morpheme(suffix))
    return result


# Transform single prexlex analysis into conll format
def _format_preflex_analysis(prefix: str, analysis_index: int, analysis: list[conllx.Morpheme]) -> list[list]:
    return [[prefix, analysis_index, i] + _format_morpheme(morpheme) for i, morpheme in enumerate(analysis)]


# Transform all lexical entries into conll format
def _format_lex_analyses(word: str, analyses: list[list[conllx.Morpheme]]) -> list[list]:
    return [row for i, analysis in enumerate(analyses) for row in _format_lex_analysis(word, i, analysis)]


# Transform all prefix lexical entries into conll format
def _format_preflex_analyses(prefix: str, analyses: list[list[conllx.Morpheme]]) -> list[list]:
    return [row for i, analysis in enumerate(analyses) for row in _format_preflex_analysis(prefix, i, analysis)]


BGULEX_COLUMNS = ['word', 'analysis', 'morpheme', 'form', 'lemma', 'cpostag', 'fpostag', 'feats']


# Turn lexical entries into conll format dataframe indexed to make it easy to search for word analyses.
# Used to map words into all possible analyses and embed words as well as OOV words that are missing
# from the embedding vocabulary by falling back on the lemmas
def _to_lex_dataframe(entries: dict) -> pd.DataFrame:
    rows = [row for word in entries for row in _format_lex_analyses(word, entries[word])]
    return pd.DataFrame(rows, columns=BGULEX_COLUMNS).set_index(['word', 'analysis', 'morpheme'])


# Turn preflex entries into a conll format dataframe indexed to make it easy to search for word analyses
# Note the dataframe index is hierarchical and in our case is made up of 3 levels: word, word + analysis,
# word + analysis + morpheme. This should make it easy to get the analyses of each word in O(1)
def _to_preflex_dataframe(entries: dict) -> pd.DataFrame:
    rows = [row for word in entries for row in _format_preflex_analyses(word, entries[word])]
    return pd.DataFrame(rows, columns=BGULEX_COLUMNS).set_index(['word', 'analysis', 'morpheme'])


# Save lexicon as a conll format dataframe
def _save_interm_lex(entries: dict, dest_file_path: Path):
    dest_dir = dest_file_path.parent.absolute()
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    lex_df = _to_lex_dataframe(entries)
    lex_df.to_csv(dest_file_path)


# Save preflex as a conll format dataframe
def _save_interm_preflex(entries: dict, dest_file_path: Path):
    dest_dir = dest_file_path.parent.absolute()
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    lex_df = _to_preflex_dataframe(entries)
    lex_df.to_csv(dest_file_path)


# Save lexical dataframe
def save_interm_lexicon(preflex_entries: dict, dest_preflex_file_path: Path,
                        lex_entries: dict, dest_lex_file_path: Path):
    _save_interm_preflex(preflex_entries, dest_preflex_file_path)
    _save_interm_lex(lex_entries, dest_lex_file_path)


# Load lexical dataframe
def load_interm_lexicon(preflex_src_file_path: Path, lex_src_file_path: Path) -> (pd.DataFrame, pd.DataFrame):
    preflex_df = pd.read_csv(preflex_src_file_path, index_col=['word', 'analysis', 'morpheme'])
    lex_df = pd.read_csv(lex_src_file_path, index_col=['word', 'analysis', 'morpheme'])
    return preflex_df, lex_df


# Parse the preflex and lex files
def load_raw_lexicon(preflex_file_path: Path = None, lex_file_path: Path = None) -> (dict, dict):
    preflex_entries, lex_entries = {}, {}
    if preflex_file_path:
        with open(preflex_file_path) as f:
            lines = [line.strip() for line in f.readlines()]
        for line in lines:
            prefix, analyses = _parse_preflex_entry(line)
            preflex_entries[prefix] = analyses
    if lex_file_path:
        with open(lex_file_path) as f:
            lines = [line.strip() for line in f.readlines()]
        for line in lines:
            form, analyses = _parse_lex_entry(line)
            lex_entries[form] = analyses
    return preflex_entries, lex_entries


def process():
    raw_root_path = Path('data/raw/HebrewResources/HebrewTreebank')
    interim_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    # processed_root_path = Path('data/processed/HebrewResources/HebrewTreebank')
    lex_name = 'bgulex'
    preflex_file_path = raw_root_path / lex_name / 'bgupreflex_withdef.utf8.hr'
    lex_file_path = raw_root_path / lex_name / 'bgulex.utf8.hr'
    preflex_entries, lex_entries = load_raw_lexicon(preflex_file_path=preflex_file_path, lex_file_path=lex_file_path)
    print(len(preflex_entries))
    print(len(lex_entries))
    preflex_file_path = interim_root_path / lex_name / 'bgupreflex.csv'
    lex_file_path = interim_root_path / lex_name / 'bgulex.csv'
    save_interm_lexicon(preflex_entries, preflex_file_path, lex_entries, lex_file_path)
    for _ in range(10):
        form = random.choice(list(preflex_entries.keys()))
        for a in preflex_entries[form]:
            for m in a:
                print(_format_morpheme(m))
        form = random.choice(list(lex_entries.keys()))
        for a in lex_entries[form]:
            for m in a:
                print(_format_morpheme(m))


def _get_analysis(analysis_data: list[tuple]) -> list[conllx.Morpheme]:
    analysis = []
    pref = conllx.Morpheme.parse(list(analysis_data[0]))

    # Parse raw string values into enum values
    pref = _norm_morpheme(pref)

    # Filter out empty prefixes
    if pref.cpostag is not None:
        analysis.append(pref)

    base = conllx.Morpheme.parse(list(analysis_data[1]))
    suff = conllx.Morpheme.parse(list(analysis_data[2]))
    feats = base.feats
    if suff.feats:
        for f in suff.feats:
            feats[f'suf_{f}'] = suff.feats[f]
    base = conllx.Morpheme.Builder(base.form)\
        .lemma_is(base.lemma)\
        .cpostag_is(base.cpostag)\
        .fpostag_is(suff.cpostag)\
        .feats_is(feats).build()

    # Parse raw string values into enum values
    base = _norm_morpheme(base)

    analysis.append(base)
    # analysis.append(suff)
    return analysis


def _get_lex_entries(lex_df: pd.DataFrame, word: str) -> list[list[conllx.Morpheme]]:

    # Verify that the word is valid (with a lex entry), if not - skip it
    try:
        analyses_data = lex_df.loc[word]
    except KeyError:
        return []

    analyses = []
    # Group by analysis index
    for _, analysis_data in analyses_data.groupby(level=0):
        analysis = _get_analysis([morpheme for _, morpheme in analysis_data.iterrows()])
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
        prefixes = []
        for _, pref in pref_data.iterrows():
            pref = conllx.Morpheme.parse(list(pref))

            # Parse raw string values into enum values
            pref = _norm_morpheme(pref)

            prefixes.append(pref)
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


# Get all possible morphological analyses of the given word according to the given lexicon
def _analyze(lex_df: pd.DataFrame, preflex_df: pd.DataFrame, word: str) -> list[list[conllx.Morpheme]]:
    analyses = _get_lex_entries(lex_df, word)
    analyses.extend(_analyze_combine(lex_df, preflex_df, word))
    return analyses


def _load_word(p: Path) -> str:
    with p.open() as f:
        return f.readline()


# The bgu MA is based on the preflex and lex data
def _load_analyzer() -> (pd.DataFrame, pd.DataFrame):
    interim_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    lex_name = 'bgulex'
    preflex_file_path = interim_root_path / lex_name / 'bgupreflex.csv'
    lex_file_path = interim_root_path / lex_name / 'bgulex.csv'
    return load_interm_lexicon(preflex_file_path, lex_file_path)


# Return the lattice format of the analyses
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


def ma():
    preflex_df, lex_df = _load_analyzer()
    word = _load_word(Path('word.txt'))
    analyses = _analyze(lex_df, preflex_df, word)
    lines = ['\t'.join(a) for a in _format_analyses(analyses)]
    print('\n'.join(lines))


if __name__ == '__main__':
    # process()
    ma()
