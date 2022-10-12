import pandas as pd
from bclm.data_processing import hebtb, hebtagset
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

    # Handle multi-value POS tags, e.g. PRP-PERS, CC-COORD etc.
    # Look and see if this is the case (i.e. join 2 or more values as the POS tag)
    while postag_index < len(feats) and postag in hebtagset.postag_values:
        # if postag_index > 0:
        #     print(f'multipart POS tag in {feats_str}: {postag}')
        postag_index += 1
        postag = '-'.join(feats[:postag_index + 1])

    # No tag cases, e.g.: '', -MF-P-3, -F-S-2, -MF-P-2, -M-S-2, -F-S-3, -MF-S-1, -M-P-2, -F-P-2
    # I fixed these in the raw lexicon file so this should not happen
    if postag_index == 0:
        print(f'WARNING: missing POS tag in {feats_str}')
        return None, feats[postag_index + 1:]

    # Use the last valid postag index
    postag = '-'.join(feats[:postag_index])
    return postag, feats[postag_index:]


# Each morpheme is represented as a '-' seperated list of values (pos tag, morph features)
def _parse_lex_feats(feat_str: str) -> dict:
    result = {}

    # The prefix part of the analysis could be (and often is) empty
    if len(feat_str) == 0:
        return result

    # Get POS tag and the rest of the features
    postag, feats = _extract_postag(feat_str)
    result['pos'] = postag
    for feat in feats:

        # Special case treatment - Personal Pronouns
        # PRP types: Demonstrative, Personal, Impersonal, Reference
        # Sometime the PRP type (DEM, PERS, IMP, REF) is not positioned directly after the PRP pos tag (handled by
        # _extract_postag) but further down the feat_str.
        # Here we handle this case where the PRP type value is further down the feat string
        if feat == 'DEM' or feat == 'PERS' or feat == 'IMP' or feat == 'REF':
            postag = f"{result['pos']}-{feat}"
            if postag in hebtagset.postag_values:
                result['pos'] = postag
                continue

        # Short version lexicon feature values, e.g. M, F, MF, S, 1, 2, A, etc.
        # Map the lexicon-style feature values into conll features
        feat_name = hebtagset.get_morph_feature_name_by_value(feat)
        feature = hebtagset.get_morph_feature_by_name(feat_name)
        result[feat_name] = feature.get_value(feat).value

    return result


# Build a morpheme
def _create_morpheme(form: str = None, lemma: str = None, postag: str = None, feats: dict = None) -> conllx.Morpheme:
    builder = conllx.Morpheme.Builder()
    if form:
        builder.form_is(form)
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
    return (_parse_lex_feats(p) for p in parts)


# Preflex entries are also represented as a ':' seperated list of morphemes except that the list always contains
# just a single part
def _parse_preflex_features(feats_str: str) -> list:
    return _parse_preflex_feat(feats_str.split(':')[0])


# Each entry in the lexicon is formatted as a word form followed by a list of possible analyses
# Each analysis is represented as a ':' seperated list of morphemes (prefix, base, suffix)
# Each morpheme is represented as a '-' seperated list of values (lemma, POS tag, features)
def _parse_lex_entry(entry: str) -> (str, list[tuple[conllx.Morpheme]]):
    analyses = set()
    entry_parts = entry.split(' ')
    word = entry_parts[0]
    lemmas = entry_parts[2::2]
    feats = entry_parts[1::2]
    for lemma, feat in zip(lemmas, feats):
        analysis = []
        prefix, base, suffix = _parse_features(feat)
        if len(base) == 0:
            print(f'WARNING: lex entry base cannot be empty: {entry}')
            continue
        postag = prefix.pop('pos', None)
        analysis.append(_create_morpheme(postag=postag, feats=prefix))
        postag = base.pop('pos', None)
        analysis.append(_create_morpheme(word, lemma, postag, base))
        postag = suffix.pop('pos', None)
        analysis.append(_create_morpheme(postag=postag, feats=suffix))
        analyses.add(tuple(analysis))
    return word, list(analyses)


# Prefixed lexical analysis is one where the lexical entry has a prefix and in addition you can find the remainder
# in the lexicon which means that the entry can be constructed from another lexical entry and a preflex entry
#
# Most lexical entries do not involve a prefix.
# The exception are entries that have the DEF prefix, these require special treatment to set the prefix form (using
# the first word character) and update the base form (removing the prefix character)
def _get_prefixed_lex_entries(preflex_entries: dict[str:list], lex_entries: dict[str:list]) -> dict[str:tuple]:
    prefixed_analyses = {}
    for word in lex_entries:
        for i, a in enumerate(lex_entries[word]):
            if a[0].cpostag:
                if a[0].cpostag != 'DEF':
                    print(f'WARNING: unexpected lex entry prefix: {word}: {a[0].cpostag}')
                    continue
                pref_form, base_form = word[0], word[1:]
                updated_analysis = []
                if pref_form in preflex_entries:
                    if base_form in lex_entries:
                        pref_update = conllx.Morpheme.Builder(a[0]).form_is(pref_form).build()
                        base_update = conllx.Morpheme.Builder(a[1]).form_is(base_form).build()
                        updated_analysis = [pref_update, base_update, a[2]]
                if len(updated_analysis) > 0:
                    prefixed_analyses[word] = (i, updated_analysis)
    return prefixed_analyses


# Some lexical entries have prefixes - the DEF prefix.
# In this we need to set the prefix form and update the base form
def _fix_lex_prefixes(preflex_entries: dict[str:list], lex_entries: dict[str:list]):
    prefixed_analyses = _get_prefixed_lex_entries(preflex_entries, lex_entries)
    print(f'Fixing {len(prefixed_analyses)} prefixed lexical analyses')
    for word in prefixed_analyses:
        (i, a) = prefixed_analyses[word]
        lex_entries[word][i] = a


# A preflex entry is formatted differently from the base lexicon
# A prefix entry can actually be made up of several prefixes (e.g. and when -> vekshe)
# The word forms are seperated by a '^' and the POS tags are seperated by a '+'
def _parse_preflex_entry(entry: str) -> (str, list[tuple[conllx.Morpheme]]):
    analyses = set()
    pref_entry_parts = entry.split(' ')
    prefix = pref_entry_parts[0]
    feat_seqs = pref_entry_parts[2::2]
    form_seqs = pref_entry_parts[1::2]
    builder = conllx.Morpheme.Builder()
    builder.form_is(prefix)
    for forms_str, feats_str in zip(form_seqs, feat_seqs):
        analysis = []
        postags = _parse_preflex_features(feats_str)
        forms = forms_str.split('^')
        for form, postag in zip(forms, postags):
            builder.form_is(form)
            builder.cpostag_is(postag)
            analysis.append(builder.build())
        analyses.add(tuple(analysis))
    return prefix, list(analyses)


# Normalize POS tags and morph features (turning lexicon style short feature values into treebank conll style values)
def _norm_morpheme(morpheme: conllx.Morpheme) -> conllx.Morpheme:
    return conllx.Morpheme.Builder() \
        .form_is(morpheme.form)\
        .lemma_is(morpheme.lemma)\
        .cpostag_is(None if not morpheme.cpostag else hebtagset.get_postag(morpheme.cpostag)) \
        .fpostag_is(None if not morpheme.fpostag else hebtagset.get_postag(morpheme.fpostag)) \
        .feats_is(None if not morpheme.feats else hebtagset.parse_features(morpheme.feats))\
        .build()


# Normalize morpheme feature values and format as conll rows
def _format_morpheme(morpheme: conllx.Morpheme) -> list[str]:
    parsed_morpheme = _norm_morpheme(morpheme)
    return hebtb.format_morpheme(parsed_morpheme)


# Transform single lexical analysis (prefix, base, suffix) into conll format
def _format_lex_analysis(word: str, analysis_index: int, analysis: list[conllx.Morpheme]) -> list[list]:
    result = []
    prefix, base, suffix = analysis
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
def _lex_entries_to_dataframe(entries: dict[str:list]) -> pd.DataFrame:
    rows = [row for word in entries for row in _format_lex_analyses(word, entries[word])]
    return pd.DataFrame(rows, columns=BGULEX_COLUMNS).set_index(['word', 'analysis', 'morpheme'])


# Turn preflex entries into a conll format dataframe indexed to make it easy to search for word analyses
# Note the dataframe index is hierarchical and in our case is made up of 3 levels: word, word + analysis,
# word + analysis + morpheme. This should make it easy to get the analyses of each word in O(1)
def _preflex_entries_to_dataframe(entries: dict) -> pd.DataFrame:
    rows = [row for word in entries for row in _format_preflex_analyses(word, entries[word])]
    return pd.DataFrame(rows, columns=BGULEX_COLUMNS).set_index(['word', 'analysis', 'morpheme'])


# Save lexicon as a conll format dataframe
def _save_interm_lex(entries: dict[str:list], dest_file_path: Path):
    dest_dir = dest_file_path.parent.absolute()
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    df = _lex_entries_to_dataframe(entries)
    df.to_csv(dest_file_path)


# Save preflex as a conll format dataframe
def _save_interm_preflex(entries: dict, dest_file_path: Path):
    dest_dir = dest_file_path.parent.absolute()
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    df = _preflex_entries_to_dataframe(entries)
    df.to_csv(dest_file_path)


# Save lexical dataframe
def _save_interm_lexicon(preflex_entries: dict, dest_preflex_file_path: Path,
                         lex_entries: dict, dest_lex_file_path: Path,
                         nikud_entries: dict, dest_nikud_file_path: Path):
    _save_interm_preflex(preflex_entries, dest_preflex_file_path)
    _save_interm_lex(lex_entries, dest_lex_file_path)
    _save_interm_lex(nikud_entries, dest_nikud_file_path)


# Load lexical dataframe
def _load_interm_lexicon(preflex_file_path: Path, lex_file_path: Path, nikud_file_path: Path) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    preflex_df = pd.read_csv(preflex_file_path, index_col=['word', 'analysis', 'morpheme']).sort_index()
    lex_df = pd.read_csv(lex_file_path, index_col=['word', 'analysis', 'morpheme']).sort_index()
    nikud_df = pd.read_csv(nikud_file_path, index_col=['word', 'analysis', 'morpheme']).sort_index()
    return preflex_df, lex_df, nikud_df


# Parse the preflex and lex files
def _load_raw_lexicon(preflex_file_path: Path = None, lex_file_path: Path = None) -> (dict[str:list], dict[str:list]):
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
            word, analyses = _parse_lex_entry(line)
            lex_entries[word] = analyses
    _fix_lex_prefixes(preflex_entries, lex_entries)
    return preflex_entries, lex_entries


def _load_nikud() -> dict[str:list]:
    nikud = {
        ':': hebtagset.POSTag.POSTag_yyCLN,
        ',': hebtagset.POSTag.POSTag_yyCM,
        '-': hebtagset.POSTag.POSTag_yyDASH,
        '.': hebtagset.POSTag.POSTag_yyDOT,
        '...': hebtagset.POSTag.POSTag_yyELPS,
        '!': hebtagset.POSTag.POSTag_yyEXCL,
        '(': hebtagset.POSTag.POSTag_yyLRB,
        '?': hebtagset.POSTag.POSTag_yyQM,
        '"': hebtagset.POSTag.POSTag_yyQUOT,
        ')': hebtagset.POSTag.POSTag_yyRRB,
        ';': hebtagset.POSTag.POSTag_yySCLN
    }
    m = conllx.Morpheme.Builder().build()
    entries = {}
    for form in nikud:
        base_builder = conllx.Morpheme.Builder()
        base_builder = base_builder.form_is(form)
        base_builder = base_builder.lemma_is(form)
        base_builder = base_builder.cpostag_is(nikud[form].value)
        base_builder = base_builder.fpostag_is(nikud[form].value)
        entries[form] = [[m, base_builder.build(), m]]
    return entries


# Parse raw string values in dataframe into conllx enum values
def data_to_morphemes(data: pd.DataFrame) -> list[conllx.Morpheme]:
    return [_norm_morpheme(conllx.Morpheme.parse(t[1:])) for t in data.itertuples()]


def load(root_path: Path) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    lex_name = 'bgulex'
    preflex_file_path = root_path / lex_name / 'bgupreflex.csv'
    lex_file_path = root_path / lex_name / 'bgulex.csv'
    nikud_file_path = root_path / lex_name / 'nikud.csv'
    return _load_interm_lexicon(preflex_file_path, lex_file_path, nikud_file_path)


def process():
    raw_root_path = Path('data/raw/HebrewResources/HebrewTreebank')
    interim_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    # processed_root_path = Path('data/processed/HebrewResources/HebrewTreebank')
    lex_name = 'bgulex'
    preflex_file_path = raw_root_path / lex_name / 'bgupreflex_withdef.utf8.hr'
    lex_file_path = raw_root_path / lex_name / 'bgulex.utf8.hr'
    preflex_entries, lex_entries = _load_raw_lexicon(preflex_file_path=preflex_file_path, lex_file_path=lex_file_path)
    nikud_entries = _load_nikud()
    print(len(preflex_entries))
    print(len(lex_entries))
    print(len(nikud_entries))
    preflex_file_path = interim_root_path / lex_name / 'bgupreflex.csv'
    lex_file_path = interim_root_path / lex_name / 'bgulex.csv'
    nikud_file_path = interim_root_path / lex_name / 'nikud.csv'
    _save_interm_lexicon(preflex_entries, preflex_file_path, lex_entries, lex_file_path, nikud_entries, nikud_file_path)
    for _ in range(10):
        form = random.choice(list(preflex_entries.keys()))
        for a in preflex_entries[form]:
            for m in a:
                print(_format_morpheme(m))
        form = random.choice(list(lex_entries.keys()))
        for a in lex_entries[form]:
            for m in a:
                print(_format_morpheme(m))


if __name__ == '__main__':
    process()
