from bclm.data_processing.format import conllx
from enum import Enum

# The list of CoNLL Hebrew tags can be examined on the UD conversion page:
# https://universaldependencies.org/tagset-conversion/he-conll-uposf.html
# In addition I found the following thesis dissertation mapping the Hebrew POS tags into UPOS:
# https://yonatanbisk.com/papers/Thesis.pdf


class Feature(Enum):

    # Full name: Feature type prefix and feature value name, i.e. Type_ValueName
    # Extracted from the enum keys
    @classmethod
    def full_names(cls) -> set[str]:
        return set(map(lambda c: c.name, cls))

    # Name: Extract feature value w/o the type prefix from the full name
    # Extracted from the enum keys
    @classmethod
    def names(cls) -> set[str]:
        return set(map(lambda n: n.split('_')[1], cls.full_names()))

    # Values: Get all values
    # Extracted from the enum values
    @classmethod
    def values(cls) -> set[str]:
        return set(map(lambda c: c.value, cls))
        # return set(map(lambda k: k.split('_')[1], cls.__members__.keys()))

    # Get feature value from the feature name value part
    # Generate the full name and use it as the enum key to return the mapped value
    @classmethod
    def get_value(cls, value: str):
        return cls[f'{cls.__name__}_{value}']

    # Map to the feature name from the feature value
    @classmethod
    def get_name(cls, value: str):
        return cls.get_value(value).name


class POSTag(Feature):

    # yy, Nikud
    POSTag_yyCLN = 'yyCLN'
    POSTag_yyCM = 'yyCM'
    POSTag_yyDASH = 'yyDASH'
    POSTag_yyDOT = 'yyDOT'
    POSTag_yyELPS = 'yyELPS'
    POSTag_yyEXCL = 'yyEXCL'
    POSTag_yyLRB = 'yyLRB'
    POSTag_yyQM = 'yyQM'
    POSTag_yyQUOT = 'yyQUOT'
    POSTag_yyRRB = 'yyRRB'
    POSTag_yySCLN = 'yySCLN'

    # !!Special!!
    # POSTag_MISS = 'MISS'  # Analysis not in the lexicon
    # POSTag_SOME_ = 'SOME_'
    POSTag_UNK = 'UNK'  # Word not in the lexicon
    POSTag_ZVL = 'ZVL'

    # Tags
    POSTag_ADVERB = 'ADVERB'
    POSTag_AT = 'AT'  # Accusative marker
    POSTag_BN = 'BN'  # BEINONI verb
    POSTag_BN_S_PP = 'BN_S_PP'  # BEINONI verb possessive
    POSTag_BNT = 'BNT'  # Gerund
    POSTag_CC = 'CC'  # Coordinating conjunction
    POSTag_CC_COORD = 'CC-COORD'  # Coordinating conjunction
    POSTag_CC_REL = 'CC-REL'  # (afr -> asher)
    POSTag_CC_SUB = 'CC-SUB'  # Subordinating conjunction
    POSTag_CD = 'CD'  # Numeral (definite)
    POSTag_CD_S_PP = 'CD_S_PP'  # (e.g. alpyta)
    POSTag_CDT = 'CDT'  # Numeral determiner (definite)
    POSTag_CONJ = 'CONJ'
    POSTag_COP = 'COP'  # Copula
    POSTag_COP_TOINFINITIVE = 'COP-TOINFINITIVE'  # to be
    POSTag_DEF = 'DEF'  # "The" (h)
    # POSTag_DEF_DT = 'DEF@DT'  # "All" (hkl -> hakol)
    POSTag_DT = 'DT'  # Determiner
    POSTag_DTT = 'DTT'  # "All", "how many"
    POSTag_DUMMY_AT = 'DUMMY_AT'  #
    POSTag_EX = 'EX'  # Existential?
    POSTag_IN = 'IN'  # Preposition (e.g. el)
    POSTag_INTJ = 'INTJ'  # Interjection
    POSTag_JJ = 'JJ'  # Adjective
    POSTag_JJT = 'JJT'  # Construct state (smikhut) adjective
    POSTag_MD = 'MD'  # Modal
    POSTag_NCD = 'NCD'  # Date/Time
    POSTag_NEG = 'NEG'
    POSTag_NN = 'NN'  # Noun (definite - definite-genetive)
    POSTag_NNP = 'NNP'  # Proper noun
    POSTag_NNPT = 'NNPT'
    POSTag_NNT = 'NNT'  # Construct state (smikhut) noun
    POSTag_NN_S_PP = 'NN_S_PP'  # Possessive Pronoun suffix (fl). e.g. pnihm (paney-hem), pnim fl hm, i.e. their face
    POSTag_P = 'P'  # Prefix
    POSTag_POS = 'POS'  # Possessive item (fl -> shel)
    POSTag_PREPOSITION = 'PREPOSITION'
    POSTag_PRP = 'PRP'  # Personal Pronoun
    POSTag_PRP_REF = 'PRP-REF'  # Reference?
    POSTag_PRP_PERS = 'PRP-PERS'  # Personal?
    POSTag_PRP_DEM = 'PRP-DEM'  # Demonstrative? (as a preposition)
    POSTag_PRP_IMP = 'PRP-IMP'  # Impersonal? (as a noun)
    POSTag_PUNC = 'PUNC'  # Nikud
    POSTag_QW = 'QW'  # Question/WH word
    POSTag_RB = 'RB'  # Adverb
    POSTag_REL = 'REL'  # Relativizer (f -> she)
    POSTag_REL_SUBCONJ = 'REL-SUBCONJ'  # (f -> she)-lifney
    POSTag_S_ANP = 'S_ANP'  # Accusative Pronoun suffix, e.g. labdh -> labd at hia, i.e. to lose her
    # POSTag_S_PP = 'S_PP'  # Possessive suffix, e.g. ildm -> ild fl hm, i.e. their child. See also NN_S_PP
    POSTag_S_PRN = 'S_PRN'  # Pronoun suffix, e.g. klphm -> klpi hm, i.e. towards them
    POSTag_TEMP = 'TEMP'  # Temporal
    POSTag_TEMP_SUBCONJ = 'TEMP-SUBCONJ'  # WH / Subordinate Conj (e.g. when, kf -> kshe)
    POSTag_TTL = 'TTL'  # Title (determiner)
    POSTag_VB = 'VB'  # Verb
    # POSTag_VB_BAREINFINITIVE = 'VB-BAREINFINITIVE'
    # POSTag_VB_TOINFINITIVE = 'VB-TOINFINITIVE'


# Morphological features: Gender, Number, Person, Tense, Polarity, Binyan
class MorphFeature(Feature):

    # Override
    # Values: Get all values
    # Extracted from the enum keys, not from the enum values.
    # Some enum keys map to the same value (MF, SP, DP) and we want to cover those key variations to match occurrences
    # in the lexicon and treebank.
    # For example:
    # Gender.values = {M, F, MF, FM},
    # Number.values = {S, D, P, SP, PS, DP, PD}
    # Polarity.values = {neg, NEGATIVE, pos, POSITIVE}
    @classmethod
    def values(cls) -> set[str]:
        return set(map(lambda k: k.split('_')[1], cls.__members__.keys()))


class Gender(MorphFeature):
    Gender_M = 'M'
    Gender_F = 'F'
    Gender_MF, Gender_FM = 'MF', 'MF'


class Number(MorphFeature):
    Number_S = 'S'
    Number_D = 'D'
    Number_P = 'P'
    Number_SP, Number_PS = 'SP', 'SP'
    Number_DP, Number_PD = 'DP', 'DP'


class Person(MorphFeature):
    Person_1 = '1'
    Person_2 = '2'
    Person_3 = '3'
    Person_A = 'A'


class Tense(MorphFeature):
    Tense_PAST = 'PAST'
    Tense_FUTURE = 'FUTURE'
    Tense_BEINONI = 'BEINONI'
    Tense_TOINFINITIVE = 'TOINFINITIVE'
    Tense_IMPERATIVE = 'IMPERATIVE'


class Polarity(MorphFeature):
    Polarity_neg, Polarity_NEGATIVE = 'neg', 'neg'
    Polarity_pos, Polarity_POSITIVE = 'pos', 'pos'


class Binyan(MorphFeature):
    Binyan_PAAL = 'PAAL'
    Binyan_PIEL = 'PIEL'
    Binyan_PUAL = 'PUAL'
    Binyan_NIFAL = 'NIFAL'
    Binyan_HIFIL = 'HIFIL'
    Binyan_HITPAEL = 'HITPAEL'
    Binyan_HUFAL = 'HUFAL'


# Mapping treebank morphological features
_morph_feat2enum = {'gen': Gender,
                    'num': Number,
                    'per': Person,
                    'tense': Tense,
                    'polar': Polarity,
                    'binyan': Binyan}
_enum2morph_feat = {f: n for n, f in _morph_feat2enum.items()}
_morph_feat_value2enum = {v: f for f in [Gender, Number, Person, Tense, Polarity, Binyan] for v in f.values()}
_morph_feat_value2feat = {v: _enum2morph_feat[f] for v, f in _morph_feat_value2enum.items()}
_morph_feat_value2name = {v: f.get_value(v).name for v, f in _morph_feat_value2enum.items()}
morph_features = _enum2morph_feat.keys()

# Mapping treebank POS tags
postags = set(POSTag.names())
postag_values = set(POSTag.values())
postag_value2name = {f.value: v for v, f in POSTag.__members__.items()}
postag_name2value = {v: k for k, v in postag_value2name.items()}


def get_postag(value: str) -> POSTag:
    name = postag_value2name[value]
    return POSTag[name]


def get_morph_feature(value: str) -> MorphFeature:
    feature = _morph_feat_value2enum[value]
    name = _morph_feat_value2name[value]
    return feature[name]


def format_parsed_features(feats: dict) -> str:
    feature_values = [f'{f}={feats[f].value}' for f in feats if feats[f] is not None]
    if not feature_values:
        return '_'
    return '|'.join(feature_values)


def parse_features(feats: dict[str: list[str]]) -> dict[str: MorphFeature]:
    result = {}
    for feat in feats:
        value = ''.join(feats[feat])
        result[feat] = get_morph_feature(value)
    return result


def get_morph_feature_name_by_value(value: str) -> str:
    return _morph_feat_value2feat[value]


# Handle both normal and suffix feature values
def get_morph_feature_by_name(feat_name: str) -> MorphFeature:
    return _morph_feat2enum[feat_name[4:]] if feat_name[:4] == 'suf_' else _morph_feat2enum[feat_name]
