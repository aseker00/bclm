from collections import defaultdict
from pathlib import Path

import pandas as pd

from bclm.data import vocab
from bclm.data import hebtagset
from bclm.data.format import conllx

# ['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'FPOSTAG', 'FEATS', 'HEAD', 'DEPREL', 'TOKEN_ID', 'TOKEN']
MORPH_DEP_COLUMN_NAMES = conllx.CONLL_COLUMN_NANES[:-2] + conllx.LATTICE_COLUMN_NAMES[-1:] + ['TOKEN']


# Lattice Format
def format_morpheme(morpheme: conllx.Morpheme) -> list:
    form_str = morpheme.form if morpheme.form else '_'
    lemma_str = morpheme.lemma if morpheme.lemma else '_'
    cpostag_str = morpheme.cpostag.value if morpheme.cpostag else '_'
    fpostag_str = morpheme.fpostag.value if morpheme.fpostag else '_'
    feats_str = hebtagset.format_parsed_features(morpheme.feats) if morpheme.feats else '_'
    return [form_str, lemma_str, cpostag_str, fpostag_str, feats_str]


# Format dependency tree as a list of rows
# Each row is a tree node represented as a morpheme and dependency HEAD relation
def format_dep_tree(dep_tree: conllx.DepTree) -> list[list]:
    result = []
    for node in dep_tree.nodes:
        row = [node.index] + format_morpheme(node.morpheme)
        edge = dep_tree.edges[node.index]
        row.extend(list(edge))
        result.append(row)
    return result


# Format morphological dependency tree as a list of conll rows
# Get the dependency information (including morpheme) from the tree and the token info from the morphological lattice
def format_mdt(mdt: conllx.MorphDepTree) -> list:
    result = []
    for t, e in zip(format_dep_tree(mdt.dep_tree), mdt.lattice.edges):
        token = mdt.lattice.tokens[e.token_id]
        t.extend([e.token_id, token])
        result.append(t)
    return result


# Read raw conll text files and generate a dataframe for each partition (train, dev, test)
def spmrl(tb_root_path, tb_name, partition, ma_name=None) -> dict[str:pd.DataFrame]:
    result = {}
    column_names = [n.lower() for n in ['SENT_ID'] + MORPH_DEP_COLUMN_NAMES]
    for part in partition:
        if ma_name is not None:
            lattices_path = tb_root_path / f'{part}_{tb_name}.lattices'
        else:
            lattices_path = tb_root_path / f'{part}_{tb_name}-gold.lattices'
        tokens_path = tb_root_path / f'{part}_{tb_name}.tokens'
        conll_path = tb_root_path / f'{part}_{tb_name}-gold.conll'
        morph_dep_trees = conllx.load_morph_dep_trees(conll_path, lattices_path, tokens_path)
        morph_dep_rows = [[i+1] + row for i, mdt in enumerate(morph_dep_trees) for row in format_mdt(mdt)]
        mdt_df = pd.DataFrame(morph_dep_rows, columns=column_names)
        result[part] = mdt_df
    return result


# Save treebank partition (train, dev, test) as dataframes
def save_conll_morph_dataset(dataset, dest_dir: Path):
    for part in dataset:
        df = dataset[part]
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{dest_dir}/{part}_conll_morph_dep.csv')


# Load treebank partition (train, dev, test) as dataframes
def load_conll_morph_dataset(partition: list, src_dir: Path) -> dict[str:pd.DataFrame]:
    return {part: pd.read_csv(src_dir / f'{part}_conll_morph_dep.csv', index_col=0) for part in partition}


# Generate all possible values for each morphological feature type
# The result will be used to construct a morphological embedding vocabulary
def extract_morph_features(dataset: dict[str:pd.DataFrame]) -> dict[str:set[str]]:
    result = defaultdict(set)
    for part in dataset:
        df = dataset[part]
        # Go over all conll rows and extract the features column
        for f in df.feats:
            if f == "_":
                continue
            for values in f.split("|"):
                fv = values.split("=")
                result[fv[0]].add(fv[1])
    return result


# Construct a morphological embedding vocabulary for each morphological feature type
def build_morph_feature_vocab(features: list) -> dict[str:vocab.Vocabulary]:
    result = {}
    for f in features:
        values = ['<pad>', '<s>', '</s>'] + ['_'] + sorted(f.full_names())
        feat_vocab = vocab.Vocabulary(values)
        result[f.__name__] = feat_vocab
    return result


# Generate all possible POS tags
# The result will be used to construct a POS embedding vocabulary
def extract_morph_postags(dataset: dict[str:pd.DataFrame], column_name) -> set[str]:
    tags = set()
    for part in dataset:
        df = dataset[part]
        for t in df[column_name]:
            tags.add(t)
    return tags


# Construct a POS embedding vocabulary
def build_morph_postag_vocab(postags: list) -> vocab.Vocabulary:
    return vocab.Vocabulary(['<pad>', '<s>', '</s>'] + ['_'] + sorted(postags))


def process():
    raw_root_path = Path('data/raw/HebrewResources/HebrewTreebank')
    interim_root_path = Path('data/interim/HebrewResources/HebrewTreebank')
    processed_root_path = Path('data/processed/HebrewResources/HebrewTreebank')
    tb_partition = ['train', 'dev', 'test']
    tb_name = 'hebtb'

    tb_dataset = spmrl(raw_root_path / tb_name, tb_name, tb_partition)
    save_conll_morph_dataset(tb_dataset, interim_root_path / tb_name)
    morph_dataset = load_conll_morph_dataset(tb_partition, interim_root_path / tb_name)
    morph_feat_vocabs = build_morph_feature_vocab(hebtagset.morph_features)
    morph_postag_vocabs = build_morph_postag_vocab(hebtagset.postags)
    feats = extract_morph_features(tb_dataset)
    for feat in feats:
        feature = hebtagset.get_morph_feature_by_name(feat)
        values = sorted(feature.full_names())
        print(f'{feat}: {values}')
    cpostags = sorted(extract_morph_postags(tb_dataset, 'cpostag'))
    print(','.join(cpostags))
    fpostags = sorted(extract_morph_postags(tb_dataset, 'fpostag'))
    print(','.join(fpostags))


def main():
    process()


if __name__ == '__main__':
    main()
