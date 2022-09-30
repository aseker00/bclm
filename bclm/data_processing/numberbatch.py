import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path
import gzip


# Numberbatch data downloaded from: https://github.com/commonsense/conceptnet-numberbatch
# mini.h5 contain words in following languages: de, en, es, fr, it, ja, nl, pt, ru, zh
# mini.h5 embeddings are int-based vectors, dim=300
# 19.08.txt contain words in all supported languages (format: b'/c/<lang>/word'
# 19.08.txt embeddings are float-based vectors, dim=300


# Load from raw text
# Each line: word-key 300-space-separated-float-values
def _read_from_raw_text(obj, in_mem: bool = True) -> (np.ndarray, np.ndarray):
    # size, dim = None, None
    keys, vectors = [], []
    with obj as f:
        lines = f.readlines() if in_mem else f
        for i, line in enumerate(lines):
            # First line lists the data shape (size, dim)
            if i == 0:
                # [size, dim] = line.strip().split()
                continue
            [key, *vec] = line.strip().split()
            keys.append(key)
            vectors.append(np.asarray(vec, dtype='float32'))
    keys = np.asarray(keys, dtype='str')
    vectors = np.asarray(vectors, dtype='float32')
    return keys, vectors


# Load from raw text
# Each line: word-key 300-space-separated-float-values
def load_raw_txt(file_path: Path) -> (np.ndarray, np.ndarray):
    if file_path.suffix == '.gz':
        f = gzip.open(file_path, 'rt', encoding='utf-8')
    elif file_path.suffix == '.txt':
        f = open(file_path, encoding='utf-8')
    return _read_from_raw_text(f, in_mem=False)


# Load from raw HDF5 (N = number of word entries)
# store['mat']['axis0'] - index (int, in the range 0:N-1)
# store['mat']['axis1'] - word keys (utf8 byte string), key: b'/c/<lang>/word'
# store['mat']['block0_values'] - word entry index (int, in the range 0:N-1)
# store['mat']['block1_values'] - embeddings vectors (int values, shape: (N, 300))
def load_raw_hdf5(file_path: Path) -> (np.ndarray, np.ndarray):
    hf = h5py.File(file_path, 'r')
    keys = np.char.decode(hf['mat']['axis1'], encoding='utf-8')
    vectors = hf['mat']['block0_values'][:]
    return keys, vectors


def _split_keys(keys: np.ndarray) -> np.ndarray:
    return np.stack(np.char.split(keys, sep='/'), axis=0)


def _get_lang_words(keys: np.ndarray) -> (np.ndarray, np.ndarray):
    keys_parts = _split_keys(keys)
    # 19.08-en (english only)
    if keys_parts.shape[1] == 1:
        langs = ['en']
        words = keys_parts[:, 0]
    # 19.08 (all languages)
    elif keys_parts.shape[1] == 4:
        langs = keys_parts[:, 2]
        words = keys_parts[:, 3]
    return langs, words


def raw_to_dataframe(keys: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    langs, words = _get_lang_words(keys)
    # return pd.DataFrame({'lang': langs, 'word': words, 'vec': values}, columns=['lang', 'word', 'vec'])
    return pd.DataFrame(list(zip([langs, words, values])), columns=['lang', 'word', 'vec'])


def raw_to_arrow(keys: np.ndarray, values: np.ndarray) -> pa.Table:
    langs, words = _get_lang_words(keys)
    langs = pa.array(langs)
    words = pa.array(words)
    vectors = pa.array(list(values))
    return pa.table([langs, words, vectors], names=['lang', 'word', 'vec'])


# Save intermediate arrow
def save_interm_partition(emb_table: pa.Table, dst_folder_path: Path):
    ds.write_dataset(emb_table, dst_folder_path, format="parquet", partitioning=["lang"],
                     existing_data_behavior='delete_matching')
    # partitioning=ds.partitioning(pa.schema([emb_table.schema.field("lang")])))


# Load intermediate arrow
def load_interm_partition(src_folder_path: Path) -> ds.Dataset:
    return ds.dataset(src_folder_path, format="parquet", partitioning=["lang"])


# Transform arrow table to lang-specific dataframe
def interm_to_dataframe(dataset: ds.Dataset, lang: str) -> pd.DataFrame:
    return dataset.to_table(filter=ds.field('lang') == lang).to_pandas()[['word', 'vec']].set_index('word')


def process(options: dict):
    words, vectors = [], []
    if options.get('src_raw_hdf5_path') is not None:
        words, vectors = load_raw_hdf5(options['src_raw_hdf5_path'])
    elif options.get('src_raw_text_path') is not None:
        words, vectors = load_raw_txt(options['src_raw_text_path'])

    if options.get('dst_inter_partition_path') is not None:
        emb_table = raw_to_arrow(words, vectors)
        save_interm_partition(emb_table, options['dst_inter_partition_path'])


# Load language specific word embeddings as a dataframe
def load_emb_df(options: dict) -> pd.DataFrame:
    lang = options['lang']
    part_path = options['src_inter_partition_path']
    emb_table = load_interm_partition(part_path)
    return interm_to_dataframe(emb_table, lang)


if __name__ == '__main__':
    options = {
        # 'src_raw_hdf5_path': Path('data/raw/numberbatch/mini.h5'),
        'src_raw_hdf5_path': None,
        'src_raw_text_path': Path('data/raw/numberbatch/numberbatch-19.08.txt.gz'),
        # 'src_raw_text_path': None,
        'src_inter_pickle_word_path': None,
        'src_inter_pickle_vec_path': None,
        'dst_inter_pickle_word_path': None,
        'dst_inter_pickle_vec_path': None,
        # 'dst_inter_partition_path': 'data/interim/numberbatch_h5',
        'dst_inter_partition_path': 'data/interim/numberbatch_txt_19.08',
        # 'src_inter_partition_path': 'data/interim/numberbatch_h5',
        'src_inter_partition_path': 'data/interim/numberbatch_txt_19.08',
        'lang': 'he'
    }
    # process(options)
    df = load_emb_df(options)
    print(df['vec'].values)
