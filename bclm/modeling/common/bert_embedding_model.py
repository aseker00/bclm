import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from pathlib import Path


class BertTokenEmbedding(nn.Module):

    def __init__(self, bert: BertModel, bert_tokenizer: BertTokenizerFast):
        super().__init__()
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer

    @property
    def embedding_dim(self):
        return self.bert.config.hidden_size

    def forward(self, input_seq):
        mask = torch.ne(input_seq, self.bert_tokenizer.pad_token_id)
        bert_output = self.bert(input_seq, attention_mask=mask)
        emb_tokens = bert_output.last_hidden_state
        return emb_tokens

    def to_data_sample(self, words_seq: list[str]) -> (pd.DataFrame, pd.DataFrame):
        # BERT tokens dataframe
        words_data = _to_words_data(words_seq, self.bert_tokenizer)
        words_data['token_id'] = np.vectorize(self.get_token_id)(words_data['token'])
        return words_data

    def get_token_id(self, token: str) -> int:
        return self.bert_tokenizer.convert_tokens_to_ids(token)


def to_input_sample(data_sample: pd.DataFrame) -> torch.Tensor:
    return torch.tensor([data_sample['token_id']])


def _to_words_data(input_word_seq: list[str], bert_tokenizer: BertTokenizerFast) -> pd.DataFrame:
    bert_tokens = [('<sos>', bert_tokenizer.cls_token)]
    bert_tokens.extend([(word, bert_token) for word in input_word_seq for bert_token in bert_tokenizer.tokenize(word)])
    bert_tokens.append(('<eos>', bert_tokenizer.sep_token))
    indices = list(range(len(bert_tokens)))
    words = [t[0] for t in bert_tokens]
    tokens = [t[1] for t in bert_tokens]
    return pd.DataFrame(list(zip(indices, words, tokens)), columns=['word_index', 'word', 'token'])


def _read_words(p: Path) -> list[str]:
    with p.open() as f:
        return [line.strip() for line in f.readlines()]


def create_bert_model(bert_model_path: Path) -> BertTokenEmbedding:
    logging.info(f'Loading BERT from: {bert_model_path}')
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
    bert_model = BertModel.from_pretrained(bert_model_path)
    return BertTokenEmbedding(bert_model, bert_tokenizer)


if __name__ == '__main__':
    ab_path = Path('onlplab/alephbert-base')
    # ab_path = Path('/Users/Amit/dev/aseker00/alephbert/experiments/transformers/bert')
    # ab_path /= '/bert-basic-wordpiece-owt-52000-10'
    emb_model = create_bert_model(ab_path)
    # print(emb_model)
    # print(list(emb_model.parameters()))
    sample_sentence = _read_words(Path('words.txt'))
    sample_data = emb_model.to_data_sample(sample_sentence)
    print(sample_data)
    sample_input = to_input_sample(sample_data)
    for word_vec in emb_model(sample_input):
        print(word_vec)
