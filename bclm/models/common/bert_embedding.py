import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertEmbedding(nn.Module):

    def __init__(self, bert: BertModel, bert_tokenizer: BertTokenizer):
        super(BertEmbedding, self).__init__()
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer

    @property
    def embedding_dim(self):
        return self.bert.config.hidden_size

    def forward(self, token_seq):
        mask = torch.ne(token_seq[:, :, 1], self.bert_tokenizer.pad_token_id)
        bert_output = self.bert(token_seq[:, :, 1], attention_mask=mask)
        bert_emb_tokens = bert_output.last_hidden_state
        emb_tokens = []
        for i in range(len(token_seq)):
            # # groupby token_id
            # mask = torch.ne(input_xtokens[i, :, 1], 0)
            idxs, vals = torch.unique_consecutive(token_seq[i, :, 0][mask[i]], return_counts=True)
            token_emb_xtoken_split = torch.split_with_sizes(bert_emb_tokens[i][mask[i]], tuple(vals))
            # token_xcontext = {k.item(): v for k, v in zip(idxs, [torch.mean(t, dim=0) for t in token_emb_xtokens])}
            emb_tokens.append(torch.stack([torch.mean(t, dim=0) for t in token_emb_xtoken_split], dim=0))
        return emb_tokens
