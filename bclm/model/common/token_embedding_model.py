import torch
import torch.nn as nn


class TokenEmbeddingModel(nn.Module):

    def __init__(self, entries: list[str], weights: torch.Tensor):
        super().__init__()
        self._vocab = {entry: i for i, entry in enumerate(entries)}
        self._embedding = nn.Embedding.from_pretrained(weights, padding_idx=0)

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    @property
    def vocab(self) -> dict[str:int]:
        return self._vocab

    @property
    def embedding(self) -> nn.Embedding:
        return self._embedding

    def forward(self, input_seq):
        return self.embedding(input_seq)
