import torch
import torch.nn as nn
import torch.nn.functional as F

from bclm.models.common.bert_embedding import BertEmbedding
from bclm.models.common.sequence_label import SequenceLabelClassifier, compute_loss


class SegmentDecoder(nn.Module):

    def __init__(self, char_emb: nn.Embedding, hidden_size, num_layers, dropout, char_dropout, char_out_size,
                 labels_configs: list = None):
        super(SegmentDecoder, self).__init__()
        if labels_configs is None:
            labels_configs = []
        self.char_emb = char_emb
        self.char_encoder = nn.GRU(input_size=char_emb.embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=False,
                                   dropout=dropout)
        self.char_decoder = nn.GRU(input_size=char_emb.embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=False,
                                   dropout=dropout)
        self.char_dropout = nn.Dropout(char_dropout)
        self.char_out = nn.Linear(in_features=self.char_decoder.hidden_size, out_features=char_out_size)
        self.classifiers = nn.ModuleList([SequenceLabelClassifier(char_out_size, config) for config in labels_configs])

    @property
    def enc_num_layers(self):
        return self.char_encoder.num_layers

    @property
    def dec_num_layers(self):
        return self.char_decoder.num_layers

    def forward(self, char_seq, enc_state, special_symbols, max_out_char_seq_len, target_char_seq, max_num_labels):
        char_scores, char_states, label_scores = [], [], []
        enc_output, dec_char_state = self._forward_encode(char_seq, enc_state)
        sos, eos, sep = special_symbols['<s>'], special_symbols['</s>'], special_symbols['<sep>']
        dec_char = sos
        for _ in self.classifiers:
            label_scores.append([])
        while (len(char_scores) < max_out_char_seq_len and
               (max_num_labels is None or all([len(scores) < max_num_labels for scores in label_scores]))):
            dec_output = self._forward_decoder_step(dec_char, dec_char_state, target_char_seq, len(char_scores), sep)
            dec_char, dec_char_output, dec_char_state, dec_labels_output = dec_output
            char_scores.append(dec_char_output)
            char_states.append(dec_char_state.view(1, -1, self.dec_num_layers * dec_char_state.shape[2]))
            if dec_labels_output:
                for output, scores in zip(dec_labels_output, label_scores):
                    scores.append(output)
            if torch.all(torch.eq(dec_char, eos)):
                break
        dec_labels_output = self._labels_decode(char_scores[-1])
        if dec_labels_output:
            for output, scores in zip(dec_labels_output, label_scores):
                scores.append(output)
        char_fill_len = max_out_char_seq_len - len(char_scores)
        char_scores = torch.cat(char_scores, dim=1)
        char_scores_out = F.pad(char_scores, (0, 0, 0, char_fill_len))
        char_states = torch.cat(char_states, dim=1)
        char_states_out = F.pad(char_states, (0, 0, 0, char_fill_len))
        label_scores_out = []
        for scores in label_scores:
            label_fill_len = max_num_labels - len(scores)
            scores = torch.cat(scores, dim=1)
            label_scores_out.append(F.pad(scores, (0, 0, 0, label_fill_len)))
        return char_scores_out, char_states_out, label_scores_out

    def form_loss(self, form_scores, form_targets, criterion: nn.CrossEntropyLoss):
        return compute_loss(form_scores, form_targets, criterion)

    def labels_losses(self, labels_scores, labels_targets, criterion: nn.CrossEntropyLoss):
        return [classifier.loss(scores, targets, criterion)
                for scores, targets, classifier in zip(labels_scores, labels_targets, self.classifiers)]

    def _forward_encode(self, char_seq, enc_state):
        mask = torch.ne(char_seq, 0)
        emb_chars = self.char_emb(char_seq[mask]).unsqueeze(1)
        enc_state = enc_state.view(1, 1, -1)
        enc_state = torch.split(enc_state, enc_state.shape[2] // self.enc_num_layers, dim=2)
        enc_state = torch.cat(enc_state, dim=0)
        enc_output, enc_state = self.char_encoder(emb_chars, enc_state)
        return enc_output, enc_state

    def _forward_decoder_step(self, cur_dec_char, dec_char_state, target_char_seq, num_scores, sep):
        emb_dec_char = self.char_emb(cur_dec_char).unsqueeze(1)
        dec_char_output, dec_char_state = self.char_decoder(emb_dec_char, dec_char_state)
        dec_char_output = self.char_dropout(dec_char_output)
        dec_char_output = self.char_out(dec_char_output)
        if target_char_seq is not None:
            next_dec_char = target_char_seq[num_scores].unsqueeze(0)
        else:
            next_dec_char = self._form_decode(dec_char_output).squeeze(0)
        if torch.eq(next_dec_char, sep):
            dec_labels_output = self._labels_decode(dec_char_output)
        else:
            dec_labels_output = None
        return next_dec_char, dec_char_output, dec_char_state, dec_labels_output

    def _labels_decode(self, dec_char_output) -> list:
        return [classifier(dec_char_output) for classifier in self.classifiers]

    def _form_decode(self, scores):
        return torch.argmax(scores, dim=-1)

    def decode(self, form_scores, label_scores) -> (torch.Tensor, torch.Tensor):
        dec_forms = self._form_decode(form_scores)
        dec_labels = [classifier.decode(scores) for scores, classifier in zip(label_scores, self.classifiers)]
        return dec_forms, dec_labels


class MorphSequenceModel(nn.Module):

    def __init__(self, emb: BertEmbedding, segment_decoder: SegmentDecoder):
        super(MorphSequenceModel, self).__init__()
        self.emb = emb
        self.segment_decoder = segment_decoder

    @property
    def embedding_dim(self):
        return self.xemb.embedding_dim

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_labels,
                target_chars=None):
        token_ctx = self.xtoken_emb(xtoken_seq.unsqueeze(dim=0))[0]
        out_char_scores, out_char_states = [], []
        out_label_scores = []
        for _ in self.segment_decoder.classifiers:
            out_label_scores.append([])
        for cur_token_idx in range(num_tokens):
            cur_token_state = token_ctx[cur_token_idx + 1]
            cur_input_chars = char_seq[cur_token_idx]
            cur_target_chars = None
            if target_chars is not None:
                cur_target_chars = target_chars[cur_token_idx]
            seg_output = self.segment_decoder(cur_input_chars, cur_token_state, special_symbols, max_form_len,
                                              cur_target_chars, max_num_labels)
            cur_token_segment_scores, cur_token_segment_states, cur_token_label_scores = seg_output
            out_char_scores.append(cur_token_segment_scores)
            out_char_states.append(cur_token_segment_states)
            for out_scores, seg_scores in zip(out_label_scores, cur_token_label_scores):
                out_scores.append(seg_scores)
        out_char_scores = torch.cat(out_char_scores, dim=0)
        out_char_states = torch.cat(out_char_states, dim=0)
        out_label_scores = [torch.cat(label_scores, dim=0) for label_scores in out_label_scores]
        return out_char_scores, out_char_states, out_label_scores

    def decode(self, morph_seg_scores, label_scores: list):
        return self.segment_decoder.decode(morph_seg_scores, label_scores)

    def form_loss(self, form_scores, form_targets, criterion: nn.CrossEntropyLoss):
        return self.segment_decoder.form_loss(form_scores, form_targets, criterion)

    def labels_losses(self, labels_scores, labels_targets, criterion: nn.CrossEntropyLoss):
        return self.segment_decoder.labels_losses(labels_scores, labels_targets, criterion)


class MorphPipelineModel(MorphSequenceModel):

    def __init__(self, xtoken_emb: BertEmbedding, segment_decoder: SegmentDecoder, hidden_size, num_layers,
                 dropout, seg_dropout, labels_configs: list = None):
        super(MorphPipelineModel, self).__init__(xtoken_emb, segment_decoder)
        if labels_configs is None:
            labels_configs = []
        self.encoder = nn.LSTM(input_size=xtoken_emb.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=False,
                               dropout=dropout)
        self.seg_dropout = nn.Dropout(seg_dropout)
        self.classifiers = nn.ModuleList([SequenceLabelClassifier(hidden_size*2, config) for config in labels_configs])

    def forward(self, xtoken_seq, char_seq, special_symbols, num_tokens, max_form_len, max_num_labels,
                target_chars=None):
        morph_scores, morph_states, _ = super().forward(xtoken_seq, char_seq, special_symbols, num_tokens,
                                                        max_form_len, max_num_labels, target_chars)
        if target_chars is not None:
            morph_chars = target_chars
        else:
            morph_chars, _ = self.decode(morph_scores, [])
            morph_chars = morph_chars.squeeze(0)
        eos, sep = special_symbols['</s>'], special_symbols['<sep>']
        eos_mask = torch.eq(morph_chars[:num_tokens], eos)
        eos_mask[:, -1] = True
        eos_mask = torch.bitwise_and(torch.eq(torch.cumsum(eos_mask, dim=1), 1), eos_mask)

        sep_mask = torch.eq(morph_chars[:num_tokens], sep)
        sep_mask = torch.bitwise_and(torch.eq(torch.cumsum(eos_mask, dim=1), 0), sep_mask)

        seg_state_mask = torch.bitwise_or(eos_mask, sep_mask)
        seg_states = morph_states[seg_state_mask]
        enc_seg_scores, _ = self.encoder(seg_states.unsqueeze(dim=1))
        enc_seg_scores = self.seg_dropout(enc_seg_scores)
        label_scores = []
        seg_sizes = torch.sum(seg_state_mask, dim=1)
        for classifier in self.classifiers:
            scores = classifier(enc_seg_scores)
            scores = torch.split_with_sizes(scores.squeeze(dim=1), tuple(seg_sizes))
            scores = nn.utils.rnn.pad_sequence(scores, batch_first=True)
            fill_len = max_num_labels - scores.shape[1]
            label_scores.append(F.pad(scores, (0, 0, 0, fill_len)))
        return morph_scores, morph_states, label_scores

    def decode(self, morph_seg_scores, label_scores: list) -> (torch.Tensor, torch.Tensor):
        dec_forms, _ = self.segment_decoder.decode(morph_seg_scores, label_scores)
        dec_labels = [classifier.decode(scores) for scores, classifier in zip(label_scores, self.classifiers)]
        return dec_forms, dec_labels

    def form_loss(self, form_scores, form_targets, criterion: nn.CrossEntropyLoss):
        return self.segment_decoder.form_loss(form_scores, form_targets, criterion)

    def labels_losses(self, labels_scores, labels_targets, criterion: nn.CrossEntropyLoss):
        return [classifier.loss(scores, targets, criterion)
                for scores, targets, classifier in zip(labels_scores, labels_targets, self.classifiers)]
