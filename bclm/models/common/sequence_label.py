import torch
import torch.nn as nn
from conditional_random_field import allowed_transitions, ConditionalRandomField


def compute_loss(scores, targets, criterion: nn.CrossEntropyLoss):
    loss_target = targets.view(-1)
    loss_input = scores.view(-1, scores.shape[-1])
    return criterion(loss_input, loss_target)


# Mask starting from the position of the first mask_value occurrence
# Done on the last dimension [batch, token, morph_labels]
# E.g. [[[28, 0, 0, 0, 0], [5, 7, 0, 0, 0], [6, 0, 0, 0, 0], [13, 13, 0, 0, 0], [8, 0, 0, 0, 0]]]
# The reason for masking from the first occurrence onward (as opposed to just using torch.ne(labels, mask_value)
# is that the mask_value might be predicted more than once (e.g. if the mask_value is the </s> (3) value):
# [[[12, 3, 3, 5, 0], [3, 1, 2, 5, 5], [5, 5, 3, 0, 0]]]
def _first_occurence_mask(labels: torch.Tensor, mask_value) -> torch.BoolTensor:
    masks = torch.eq(labels, mask_value)
    return ~torch.cumsum(masks, dim=-1).bool()


def _crf_prepare(scores, labels) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    masks = torch.ne(labels, 0)
    masked_scores = [scores[mask] for scores, mask in zip(scores, masks)]
    masked_labels = [label[mask] for label, mask in zip(labels, masks)]
    logits = nn.utils.rnn.pad_sequence(masked_scores, batch_first=True)
    tags = nn.utils.rnn.pad_sequence(masked_labels, batch_first=True)
    return logits, tags, masks


class SequenceLabelClassifier(nn.Module):

    def __init__(self, char_emb_size, config: dict):
        super(SequenceLabelClassifier, self).__init__()
        self.config = config
        self.num_labels = len(config['id2label'])
        self.ff = nn.Linear(in_features=char_emb_size, out_features=self.num_labels)
        self.crf = None
        if 'crf_trans_type' in config:
            constraint_type = config['crf_trans_type']
            labels = config['id2label']
            transitions = allowed_transitions(constraint_type=constraint_type, labels=labels)
            self.crf = ConditionalRandomField(num_tags=self.num_labels, constraints=transitions)

    def forward(self, dec_chars):
        return self.ff(dec_chars)

    def loss(self, scores, targets, criterion: nn.CrossEntropyLoss):
        if self.crf is None:
            loss_value = compute_loss(scores, targets, criterion)
        else:
            crf_scores, crf_tags, _ = _crf_prepare(scores, targets)
            crf_masks = torch.ne(crf_tags, 0).bool()
            crf_log_likelihood = self.crf(inputs=crf_scores, tags=crf_tags, mask=crf_masks)
            crf_log_likelihood /= torch.sum(crf_masks)
            loss_value = -crf_log_likelihood
        return loss_value

    def decode(self, scores) -> torch.Tensor:
        decoded_labels = torch.argmax(scores, dim=-1)
        if self.crf is not None:
            crf_scores, crf_tags, token_masks = _crf_prepare(scores, decoded_labels)
            crf_masks = torch.ne(crf_tags, 0).bool()
            crf_decoded_labels = self.crf.viterbi_tags(logits=crf_scores, mask=crf_masks)
            for labels, crf_labels, token_mask in zip(decoded_labels, crf_decoded_labels, token_masks):
                idxs_vals = [torch.unique_consecutive(mask, return_counts=True) for mask in token_mask]
                idxs = torch.cat([idx for idx, _ in idxs_vals])
                vals = torch.cat([val for _, val in idxs_vals])
                decoded_token_tags = torch.split_with_sizes(torch.tensor(crf_labels[0]), tuple(vals[idxs]))
                # TODO: this doesn't do the right thing if a token is decoded as all pads (0)
                # In such a case the first mask is all False and the above split doesn't indicate that this token
                # should be skipped
                for idx, token_tags in enumerate(decoded_token_tags):
                    labels[idx, :len(token_tags)] = token_tags
        return decoded_labels
