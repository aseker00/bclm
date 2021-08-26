import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bclm.models.morph.morph_model import MorphSequenceModel


def predict(model: MorphSequenceModel, data: DataLoader):
    for i, batch in enumerate(data):
        batch = tuple(t.to(device) for t in batch)
        for sent_xtoken, sent_token_chars, sent_form_chars, sent_labels in zip(*batch):
            input_token_chars = sent_token_chars[:, :, -1]
            num_tokens = len(sent_token_chars[sent_token_chars[:, 0, 1] > 0])
            target_token_form_chars = sent_form_chars[:, :, -1]
            max_form_len = target_token_form_chars.shape[1]
            target_token_labels = sent_labels[:, :, 2:]
            max_num_labels = target_token_labels.shape[1]

            form_scores, _, label_scores = model(sent_xtoken, input_token_chars, char_special_symbols, num_tokens,
                                                 max_form_len, max_num_labels,
                                                 target_token_form_chars if use_teacher_forcing else None)
            batch_form_scores.append(form_scores)
            batch_label_scores.append(label_scores)
            batch_form_targets.append(target_token_form_chars[:num_tokens])
            batch_label_targets.append(target_token_labels[:num_tokens])
            batch_token_chars.append(input_token_chars[:num_tokens])
            batch_sent_ids.append(sent_form_chars[:, :, 0].unique().item())
            batch_num_tokens.append(num_tokens)

        # Decode
        batch_form_scores = nn.utils.rnn.pad_sequence(batch_form_scores, batch_first=True)
        batch_label_scores = [nn.utils.rnn.pad_sequence(label_scores, batch_first=True)
                              for label_scores in list(map(list, zip(*batch_label_scores)))]
        with torch.no_grad():
            batch_decoded_chars, batch_decoded_labels = model.decode(batch_form_scores, batch_label_scores)

        # Form Loss
        batch_form_targets = nn.utils.rnn.pad_sequence(batch_form_targets, batch_first=True)
        form_loss = model.form_loss(batch_form_scores, batch_form_targets, criterion)
        print_form_loss += form_loss.item()

        # Label Losses
        batch_label_targets = [[t[:, :, j] for j in range(t.shape[-1])] for t in batch_label_targets]
        batch_label_targets = [nn.utils.rnn.pad_sequence(label_targets, batch_first=True)
                               for label_targets in list(map(list, zip(*batch_label_targets)))]
        label_losses = model.labels_losses(batch_label_scores, batch_label_targets, criterion)
        for j in range(len(label_losses)):
            print_label_losses[j] += label_losses[j].item()

        # Optimization Step
        if optimizer is not None:
            form_loss.backward(retain_graph=len(label_losses) > 0)
            for j in range(len(label_losses)):
                label_losses[j].backward(retain_graph=(j < len(label_losses) - 1))
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # To Lattice
        for j in range(len(batch_sent_ids)):
            sent_id = batch_sent_ids[j]
            input_chars = batch_token_chars[j]
            target_form_chars = batch_form_targets[j]
            target_labels = [label_targets[j] for label_targets in batch_label_targets]
            decoded_form_chars = batch_decoded_chars[j]
            decoded_labels = [decoded_labels[j] for decoded_labels in batch_decoded_labels]
            num_tokens = batch_num_tokens[j]
            input_chars = input_chars.to('cpu')
            target_form_chars = target_form_chars[:num_tokens].to('cpu')
            decoded_form_chars = decoded_form_chars[:num_tokens].to('cpu')
            target_labels = [labels[:num_tokens].to('cpu') for labels in target_labels]
            decoded_labels = [labels[:num_tokens].to('cpu') for labels in decoded_labels]
            input_tokens = utils.to_sent_tokens(input_chars, char_vocab['id2char'])
            target_morph_segments = utils.to_token_morph_segments(target_form_chars,
                                                                  char_vocab['id2char'],
                                                                  char_eos, char_sep)
            decoded_morph_segments = utils.to_token_morph_segments(decoded_form_chars,
                                                                   char_vocab['id2char'],
                                                                   char_eos, char_sep)
            target_morph_labels = utils.to_token_morph_labels(target_labels, label_names,
                                                              label_vocab['id2labels'],
                                                              label_pads)
            decoded_morph_labels = utils.to_token_morph_labels(decoded_labels, label_names,
                                                               label_vocab['id2labels'],
                                                               label_pads)

            decoded_token_lattice_rows = (sent_id, input_tokens, decoded_morph_segments, decoded_morph_labels)
            print_decoded_lattice_rows.append(decoded_token_lattice_rows)
            print_target_forms.append(target_morph_segments)
            print_target_labels.append(target_morph_labels)
            print_decoded_forms.append(decoded_morph_segments)
            print_decoded_labels.append(decoded_morph_labels)

        # Log Print Eval
        if (i + 1) % print_every == 0:
            sent_id, input_tokens, decoded_segments, decoded_labels = print_decoded_lattice_rows[-1]
            target_segments = print_target_forms[-1]
            target_labels = print_target_labels[-1]
            decoded_segments = print_decoded_forms[-1]
            decoded_labels = print_decoded_labels[-1]

            print(f'epoch {epoch} {phase}, batch {i + 1} form char loss: {print_form_loss / print_every}')
            for j in range(len(label_names)):
                print(
                    f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} loss: {print_label_losses[j] / print_every}')
            print(f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} input tokens  : {input_tokens}')
            print(
                f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} target forms  : {list(reversed(target_segments))}')
            print(
                f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} decoded forms : {list(reversed(decoded_segments))}')
            for j in range(len(label_names)):
                target_values = [labels[j] for labels in target_labels]
                print(
                    f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} target {label_names[j]} labels  : {list(reversed([target_values]))}')
                decoded_values = [labels[j] for labels in decoded_labels]
                print(
                    f'epoch {epoch} {phase}, batch {i + 1} sent #{sent_id} decoded {label_names[j]} labels : {list(reversed([decoded_values]))}')
            total_form_loss += print_form_loss
            for j, label_loss in enumerate(print_label_losses):
                total_label_losses[j] += label_loss
            print_form_loss = 0
            print_label_losses = [0 for _ in range(len(label_names))]

            total_decoded_forms.extend(print_decoded_forms)
            total_decoded_labels.extend(print_decoded_labels)
            total_target_forms.extend(print_target_forms)
            total_target_labels.extend(print_target_labels)
            total_decoded_lattice_rows.extend(print_decoded_lattice_rows)

            aligned_scores, mset_scores = utils.morph_eval(print_decoded_forms, print_target_forms)
            # print(f'epoch {epoch} {phase}, batch {i + 1} form aligned scores: {aligned_scores}')
            print(f'epoch {epoch} {phase}, batch {i + 1} form mset scores: {mset_scores}')

            for j in range(len(label_names)):
                if label_names[j][:3].lower() in ['tag', 'bio', 'gen', 'num', 'per', 'ten']:
                    decoded_values = [labels[j] for sent_labels in print_decoded_labels for labels in sent_labels]
                    target_values = [labels[j] for sent_labels in print_target_labels for labels in sent_labels]
                    aligned_scores, mset_scores = utils.morph_eval(decoded_values, target_values)
                    # print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} aligned scores: {aligned_scores}')
                    print(f'epoch {epoch} {phase}, batch {i + 1} {label_names[j]} mset scores: {mset_scores}')

            print_target_forms = []
            print_target_labels = []
            print_decoded_forms = []
            print_decoded_labels = []
            print_decoded_lattice_rows = []

    # Log Total Eval
    if print_form_loss > 0:
        total_form_loss += print_form_loss
        for j, label_loss in enumerate(print_label_losses):
            total_label_losses[j] += label_loss
        total_decoded_forms.extend(print_decoded_forms)
        total_decoded_labels.extend(print_decoded_labels)
        total_target_forms.extend(print_target_forms)
        total_target_labels.extend(print_target_labels)
        total_decoded_lattice_rows.extend(print_decoded_lattice_rows)

    print(f'epoch {epoch} {phase}, total form char loss: {total_form_loss / len(data)}')
    for j in range(len(label_names)):
        print(f'epoch {epoch} {phase}, total {label_names[j]} loss: {total_label_losses[j] / len(data)}')

    for j in range(len(label_names)):
        if label_names[j][:3].lower() in ['tag', 'bio', 'gen', 'num', 'per', 'ten']:
            decoded_values = [labels[j] for sent_labels in total_decoded_labels for labels in sent_labels]
            target_values = [labels[j] for sent_labels in total_target_labels for labels in sent_labels]
            aligned_scores, mset_scores = utils.morph_eval(decoded_values, target_values)
            # print(f'epoch {epoch} {phase}, total {label_names[j]} aligned scores: {aligned_scores}')
            print(f'epoch {epoch} {phase}, total {label_names[j]} mset scores: {mset_scores}')

    return utils.get_lattice_data(total_decoded_lattice_rows, label_names)
