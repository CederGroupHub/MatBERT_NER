import torch


def valid_sequence_output(sequence_output, label_ids, attention_mask, valid_mask, device):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_sequence = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
    if label_ids is not None:
        valid_label_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    else:
        valid_label_ids = None
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        jj = 0
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                valid_sequence[i][jj] = sequence_output[i][j]
                if label_ids is not None:
                    valid_label_ids[i][jj] = label_ids[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
                jj += 1
    return valid_sequence, valid_label_ids, valid_attention_mask