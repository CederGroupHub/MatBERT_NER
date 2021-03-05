import torch


def valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, device):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask