import torch


def valid_sequence_output(sequence_output, label_ids, attention_mask, valid_mask, device):
    '''
    Constructs valid tensors for the output BERT sequences, labels ids and attention mask by filtering out invalid indices
        Arguments:
            sequence_output: Batch of output representation of sequence from BERT
            label_ids: Batch of sequence labels
            attention_mask: Batch of sequence attention masks
            valid_mask: Batch of sequence valid masks
            device: Device used for computation
        Returns:
            valid_sequence, valid_label_ids, valid_attention_mask
    '''
    # get shape of bert output sequence
    batch_size, max_len, feat_dim = sequence_output.shape
    # initialize empty valid sequence
    valid_sequence = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
    # initialize valid labels if label ids provided
    if label_ids is not None:
        valid_label_ids = torch.zeros(batch_size, max_len, dtype=torch.uint8, device=device)
    else:
        valid_label_ids = None
    # initialize valid attention mask
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    # loop through samples in batch
    for i in range(batch_size):
        # valid index starts at zero
        k = 0
        # loop through length of sample
        for j in range(max_len):
            # if valid entry
            if valid_mask[i][j].item() == 1:
                # fill in the valid tensors
                valid_sequence[i][k] = sequence_output[i][j]
                if label_ids is not None:
                    valid_label_ids[i][k] = label_ids[i][j]
                valid_attention_mask[i][k] = attention_mask[i][j]
                # increment index for valid tensors
                k += 1
    # return valid tensors
    return valid_sequence, valid_label_ids, valid_attention_mask
