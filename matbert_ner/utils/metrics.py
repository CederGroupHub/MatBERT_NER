import torch


def accuracy(predicted, labels):
    predicted = torch.max(predicted,-1)[1]

    true = torch.where(labels > 0, labels, 0)
    predicted = torch.where(labels > 0, predicted, -1)

    acc = (true==predicted).sum().item()/torch.count_nonzero(true)
    return acc
