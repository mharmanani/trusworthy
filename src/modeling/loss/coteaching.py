import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# def loss_coteaching(loss1, loss2, forget_rate):
#    loss1 = loss1.view(-1)
#    loss2 = loss2.view(-1)
#    loss1_sorted, ind1_sorted = torch.sort(loss1)
#    loss2_sorted, ind2_sorted = torch.sort(loss2)
#    num_remember = int((1 - forget_rate) * len(loss1_sorted))
#    ind1_update = ind1_sorted[:num_remember]
#    ind2_update = ind2_sorted[:num_remember]
#    loss1_update = torch.mean(loss1[ind2_update])
#    loss2_update = torch.mean(loss2[ind1_update])
#    return loss1_update, loss2_update


def coteaching_loss(loss1, loss2, forget_rate, device):

    r_t = 1 - forget_rate
    total_samples = len(loss1)
    samples_to_remember = int(r_t * total_samples)

    _, ind_for_loss1 = torch.topk(loss2, samples_to_remember, largest=False)
    _, ind_for_loss2 = torch.topk(loss1, samples_to_remember, largest=False)

    loss_filter_1 = torch.zeros((loss1.size(0))).to(device)
    loss_filter_1[ind_for_loss1] = 1.0
    loss1 = (loss_filter_1 * loss1).sum()

    loss_filter_2 = torch.zeros((loss2.size(0))).to(device)
    loss_filter_2[ind_for_loss2] = 1.0
    loss2 = (loss_filter_2 * loss2).sum()

    loss = loss1 + loss2

    return loss


# Loss functions
# def loss_coteaching(
#    logits_1, logits_2, target, forget_rate, criterion=nn.CrossEntropyLoss(reduce=False)
# ):
#    loss_1 = criterion(logits_1, target)
#    ind_1_sorted = torch.argsort(loss_1, dim=0, )
#    loss_1_sorted = loss_1[ind_1_sorted]
#
#    loss_2 = criterion(logits_2, target)
#    ind_2_sorted = torch.tensor(np.argsort(loss_2.data)).cuda()
#    loss_2_sorted = loss_2[ind_2_sorted]
#
#    remember_rate = 1 - forget_rate
#    num_remember = int(remember_rate * len(loss_1_sorted))
#
#    # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(
#    #    num_remember
#    # )
#    # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(
#    #    num_remember
#    # )
#
#    ind_1_update = ind_1_sorted[:num_remember]
#    ind_2_update = ind_2_sorted[:num_remember]
#    # exchange
#    loss_1_update = F.cross_entropy(logits_1[ind_2_update], target[ind_2_update])
#    loss_2_update = F.cross_entropy(logits_2[ind_1_update], target[ind_1_update])
#
#    return (
#        torch.sum(loss_1_update) / num_remember,
#        torch.sum(loss_2_update) / num_remember,
#        # pure_ratio_1,
#        # pure_ratio_2,
#    )
#
