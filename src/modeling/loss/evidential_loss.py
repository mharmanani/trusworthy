import torch
import torch.nn.functional as F


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl


def loglikelihood_loss(y, alpha):
    # y = y.to(device)
    # alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device):
    # y = y.to(device)
    # alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, reduction='none', device=None):
    if device is None:
        device = torch.device(f'cuda:{output.get_device()}' if torch.cuda.is_available() else 'cpu')
    evidence = relu_evidence(output)
    # evidence = softplus_evidence(output)
    alpha = evidence + 1
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    if reduction == 'none':
        return loss
    return torch.mean(loss)


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, reduction='none', device=None):
    if device is None:
        device = torch.device(f'cuda:{output.get_device()}' if torch.cuda.is_available() else 'cpu')
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    if reduction == 'none':
        return loss
    return torch.mean(loss)


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, reduction='none', device=None):
    if device is None:
        device = torch.device(f'cuda:{output.get_device()}' if torch.cuda.is_available() else 'cpu')
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    if reduction == 'none':
        return loss
    return torch.mean(loss)


class Edl_losses(torch.nn.Module):
    def __init__(self, loss_name, num_classes):
        super(Edl_losses, self).__init__()
        if loss_name == 'edl_mse_loss':
            self.loss = edl_mse_loss
            # self.loss = edl_digamma_loss
            # self.loss = edl_log_loss
        # else:
        #     self.loss = edl_log_loss
        self.num_classes = num_classes

    def forward(self, output, target, epoch_num, reduction='none'):
        if len(target.shape) == 1:
            target = F.one_hot(target, num_classes=self.num_classes)
        return self.loss(output, target, epoch_num, self.num_classes, 50, reduction=reduction).reshape(-1)
