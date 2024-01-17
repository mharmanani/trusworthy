import torch


def compute_temperature_and_bias_for_calibration(prob, y, mode="ce", lr=1e-2, max_iter=10000):
    import numpy as np

    raw_logits = torch.log(prob / (1 - prob))
    target = torch.tensor(y, dtype=torch.float32).squeeze()
    raw_logits = raw_logits.squeeze()

    temp = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

    bias = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

    from torch.nn.functional import binary_cross_entropy

    optimizer = torch.optim.LBFGS([temp, bias], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        output = raw_logits / temp
        output = output + bias
        output = torch.sigmoid(output)

        ind_cancer = target == 1
        ind_normal = target == 0

        if mode == "brier":
            loss = (output - target).pow(2)
            loss[ind_cancer] *= ind_normal.sum() / ind_cancer.sum()

        elif mode == "ce":
            loss = binary_cross_entropy(output, target, reduction="none")
            loss[ind_cancer] *= ind_normal.sum() / ind_cancer.sum()

        #elif mode == "nll":
        #    # get nll loss
        #    preds = output > 0.5
        #    

        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        loss = loss.mean()

        loss.backward()
        return loss

    loss_init = closure()
    optimizer.step(closure)

    # check that loss is decreasing
    loss_final = closure()

    if loss_final > loss_init:
        print(f"Loss increased from {loss_init} to {loss_final}")
        print(f"Using initial values")
        return 1.0, 0.0

    return temp.data, bias.data


def apply_temperature_and_bias(prob, temp, bias, return_numpy=True):
    # convert to torch if necessary
    if not isinstance(prob, torch.Tensor):
        prob = torch.tensor(prob)

    logits = torch.log(prob / (1 - prob))
    logits = logits / temp + bias
    prob = torch.sigmoid(logits)

    return prob.numpy() if return_numpy else prob


 