import torch
import torch.nn as nn

from copy import deepcopy


def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    r"""computes the average of learners and store it into target_learner

    Parameters
    ----------
    learners: List[Learner]

    target_learner: Learner

    weights: 1-D torch.tensor
        tensor of the same size as learners, having values between 0 and 1, and summing to 1,
        if not provided, uniform weights are used

    average_params: bool
        if set to true the parameters are averaged; default is True

    average_gradients: bool
        if set to true the gradient are averaged; default is False

    Returns
    -------
        None
    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    param_tensors = []
    grad_tensors = []

    for learner in learners:
        if average_params:
            param_tensors.append(deepcopy(learner.get_param_tensor()))

        if average_gradients:
            grad_tensors.append(deepcopy(learner.get_grad_tensor()))

    if average_params:
        param_tensors = torch.stack(param_tensors)
        average_params_tensor = weights @ param_tensors
        target_learner.set_param_tensor(average_params_tensor)

    if average_gradients:
        grad_tensors = torch.stack(grad_tensors)
        average_grads_tensor = weights @ grad_tensors
        target_learner.set_grad_tensor(average_grads_tensor)


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())
