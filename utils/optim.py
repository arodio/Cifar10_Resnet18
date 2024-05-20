import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy


class ProxSGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = ProxSGD(model.parameters(), lr=0.1, mu=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, mu=0., momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ProxSGD, self).__init__(params, defaults)

        self.mu = mu

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['initial_params'] = deepcopy(p.data)

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # add proximal term
                d_p.add_(p.data - param_state['initial_params'], alpha=self.mu)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss

    def set_initial_params(self, initial_params):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups = list(initial_params)
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(initial_param_groups[0], dict):
            # TODO: handle multiple optimization groups
            initial_param_groups = [{'params': initial_param_groups}]

        for param_group, initial_param_group in zip(self.param_groups, initial_param_groups):
            for param, initial_param in zip(param_group['params'], initial_param_group['params']):
                param_state = self.state[param]
                param_state['initial_params'] = deepcopy(initial_param.data)
