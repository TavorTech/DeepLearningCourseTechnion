import abc
import torch
from torch import Tensor


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Layers, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            #  Update the gradient according to regularization and then
            #  update the parameters tensor.
            # ====== YOUR CODE: ======
            with torch.no_grad():
                dp.add_(self.reg * p)           # L2 reg
                p.sub_(self.learn_rate * dp)    # parameter update
            # ========================


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        # Initialize a velocity term for each parameter
        self.velocities = {id(p): torch.zeros_like(p) for p, _ in self.params}
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.
            # ====== YOUR CODE: ======
            with torch.no_grad():
                # Add L2 regularization term to gradient
                dp.add_(self.reg * p)

                # Update velocity: v = momentum * v - learn_rate * dp
                v = self.velocities[id(p)]
                v.mul_(self.momentum).sub_(self.learn_rate * dp)

                # Update parameters: p = p + v
                p.add_(v)
            # ========================


class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.9, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        # A list or dictionary that will store the moving average of the squared gradients
        # for each parameter. We often call this 'r' or 'cache' in tutorials.
        self.r_cache = []
        for p, dp in self.params:
            # Initialize r to zeros of the same shape as p
            self.r_cache.append(torch.zeros_like(p))
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.
            # ====== YOUR CODE: ======
            for i, (p, dp) in enumerate(self.params):
                if dp is None:
                    continue
    
                # 1) Add L2 regularization term to the gradient
                dp = dp + self.reg * p  # dtheta_t += lambda * theta_t
    
                # 2) Update the moving average of the squared gradients
                self.r_cache[i] = self.decay * self.r_cache[i] + (1 - self.decay) * (dp * dp)
    
                # 3) Compute the per-parameter RMSProp scaling factor
                rms = torch.sqrt(self.r_cache[i] + self.eps)
    
                # 4) Update the parameter: theta_{t+1} = theta_t - (lr / rms) * dtheta_t
                p -= self.learn_rate * (dp / rms)
            # ========================
