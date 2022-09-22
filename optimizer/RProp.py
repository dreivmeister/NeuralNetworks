from optimizer import Optimizer
import numpy as np
class RProp(Optimizer):
    def __init__(self, eta_p, eta_n):
        self.eta_p = eta_p
        self.eta_n = eta_n

    def update(self, W_grads, W, W_prev_sign, W_delta):
        """
        Update RProp values in one iteration.
        Args:
            X: input data.
            t: targets.
            W: Current weight parameters.
            W_prev_sign: Previous sign of the W gradient.
            W_delta: RProp update values (Delta).
            eta_p, eta_n: RProp hyperparameters.
        Returns:
            (W_delta, W_sign): Weight update and sign of last weight
                            gradient.
        """
        W_sign = np.sign(W_grads)
        for i, _ in enumerate(W):
            if W_sign[i] == W_prev_sign[i]:
                W_delta[i] *= self.eta_p
            else:
                W_delta[i] *= self.eta_n
        return W_delta, W_sign

