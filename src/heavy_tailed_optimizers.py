"""
Heavy-Tailed Stochastic Optimization Algorithms
===============================================
Implementation of optimization methods under heavy-tailed stochastic gradients:
- Standard Stochastic Gradient Descent (SGD)
- Clipped Stochastic Gradient Descent (Clipped-SGD)
- Normalized Stochastic Gradient Descent (Norm-SGD)

Theoretical setting:
E[||g_t||^p | w_t] <= G^p for p in (1, 2], where second-moment (variance) may be unbounded.
"""

import numpy as np

class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, w, grad, iteration=1):
        raise NotImplementedError


class StandardSGD(Optimizer):
    r"""
    Standard SGD: w_{t+1} = w_t - \eta_t * g_t
    Vulnerable when gradient noise exhibits heavy tails or unbounded variance.
    """
    def __init__(self, lr=0.01, lr_decay=0.5):
        super().__init__(lr)
        self.lr_decay = lr_decay

    def step(self, w, grad, iteration=1):
        eta_t = self.lr / (iteration ** self.lr_decay)
        return w - eta_t * grad


class ClippedSGD(Optimizer):
    r"""
    Clipped SGD:
    w_{t+1} = w_t - \eta_t * clip(g_t, \tau_t)
    clip(g, \tau) = min(1, \tau / ||g||_2) * g

    Under bounded p-th moments (p in (1, 2]), gradient clipping balances the
    bias-variance trade-off and ensures stable convergence rates O(T^{-(p-1)/(2p-1)}).
    """
    def __init__(self, lr=0.01, clip_threshold=1.0, lr_decay=0.5, adaptive_threshold=False, p_moment=1.5):
        super().__init__(lr)
        self.clip_threshold = clip_threshold
        self.lr_decay = lr_decay
        self.adaptive_threshold = adaptive_threshold
        self.p_moment = p_moment

    def step(self, w, grad, iteration=1):
        grad_norm = np.linalg.norm(grad) + 1e-12
        
        # Theoretical threshold scaling if adaptive: tau_t ~ t^{(p-1)/(2p-1)}
        if self.adaptive_threshold:
            exponent = (self.p_moment - 1.0) / (2.0 * self.p_moment - 1.0)
            tau_t = self.clip_threshold * (iteration ** exponent)
        else:
            tau_t = self.clip_threshold

        scaling = min(1.0, tau_t / grad_norm)
        clipped_grad = scaling * grad
        eta_t = self.lr / (iteration ** self.lr_decay)
        return w - eta_t * clipped_grad


class NormalizedSGD(Optimizer):
    r"""
    Normalized SGD:
    w_{t+1} = w_t - \eta_t * (g_t / (||g_t||_2 + eps))
    Resilient to arbitrary gradient scales, useful in non-smooth or heavy-tailed regimes.
    """
    def __init__(self, lr=0.01, eps=1e-8, lr_decay=0.5):
        super().__init__(lr)
        self.eps = eps
        self.lr_decay = lr_decay

    def step(self, w, grad, iteration=1):
        grad_norm = np.linalg.norm(grad) + self.eps
        norm_grad = grad / grad_norm
        eta_t = self.lr / (iteration ** self.lr_decay)
        return w - eta_t * norm_grad
