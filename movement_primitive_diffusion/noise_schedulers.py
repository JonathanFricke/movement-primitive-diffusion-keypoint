import math
from abc import ABC, abstractmethod

import numpy as np
import torch


def append_zero(action):
    return torch.cat([action, action.new_zeros([1])])


class NoiseScheduler(ABC):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype

    @abstractmethod
    def get_sigmas(self, n, *args, **kwargs):
        raise NotImplementedError


class KarrasNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float, rho=7.0, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n, dtype=self.dtype)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas


class ExponentialNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), n, dtype=self.dtype).exp()
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """Constructs an linear noise schedule."""
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, n, dtype=self.dtype)
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas


class NNScheduler(NoiseScheduler, torch.nn.Module):
    def __init__(self, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.append_zero = append_zero

        self.lin1 = DenseMonotone(1, 1)
        self.lin2 = DenseMonotone(1, 256)
        self.lin3 = DenseMonotone(256, 1, bias=False)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_sigmas(self, n):
        """Constructs an exponential noise schedule."""
        # sigmas = torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), n, dtype=self.dtype).exp()
        # if self.append_zero:
        #     return append_zero(sigmas)
        # else:
        #     return sigmas
        sigmas = self.lin1(n)

        if True:
            sigmas_ = 2. * (n - .5)  # scale input to [-1, +1]
            sigmas_ = self.lin2(sigmas_)
            sigmas_ = 2 * (torch.nn.Sigmoid(sigmas_) - .5)  # more stable than jnp.tanh(h)
            sigmas_ = self.lin3(sigmas_) / self.n_features
            sigmas += sigmas_
        return sigmas


        # sigmas = self.encoder(n)
        # if self.append_zero:
        #     return append_zero(sigmas)
        # else:
        #     return sigmas


class DenseMonotone(torch.nn.Module):
    """
    Strictly increasing Dense layer.
    
    This layer behaves like a fully-connected (linear) layer but forces the weight matrix
    to be nonnegative (by taking its absolute value) during the forward pass.
    """
    def __init__(self, in_features, out_features, bias=True, 
                 kernel_init=None, bias_init=None, dtype=torch.float32):
        """
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If set to False, the layer will not learn an additive bias.
            kernel_init (callable, optional): A function to initialize the weight matrix.
            bias_init (callable, optional): A function to initialize the bias.
            dtype (torch.dtype): Data type for the parameters.
        """
        super(DenseMonotone, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.dtype = dtype

        # Initialize weight parameter.
        # We create an uninitialized parameter and then apply the initializer.
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features, dtype=self.dtype))
        if kernel_init is not None:
            kernel_init(self.weight)
        else:
            # Default initializer similar to Kaiming uniform.
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize bias parameter if needed.
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=self.dtype))
            if bias_init is not None:
                bias_init(self.bias)
            else:
                # Default initialization for bias.
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        # Ensure inputs are in the correct dtype.
        inputs = inputs.to(self.dtype)
        # Enforce non-negativity on the kernel weights.
        weight = torch.abs(self.weight)
        # Standard linear operation: dot-product of inputs and kernel.
        y = torch.matmul(inputs, weight)
        if self.use_bias:
            y = y + self.bias.to(self.dtype)
        return y


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, s=0.008, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.s = s
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        also called squaredcos_cap_v2
        """
        steps = n + 1
        # TODO: From 0 to N with step size N?
        x = torch.linspace(0, steps, steps, dtype=self.dtype)
        alphas_cumprod = torch.cos(((x / steps) + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # TODO: Should this be clipped to sigma_min and sigma_max instead of fixed values?
        betas_clipped = torch.clamp(betas, min=0.0, max=0.999)
        # TODO: Flip needs an dims argument
        sigmas = torch.flip(betas_clipped, dims=[0])
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas
        

class VENoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """Constructs a continuous VE noise schedule."""
        # (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        steps = n + 1
        t = torch.linspace(0, steps, n, dtype=self.dtype)
        t = (self.sigma_max**2) * ((self.sigma_min**2 / self.sigma_max**2) ** (t / (n - 1)))
        sigmas = torch.sqrt(t)
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas


class IDDPMNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float, M=1000, j_0=0, C_1=0.001, C_2=0.008, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.M = M
        self.j_0 = j_0
        self.C_1 = C_1
        self.C_2 = C_2
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """Constructs a continuous IDDPM noise schedule."""
        # (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        step_indices = torch.arange(n, dtype=self.dtype)
        u = torch.zeros(self.M + 1, dtype=self.dtype)
        alpha_bar = lambda j: (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2
        for j in torch.arange(self.M, self.j_0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self.C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= self.sigma_min, u <= self.sigma_max)]
        sigmas = u_filtered[((len(u_filtered) - 1) / (n - 1) * step_indices).round().to(torch.int64)]
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas


class VPNoiseScheduler(NoiseScheduler):
    def __init__(self, beta_d=19.9, beta_min=0.1, eps_s=1e-3, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.eps_s = eps_s

    def get_sigmas(self, n):
        """Constructs a continuous VP noise schedule."""
        t = torch.linspace(1, self.eps_s, n, dtype=self.dtype)
        sigmas = torch.sqrt(torch.exp(self.beta_d * t**2 / 2 + self.beta_min * t) - 1)
        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas


class PolyNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float, rho=1.0, dtype=torch.float32, append_zero: bool = True):
        super().__init__(dtype)
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.append_zero = append_zero

    def get_sigmas(self, n):
        """Constructs an polynomial in log sigma noise schedule."""
        ramp = torch.linspace(1, 0, n, dtype=self.dtype) ** self.rho
        sigmas = torch.exp(ramp * (math.log(self.sigma_max) - math.log(self.sigma_min)) + math.log(self.sigma_min))

        if self.append_zero:
            return append_zero(sigmas)
        else:
            return sigmas

