import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Union, Tuple

from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel
from movement_primitive_diffusion.models.base_model import BaseModel
from movement_primitive_diffusion.models.scaling import Scaling


class DiffusionModel(BaseModel):
    def __init__(
        self,
        inner_model_config: DictConfig,
        scaling_config: DictConfig,
    ):
        super().__init__()
        self.inner_model: BaseInnerModel = hydra.utils.instantiate(inner_model_config)
        self.scaling: Scaling = hydra.utils.instantiate(scaling_config)

    def loss(self, state: torch.Tensor, action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict, return_denoised: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the loss of the model with the current mini-batch
        Args:
            state: state tensor [batch_size, obs_dim]
            action: Action tensor [batch_size, action_dim]
            sigma: Noise level tensor [batch_size, 1]
            extra_inputs: Extra inputs dictionary

        Returns:

        """

        # Noise is sampled from a normal distribution with mean 0 and std sigma
        for _ in range(action.ndim - sigma.ndim):
            sigma = sigma.unsqueeze(-1)
        assert torch.all(sigma >= 0), "Sigma must be positive"

        # Noise is first drawn from a normal distribution with mean 0 and std 1, then scaled by sigma (the desired std)
        noise = torch.randn_like(action) * sigma

        # Forward process of diffusion probabilistic model. See https://arxiv.org/pdf/2206.00927.pdf and https://arxiv.org/pdf/2206.00364.pdf
        noised_action = action + noise

        # Predict the denoised action
        denoised_action = self.forward(state, noised_action, sigma, extra_inputs)

        # Compute the L2 denoising error. See https://arxiv.org/pdf/2206.00364.pdf
        loss = (action - denoised_action).pow(2).flatten().mean()

        if return_denoised:
            return loss, denoised_action
        else:
            return loss


    def geodesic_distance_matrices(self, m1, m2):
        batch=m1.shape[0]
        time=m1.shape[1]
        m = torch.bmm(m1, m2.transpose(-2,-1)) #batch*3*3
        
        cos = (  m[:,:,0,0] + m[:,:,1,1] + m[:,:,2,2] - 1)/2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch, time).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch, time).cuda())*-1)
        
        theta = torch.acos(cos)
        #theta = torch.min(theta, 2*np.pi - theta)
        
        return theta


    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, applying the noise levels to the input
        Then passing it through the inner model
        Args:
            state: state tensor [batch_size, obs_dim]
            noised_action: Action tensor [batch_size, action_dim]
            sigma: Noise level tensor [batch_size, 1]
            extra_inputs: Extra inputs dictionary

        Returns:
            denoised_action: Denoised action tensor [batch_size, action_dim]

        """
        # Get scaling factors and store c_out to correctly scale the loss
        c_skip, self.c_out, c_in, c_noise = self.scaling(sigma)

        # Compute the denoised action by first passing the (scaled) noised action through the inner model and then applying the scaling factors (including skip connection)
        inner_model_output = self.inner_model(state, c_in * noised_action, c_noise, extra_inputs)
        denoised_action = c_skip * noised_action + self.c_out * inner_model_output

        return denoised_action

class DiffusionModelEDM(BaseModel):
    def __init__(
        self,
        inner_model_config: DictConfig,
        scaling_config: DictConfig,
    ):
        super().__init__()
        self.inner_model: BaseInnerModel = hydra.utils.instantiate(inner_model_config)
        self.scaling: Scaling = hydra.utils.instantiate(scaling_config)

    def loss(self, state: torch.Tensor, action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict, return_denoised: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the loss of the model with the current mini-batch
        Args:
            state: state tensor [batch_size, obs_dim]
            action: Action tensor [batch_size, action_dim]
            sigma: Noise level tensor [batch_size, 1]
            extra_inputs: Extra inputs dictionary

        Returns:

        """

        # Noise is sampled from a normal distribution with mean 0 and std sigma
        for _ in range(action.ndim - sigma.ndim):
            sigma = sigma.unsqueeze(-1)
        assert torch.all(sigma >= 0), "Sigma must be positive"

        # Noise is first drawn from a normal distribution with mean 0 and std 1, then scaled by sigma (the desired std)
        noise = torch.randn_like(action) * sigma

        # Forward process of diffusion probabilistic model. See https://arxiv.org/pdf/2206.00927.pdf and https://arxiv.org/pdf/2206.00364.pdf
        noised_action = action + noise

        # Predict the denoised action
        denoised_action, target = self.forward(state, action, noised_action, sigma, extra_inputs)
        
        loss = (denoised_action - target).pow(2).flatten().mean()

        if return_denoised:
            return loss, denoised_action
        else:
            return loss


    def forward(self, state: torch.Tensor, action: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, applying the noise levels to the input
        Then passing it through the inner model
        Args:
            state: state tensor [batch_size, obs_dim]
            noised_action: Action tensor [batch_size, action_dim]
            sigma: Noise level tensor [batch_size, 1]
            extra_inputs: Extra inputs dictionary

        Returns:
            denoised_action: Denoised action tensor [batch_size, action_dim]

        """
        # Get scaling factors and store c_out to correctly scale the loss
        c_skip, self.c_out, c_in, c_noise = self.scaling(sigma)

        # Compute the denoised action by first passing the (scaled) noised action through the inner model and then applying the scaling factors (including skip connection)
        inner_model_output = self.inner_model(state, c_in * noised_action, c_noise, extra_inputs)

        # https://github.com/intuitive-robots/mdt_policy/blob/e7a78e744751724d3faa5d113fe8cb5f678c052a/mdt/models/edm_diffusion/score_wrappers.py#L45
        target = (action - c_skip * noised_action) / self.c_out
        denoised_action = inner_model_output

        return denoised_action, target
