#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDPM with conditional generation.

@author: mrjohn
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from unet import real_to_complex, complex_to_real

from tqdm import tqdm

#%% Variance schedule for DDPM
def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,                        # \alpha_t
        "oneover_sqrta": oneover_sqrta,            # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,                # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,                  # \bar{\alpha_t}
        "sqrtab": sqrtab,                          # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,                        # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


#%% Implementation of DDPM (see Ho et al. paper)
class DDPM(nn.Module):
    def __init__(self, eps_model, betas, n_T, criterion=nn.MSELoss(), complex_in=False, device=None, n_classes=4):
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion
        self.complex_in = complex_in
        self.drop_prob = 0.1
        self.n_classes = n_classes
        self.device = device


    def forward(self, x, c):
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        if self.complex_in: eps = eps + torch.randn_like(x)*1j

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        
        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        eps_pred = self.eps_model(x_t, _ts / self.n_T, c, context_mask)
        if self.complex_in:
            return self.criterion(complex_to_real(eps), complex_to_real(eps_pred))
        else:
            return self.criterion(eps, eps_pred)


    def sample(self, n_sample: int, size) -> torch.Tensor:
        """
        Starts from noise sample and performs reverse diffusion.
        This implements Algorithm 2 in the paper.
        """
        x_i = torch.randn(n_sample, *size) 
        if self.complex_in: x_i = x_i + torch.randn(n_sample, *size)*1j
        x_i = x_i.to(self.device)  # x_T ~ N(0, 1)
        
        if n_sample >= self.n_classes:
            c_i = torch.arange(0, self.n_classes).to(self.device) # context for us just cycles throught the labels
            c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        else:
            c_i = torch.randint(0, self.n_classes, (n_sample,)).to(self.device)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(self.device)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        with tqdm(total=self.n_T, position=0, leave=True) as sample_pbar:
            for i in range(self.n_T, 0, -1):
                if i > 1: 
                    z = torch.randn(n_sample, *size).to(self.device) 
                    if self.complex_in:
                        z = z + torch.randn(n_sample, *size).to(self.device) * 1j
                else:
                    z = 0.

                eps = self.eps_model(x_i, torch.tensor(i).to(self.device) / self.n_T, c_i, context_mask)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                sample_pbar.update(1)
        sample_pbar.close()

        return x_i
       
    
    def sample_all(self, size):
        """
        Return samples from various timesteps to visualize and create animations.
        Implements Algorithm 2.
        """
        
        x_list, nT_list = [], []
        c_i = torch.arange(0, self.n_classes).to(self.device)
        x_i = torch.randn(1, *size) 
        if self.complex_in: x_i = x_i + torch.randn(1, *size)*1j
        
        x_i = torch.tile(x_i, (self.n_classes, 1,1,1))
        
        x_i = x_i.to(self.device)  # x_T ~ N(0, 1)
        x_list.append(x_i.detach().cpu().numpy())
        nT_list.append(self.n_T)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(self.device)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        with tqdm(total=self.n_T, position=0, leave=True) as sample_pbar:
            for i in range(self.n_T, 0, -1):
                if i > 1: 
                    z = torch.randn(1, *size) 
                    if self.complex_in: z = z + torch.randn(1, *size)*1j
                    z = torch.tile(z, (self.n_classes, 1,1,1))
                    z = z.to(self.device)
                else:
                    z = 0.

                eps = self.eps_model(x_i, torch.tensor(i).to(self.device) / self.n_T, c_i, context_mask)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

                if i%20 == 0 or i < 20: 
                    x_list.append(x_i.detach().cpu().numpy())
                    nT_list.append(i)
                    
                sample_pbar.update(1)
        sample_pbar.close()
                
        x_list.append(x_i.detach().cpu().numpy())
        nT_list.append(0)

        return x_list, nT_list
    
    
    def sample_compare(self, n_samples, size):
        """
        Generate a large number of samples. 
        Implements Algorithm 2.
        """
        x_list = np.zeros((self.n_classes, n_samples, size[-2], size[-1]))

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for j in range(n_samples):
            c_i = torch.arange(0, self.n_classes).to(self.device)
            x_i = torch.randn(1, *size) 
            if self.complex_in: x_i = x_i + torch.randn(1, *size)*1j

            x_i = torch.tile(x_i, (self.n_classes, 1,1,1))
            x_i = x_i.to(self.device)  # x_T ~ N(0, 1)

            # don't drop context at test time
            context_mask = torch.zeros_like(c_i).to(self.device)
            with tqdm(total=self.n_T, position=0, leave=True) as sample_pbar:
                sample_pbar.set_description(f'[Sample {j+1} of {n_samples}]')
                for i in range(self.n_T, 0, -1):
                    if i > 1: 
                        z = torch.randn(1, *size) 
                        if self.complex_in: z = z + torch.randn(1, *size)*1j
                        z = torch.tile(z, (self.n_classes, 1,1,1))
                        z = z.to(self.device)
                    else:
                        z = 0.

                    eps = self.eps_model(x_i, torch.tensor(i).to(self.device) / self.n_T, c_i, context_mask)
                    x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                    )
                    sample_pbar.update(1)
                
                x_list[:,j:j+1,:,:] = x_i.detach().cpu().numpy()    
            sample_pbar.close()

        return x_list