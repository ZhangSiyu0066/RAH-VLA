import torch
import torch.nn as nn
import re
from .ms_vlam_projector import MSVLAMProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        print(x)

        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    use_msvlam = getattr(config, 'use_msvlam', False)

    if use_msvlam:
        # Use MS-VLAM projector for multi-scale alignment
        vision_hidden_size = getattr(config, 'mm_hidden_size', 1024)
        text_hidden_size = getattr(config, 'hidden_size', 4096)
        alignment_dim = getattr(config, 'msvlam_alignment_dim', 768)
        num_sam_regions = getattr(config, 'msvlam_num_regions', 16)
        spp_pool_sizes = getattr(config, 'msvlam_spp_sizes', [1, 2, 4])
        alpha = getattr(config, 'msvlam_alpha', 1.0)
        beta = getattr(config, 'msvlam_beta', 1.0)
        gamma = getattr(config, 'msvlam_gamma', 1.0)
        mu = getattr(config, 'msvlam_mu', 0.5)
        temperature = getattr(config, 'msvlam_temperature', 0.07)
        
        return MSVLAMProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            alignment_dim=alignment_dim,
            num_sam_regions=num_sam_regions,
            spp_pool_sizes=spp_pool_sizes,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mu=mu,
            temperature=temperature
        )

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
