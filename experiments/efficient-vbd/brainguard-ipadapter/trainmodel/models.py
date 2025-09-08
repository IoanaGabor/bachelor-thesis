# This file is an adaptation of https://github.com/kunzhan/BrainGuard/blob/main/trainmodel/models.py

import torch
import torch.nn as nn
from functools import partial

class BrainGuardModule(nn.Module):
    def __init__(self):
        super(BrainGuardModule, self).__init__()
    def forward(self, x):
        return x
    
class RidgeRegression(torch.nn.Module):
    def __init__(self, input_size, out_features): 
        super(RidgeRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, out_features)
    def forward(self, x):
        x = self.linear(x)
        return x

class BrainNetwork(nn.Module):
  def __init__(self, out_dim_image=768, in_dim=15724, latent_size=768, h=2048, n_blocks=4, norm_type='ln', use_projector=True, act_first=False, drop1=.5, drop2=.15, train_type='vision'):
    super().__init__()
    norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
    act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
    act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
    self.mlp = nn.ModuleList([
        nn.Sequential(
            nn.Linear(h, h),
            *[item() for item in act_and_norm],
            nn.Dropout(drop2)
        ) for _ in range(n_blocks)
    ])
    self.head_image = nn.Linear(h, out_dim_image, bias=True)
    self.n_blocks = n_blocks
    self.latent_size = latent_size
    self.use_projector = use_projector
    self.train_type = train_type
    if use_projector:
        self.projector_image = nn.Sequential(
        nn.LayerNorm(self.latent_size),
        nn.GELU(),
        nn.Linear(self.latent_size, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, 2048),
        nn.LayerNorm(2048),
        nn.GELU(),
        nn.Linear(2048, self.latent_size)
)
        
  def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x_image = self.head_image(x) 

        if self.use_projector: 
            return self.projector_image(x_image.reshape(len(x_image), -1, self.latent_size))
        return x
