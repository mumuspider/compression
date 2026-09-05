"""
Siren: Implicit Neural Representations with Periodic Activation Functions
Basic coordinate-based neural network for HSI compression
"""

import torch
from torch import nn
import numpy as np


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                             np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        out = torch.sin(self.omega * self.linear(input))
        return out


class Siren(nn.Module):
    """
    Siren: Sinusoidal Representation Networks
    Uses sine activation functions for implicit neural representations
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 first_omega=30,
                 hidden_omega=30.0):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega=first_omega))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega=hidden_omega))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega,
                                        np.sqrt(6 / hidden_features) / hidden_omega)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
