import torch
import torch.nn as nn
import torch.nn.functional as F

class LTCCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.a = nn.Linear(input_dim, hidden_dim)  # for tau(u)
        self.c = nn.Parameter(torch.zeros(hidden_dim))
        self.act = nn.Tanh()
        self.eps = 1e-3

    def forward(self, u, x, dt=0.05):
        tau = F.softplus(self.a(u) + self.c) + self.eps
        x_dot = -x + self.act(self.W_in(u))
        x_next = x + dt * (x_dot / tau)
        return x_next, tau
