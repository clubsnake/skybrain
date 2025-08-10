import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) recurrent cell.

    Continuous-time formulation (discretized with forward Euler):
        dx/dt = (g(u, x) - x) / tau(u)

    where tau(u) > 0 is an input-dependent time constant and
    g(u, x) is a nonlinear driving function of inputs and hidden state.

    This implementation keeps tau dependent on u for stability and uses
    a simple tanh driving function with input and recurrent terms.

    Args:
        input_dim: dimension of input vector u
        hidden_dim: dimension of hidden state x
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Time constant network tau(u) = softplus(Au + c) + eps
        self.A_tau = nn.Linear(input_dim, hidden_dim)
        self.c_tau = nn.Parameter(torch.zeros(hidden_dim))
        self.act = nn.Tanh()
        self.eps = 1e-3

    def forward(self, u: torch.Tensor, x: torch.Tensor, dt: float = 0.05):
        """
        Forward pass for one time step.

        Args:
            u: (batch, input_dim)
            x: (batch, hidden_dim) previous hidden state
            dt: integration step (seconds)

        Returns:
            x_next: (batch, hidden_dim) next hidden state
            tau: (batch, hidden_dim) time constants
        """
        # Compute time constants (positive)
        tau = F.softplus(self.A_tau(u) + self.c_tau) + self.eps
        # Driving function combines input and recurrent contributions
        g = self.act(self.W_in(u) + self.W_rec(x))
        # LTC dynamics towards g with time constant tau
        x_dot = (g - x) / tau
        x_next = x + dt * x_dot
        return x_next, tau
