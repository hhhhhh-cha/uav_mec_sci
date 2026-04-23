import torch
import torch.nn as nn


class MLPActor(nn.Module):
    """
    Simple MLP actor for the current trainable proposed-policy prototype.

    Input:
        obs: [batch_size, obs_dim]

    Output:
        raw_action: [batch_size, action_dim]

    Notes:
    - This outputs unconstrained raw values.
    - Actual action ranges are handled by ProposedLearnedPolicy._decode_raw_action().
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)