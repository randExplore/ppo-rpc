import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_states):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(num_states, 512)
        self.dense2 = nn.Linear(512, 128)
        self.dense3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v_net = MLP(obs_dim)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)
