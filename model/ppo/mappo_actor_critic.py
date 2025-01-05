import numpy as np
import torch
import torch.nn as nn
from .actor import MLPCategoricalActor, MLPGaussianActor
from .critic import MLPCritic
from typing import List
from gymnasium.spaces import Space, Box, Discrete


class MAActorCriticMLP(nn.Module):
    def __init__(self,
                 obs_dim_each: List, num_agent: int,
                 action_space: List[Space],
                 device: str = "cpu"):
        super().__init__()
        assert num_agent == len(obs_dim_each), ("Number of agents should be the same "
                                                "as the length of obs_dim_each list!")
        assert num_agent == len(action_space), ("Number of agents should be the same "
                                                "as the length of action_space list!")
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pi = []
        self.num_agent = num_agent
        self.obs_dim_all = sum(obs_dim_each)

        for i in range(num_agent):
            if isinstance(action_space[i], Box):
                self.pi.append(MLPGaussianActor(obs_dim_each[i], action_space[i].shape[0], action_space[i].high))
                self.pi[i].to(self.device)
            elif isinstance(action_space[i], Discrete):
                self.pi.append(MLPCategoricalActor(obs_dim_each[i], action_space[i].n))
                self.pi[i].to(self.device)

        # all agent's states are used in the critic net for cooperative tasks
        self.v = []
        for i in range(num_agent):
            self.v.append(MLPCritic(self.obs_dim_all))
            self.v[i].to(self.device)
