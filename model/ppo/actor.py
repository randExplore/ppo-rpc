import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np


class MLP(nn.Module):
    def __init__(self, num_states, num_actions):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(num_states, 512)
        self.dense2 = nn.Linear(512, 128)
        self.dense3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x)


class MLPCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.logits_net = MLP(obs_dim, act_dim)

    def distribution(self, obs):
        logits = self.logits_net(obs)
        prob = F.softmax(logits, dim=-1)
        return Categorical(probs=prob)

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def get_greedy(self, obs):
        logits = self.logits_net(obs)
        prob = F.softmax(logits, dim=-1)
        return torch.argmax(prob, dim=-1)

    def forward(self, obs, act=None):
        pi = self.distribution(obs)
        logp_act = None
        if act is not None:
            logp_act = self.log_prob_from_distribution(pi, act)
        return pi, logp_act


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_upper_limit):
        super().__init__()
        log_std = -1.0 * np.ones(act_dim, dtype=np.float32)
        self.act_upper_limit = torch.from_numpy(act_upper_limit)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLP(obs_dim, act_dim)

    def distribution(self, obs):
        self.act_upper_limit = self.act_upper_limit.to(obs.device)
        mu = self.act_upper_limit * F.tanh(self.mu_net(obs)) # limit the action value to its act_limit range
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def get_greedy(self, obs):
        mu = self.mu_net(obs)
        return mu

    def forward(self, obs, act=None):
        pi = self.distribution(obs)
        logp_act = None
        if act is not None:
            logp_act = self.log_prob_from_distribution(pi, act)
        return pi, logp_act