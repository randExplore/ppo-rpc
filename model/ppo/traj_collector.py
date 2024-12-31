import numpy as np
import scipy
import torch
from typing import List


class TrajectoryRecord:
    def __init__(self, num_agent: int,
                 obs_dim_each: List, act_dim_each: List,
                 trajectory_length: int = 400, gamma: float = 0.99, lamda: float = 0.95):
        assert num_agent == len(obs_dim_each), ("Number of agents should be the same "
                                                "as the length of obs_dim_each list!")
        assert num_agent == len(act_dim_each), ("Number of agents should be the same "
                                                "as the length of act_dim_each list!")
        self.num_agent = num_agent
        self.obs_traj = {}
        self.act_traj = {}
        for i in range(num_agent):
            self.obs_traj[i] = np.zeros((trajectory_length, obs_dim_each[i]), dtype=np.float32)
            self.act_traj[i] = np.zeros((trajectory_length, act_dim_each[i]), dtype=np.float32)
        self.adv_traj = np.zeros((num_agent, trajectory_length), dtype=np.float32)
        self.rew_traj = np.zeros((num_agent, trajectory_length), dtype=np.float32)
        self.ret_traj = np.zeros((num_agent, trajectory_length), dtype=np.float32)
        self.val_traj = np.zeros((num_agent, trajectory_length), dtype=np.float32)
        self.logp_traj = np.zeros((num_agent, trajectory_length), dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, trajectory_length

    def add(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        for i in range(self.num_agent):
            self.obs_traj[i][self.ptr] = obs[i]
            self.act_traj[i][self.ptr] = act[i]
            self.logp_traj[i][self.ptr] = logp[i]
            self.val_traj[i][self.ptr] = val[i]
            self.rew_traj[i][self.ptr] = rew[i]
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def last_step(self, last_val):
        path_slice = slice(self.path_start_idx, self.ptr)
        for i in range(self.num_agent):
            vals = np.append(self.val_traj[i][path_slice], last_val[i])
            rews = np.append(self.rew_traj[i][path_slice], last_val[i])
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_traj[i][path_slice] = self.discount_cumsum(deltas, self.gamma * self.lamda)  # GAE calculation
            self.ret_traj[i][path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def normalize(self, x):
        x = np.array(x, dtype=np.float32)
        global_sum, global_n = np.sum(x), x.shape[0]
        mean = global_sum / global_n
        global_sum_sq = np.sum((x - mean) ** 2)
        std = np.sqrt(global_sum_sq / global_n)  # compute std
        return mean, std

    def retrieve_trajectory(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        data_all_agent = []
        for i in range(self.num_agent):
            adv_mean, adv_std = self.normalize(self.adv_traj[i])
            self.adv_traj[i] = (self.adv_traj[i] - adv_mean) / adv_std  # advantage normalization
            data = dict(obs=self.obs_traj[i], act=self.act_traj[i], ret=self.ret_traj[i],
                        adv=self.adv_traj[i], logp=self.logp_traj[i])
            data_all_agent.append({k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()})
        return data_all_agent
