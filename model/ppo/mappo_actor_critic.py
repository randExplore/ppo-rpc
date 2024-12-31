import numpy as np
import torch
import torch.nn as nn
from .actor import MLPCategoricalActor, MLPGaussianActor
from .critic import MLPCritic
from typing import List
from gymnasium.spaces import Space, Box, Discrete
# import torch.distributed.rpc as rpc
# import threading


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


# class MAActorCriticParameterServer(nn.Module):
#     def __init__(self, batch_update_size: int,
#                  obs_dim_each: List, act_dim_each: List, num_agent: int,
#                  action_space: List[Union[Box, Discrete]],
#                  device: str = "cpu", actor_lr: float = 1e-4, critic_lr: float = 5e-4,):
#         super().__init__()
#         assert num_agent == len(obs_dim_each), ("Number of agents should be the same "
#                                                 "as the length of obs_dim_each list!")
#         assert num_agent == len(act_dim_each), ("Number of agents should be the same "
#                                                 "as the length of act_dim_each list!")
#         assert num_agent == len(action_space), ("Number of agents should be the same "
#                                                 "as the length of action_space list!")
#         self.batch_update_size = batch_update_size
#         self.device = torch.device(
#             "cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
#         self.pi = []
#         self.num_agent = num_agent
#         self.obs_dim_all = sum(obs_dim_each)
#
#         for i in range(num_agent):
#             if isinstance(action_space[i], Box):
#                 self.pi.append(MLPGaussianActor(obs_dim_each[i], act_dim_each[i]))
#                 self.pi[i].to(self.device)
#             elif isinstance(action_space, Discrete):
#                 self.pi.append(MLPCategoricalActor(obs_dim_each[i], act_dim_each[i]))
#                 self.pi[i].to(self.device)
#
#         # all agent's states are used in the critic net for cooperative tasks
#         self.v = []
#         for i in range(num_agent):
#             self.v.append(MLPCritic(self.obs_dim_all))
#             self.v[i].to(self.device)
#
#         self.lock = threading.Lock()
#         self.future_models = torch.futures.Future()
#         self.batch_update_size = batch_update_size
#         self.curr_update_size = 0
#         self.pi_params = []
#         self.vf_params = []
#         for i in range(num_agent):
#             self.pi_params += list(self.pi[i].parameters())
#             self.vf_params += list(self.v[i].parameters())
#         self.pi_optimizer = torch.optim.Adam(self.pi_params, lr=actor_lr)
#         self.vf_optimizer = torch.optim.Adam(self.vf_params, lr=critic_lr)
#         for p in self.pi_params:
#             p.grad = torch.zeros_like(p)
#         for p in self.vf_params:
#             p.grad = torch.zeros_like(p)
#
#     @staticmethod
#     def step(ps_rref: rpc.RRef, obs_all_agent: torch.Tensor):
#         """
#         remote function
#         calculate actions, log_probability, value function
#         :param ps_rref: parameter server RRef
#         :param obs_all_agent: #  all agent's observations, shape is [num_agent, batch, state_dim]
#         :return: list of actions for all agents, global value functions for all agents, log_prob for all agents' actions
#         """
#         self = ps_rref.local_value()
#         actions = []
#         logp_actions = []
#         obs_all_agent = obs_all_agent.to(self.device)
#         central_obs = obs_all_agent.swapaxes(0, 1)
#         central_obs = central_obs.view(-1,
#                                        self.obs_dim_all)  # shape is [batch, sum of all agents' state_dim]
#         values = []
#         with torch.no_grad():
#             for i in range(self.num_agent):
#                 pi = self.pi[i].distribution(obs_all_agent[i])
#                 a = pi.sample()
#                 logp_a = self.pi[i].log_prob_from_distribution(pi, a)
#                 actions.append(a.to("cpu").numpy())
#                 logp_actions.append(logp_a.to("cpu").numpy())
#                 v = self.v[i](central_obs)
#                 values.append(v.to("cpu").numpy())
#         return actions, values, logp_actions
#
#     def act_greedy(self, obs_all_agent: torch.Tensor):
#         # used for evaluation/test purpose after training is done
#         obs_all_agent = obs_all_agent.to(self.device)
#         actions = []
#         with torch.no_grad():
#             for i in range(self.num_agent):
#                 a = self.pi[i].get_greedy(obs_all_agent[i])
#                 actions.append(a.to("cpu").numpy())
#         return actions
#
#     @staticmethod
#     @rpc.functions.async_execution
#     def train_and_sync_model(ps_rref: rpc.RRef, pi_grads: List, v_grads: List):
#         self = ps_rref.local_value()
#         print(f"ParameterServer got {self.curr_update_size}/{self.batch_update_size} updates")
#         for p, g in zip(self.pi_params, pi_grads):
#             p.grad += g
#         for p, g in zip(self.vf_params, v_grads):
#             p.grad += g
#         with self.lock:
#             self.curr_update_size += 1
#             fut = self.future_models
#             if self.curr_update_size >= self.batch_update_size:
#                 for p in self.pi_params:
#                     p.grad /= self.batch_update_size
#                 for p in self.vf_params:
#                     p.grad /= self.batch_update_size
#
#                 self.curr_update_size = 0
#                 self.pi_optimizer.step()
#                 self.pi_optimizer.zero_grad(set_to_none=False)
#                 self.vf_optimizer.step()
#                 self.vf_optimizer.zero_grad(set_to_none=False)
#
#                 fut.set_result([self.pi, self.v])
#                 print("ParameterServer updated model/agents")
#                 self.future_models = torch.futures.Future()
#
#         return fut
#