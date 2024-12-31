import os
from tqdm import tqdm
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.distributed.rpc as rpc
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.spaces import Box
import random
import vmas

from .mappo_actor_critic import MAActorCriticMLP
from .traj_collector import TrajectoryRecord

import warnings
warnings.filterwarnings('ignore')

ENV_RUNNER_NAME = "env_runner_{}"


class EnvRunner:
    def __init__(self, env_name: str, num_steps_per_episode: int, num_agent: int,
                 gamma: float = 0.99, lamda: float = 0.95):
        self.num_agent = num_agent
        self.rank = rpc.get_worker_info().id - 1
        self.num_steps_per_episode = num_steps_per_episode
        self.env = self.make_env(self.rank, env_name, self.num_agent)
        self.obs_dim_each, self.action_space_each, self.act_dim_each = self.get_obs_act_dims(num_agent, self.env)
        self.trajectory = TrajectoryRecord(num_agent, self.obs_dim_each, self.act_dim_each, num_steps_per_episode,
                                           gamma=gamma, lamda=lamda)
        self.select_action = PPOTrainer.step

    @staticmethod
    def get_obs_act_dims(num_agent, env):
        obs_dim_each = []
        action_space_each = []
        act_dim_each = []
        if num_agent == 1:
            obs_dim_each.append(env.observation_space.shape[0])
            action_space_each.append(env.action_space)
            if len(env.action_space.shape) > 0:
                act_dim_each.append(env.action_space.shape[0])
            else:
                act_dim_each.append(1)
        else:
            for i in range(num_agent):
                obs_dim_each.append(env.observation_space[i].shape[0])
                action_space_each.append(env.action_space[i])
                if len(env.action_space[i].shape) > 0:
                    act_dim_each.append(env.action_space[i].shape[0])
                else:
                    act_dim_each.append(1)

        return obs_dim_each, action_space_each, act_dim_each

    @staticmethod
    def make_env(rank, layout, num_agent,
                 num_steps_per_episode=100,
                 render_mode=None,
                 is_testing_mode=False):
        random.seed(rank)
        if num_agent == 1:
            if layout == "LunarLander-v3":
                env = gym.make(layout, continuous=True, render_mode=render_mode)
            else:
                env = gym.make(layout, render_mode=render_mode)
        else:
            env = vmas.make_env(
                scenario=layout,
                num_envs=1,
                device="cpu",
                continuous_actions=True,
                wrapper="gymnasium",
                max_steps=num_steps_per_episode,
                terminated_truncated=True,
                n_agents=num_agent,
                is_testing_mode=is_testing_mode
            )
        seed = random.randint(0, 10000)
        env.reset(seed=seed)
        return env

    @staticmethod
    def process_reset(num_agent, env):
        states, info = env.reset()
        if num_agent == 1:
            state = states
        else:
            state = []
            for i in range(num_agent):
                cur_state = states[i]
                state.append(cur_state.reshape(1, -1))
        return state

    @staticmethod
    def process_act(num_agent, acts, action_space_each):
        if num_agent == 1:
            actions = acts[0]
            if isinstance(action_space_each[0], Box):
                actions = np.clip(actions, action_space_each[0].low, action_space_each[0].high)
            if actions.shape[0] == 1:
                return actions[0]
            else:
                return actions
        else:
            actions = [act[0] for act in acts]
            for i in range(len(actions)):
                if isinstance(action_space_each[i], Box):
                    actions[i] = np.clip(actions[i], action_space_each[i].low, action_space_each[i].high)
            return actions

    @staticmethod
    def process_obs(num_agent, state):
        if num_agent <= 1:
            obs = np.reshape(state,
                             [1, 1, state.shape[0]])  # shape is [num_agent, 1, obs_dim]
        else:
            obs = np.array(state)  # shape is [num_agent, 1, obs_dim]
        return obs

    @staticmethod
    def process_act_step(env, num_agent, actions):
        observation, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        next_state = []
        if num_agent == 1:
            next_state = observation
        else:
            for i in range(num_agent):
                next_state.append(observation[i].reshape(1, -1))
        r = np.array([reward]).reshape(-1, 1)
        return next_state, r, done

    def get_episode(self, agent_rref):
        cur_epi_reward = 0
        cur_epi_reward_all = []

        state = self.process_reset(self.num_agent, self.env)

        ep_len = 0
        ep_len_all = []
        for t in range(self.num_steps_per_episode):
            # Obtain observations for each agent
            obs = self.process_obs(self.num_agent, state)
            acts, v_vals, logps = rpc.rpc_sync(
                agent_rref.owner(),
                self.select_action,
                args=(agent_rref, torch.as_tensor(obs, dtype=torch.float32))
            )
            actions = self.process_act(self.num_agent, acts, self.action_space_each)
            next_state, r, done = self.process_act_step(self.env, self.num_agent, actions)

            ep_len += 1
            a = np.array([actions]).reshape(self.num_agent, 1, self.act_dim_each[0])
            logps = np.array(logps)

            v_vals = np.array([val[0] for val in v_vals]).reshape(-1, 1)
            cur_epi_reward += np.sum(r)
            self.trajectory.add(obs, a, r, v_vals, logps)
            state = next_state
            timeout = (ep_len == self.num_steps_per_episode)
            terminal = done or timeout
            if terminal:
                if timeout:
                    obs = self.process_obs(self.num_agent, state)
                    _, v, _ = rpc.rpc_sync(agent_rref.owner(),
                                           self.select_action,
                                           args=(agent_rref, torch.as_tensor(obs, dtype=torch.float32)))
                else:
                    v = [0] * self.num_agent
                self.trajectory.last_step(v)

                # reset to start the new episode
                state = self.process_reset(self.num_agent, self.env)
                ep_len_all.append(ep_len)
                cur_epi_reward_all.append(cur_epi_reward)
                ep_len = 0
                cur_epi_reward = 0

        data_episode = self.trajectory.retrieve_trajectory()

        return [data_episode,
                np.mean(cur_epi_reward_all) if cur_epi_reward == 0 else np.mean(cur_epi_reward_all + [cur_epi_reward]),
                np.mean(ep_len_all) if ep_len == 0 else np.mean(ep_len_all + [ep_len])]


class PPOTrainer:
    def __init__(self, world_size: int,
                 num_agent: int, env_name: str,
                 num_steps_per_episode: int = 1000, device: str = "cpu",
                 clip_ratio: float = 0.2,
                 actor_lr: float = 1e-4, critic_lr: float = 5e-4, train_actor_epoch: int = 80,
                 train_critic_epoch: int = 80, kl_target: float = 0.01, alpha: float = 5e-3,
                 save_model: bool = True, save_root_dir: str = os.getcwd(), seed: int = 1, only_evaluate: bool = False):
        self.env_name = env_name
        self.num_steps_per_episode = num_steps_per_episode
        self.testing_steps = 1000 if env_name == "road_traffic" else self.num_steps_per_episode

        self.env_testing = EnvRunner.make_env(0, env_name, num_agent,
                                              num_steps_per_episode=self.testing_steps,
                                              render_mode="rgb_array",
                                              is_testing_mode=True)  # used for evaluation
        self.num_agent = num_agent
        self.obs_dim_each, action_space_each, _ = EnvRunner.get_obs_act_dims(num_agent, self.env_testing)
        self.action_space_each = action_space_each

        # setup agents' network
        self.set_seed(seed)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.ma_ppo_actor_critic = MAActorCriticMLP(self.obs_dim_each,
                                                    num_agent, action_space_each, device)

        # setup agents' network training
        self.clip_ratio = clip_ratio
        self.train_actor_epoch = train_actor_epoch
        self.train_critic_epoch = train_critic_epoch
        self.kl_target = kl_target
        self.pi_params = []
        self.vf_params = []
        for i in range(num_agent):
            self.pi_params += list(self.ma_ppo_actor_critic.pi[i].parameters())
            self.vf_params += list(self.ma_ppo_actor_critic.v[i].parameters())
        self.pi_optimizer = optim.Adam(self.pi_params, lr=actor_lr)
        self.vf_optimizer = optim.Adam(self.vf_params, lr=critic_lr)
        self.save_model = save_model
        self.save_dir = os.path.join(save_root_dir, "Results", 'ppo', env_name, "Checkpoints")
        self.saved_logs_path = os.path.join(save_root_dir, "Results", 'ppo', env_name, "Summary")
        os.makedirs(self.saved_logs_path, exist_ok=True)
        self.writer = SummaryWriter(self.saved_logs_path)
        self.load_model()
        self.alpha = alpha  # it's used in the entropy loss term

        # setup environment runners
        if not only_evaluate:
            self.agent_rref = rpc.RRef(self)
            self.env_runner_rrefs = []
            for env_rank in range(1, world_size):
                ob_info = rpc.get_worker_info(ENV_RUNNER_NAME.format(env_rank))
                self.env_runner_rrefs.append(rpc.remote(ob_info, EnvRunner, args=(env_name, num_steps_per_episode,
                                                                                  num_agent,)))

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def step(agent_rref: rpc.RRef, obs_all_agent: torch.Tensor):
        """
        remote function for observers
        calculate actions, log_probability, value function
        :param agent_rref: agent's RRef
        :param obs_all_agent: #  all agent's observations, shape is [num_agent, batch, state_dim]
        :return: list of actions for all agents, global value functions for all agents, log_prob for all agents' actions
        """
        self = agent_rref.local_value()
        actions = []
        logp_actions = []
        obs_all_agent = obs_all_agent.to(self.device)
        central_obs = obs_all_agent.swapaxes(0, 1)
        central_obs = central_obs.view(-1,
                                       sum(self.obs_dim_each))  # shape is [batch, num_agent * state_dim]
        values = []
        with torch.no_grad():
            for i in range(self.num_agent):
                pi = self.ma_ppo_actor_critic.pi[i].distribution(obs_all_agent[i])
                a = pi.sample()
                logp_a = self.ma_ppo_actor_critic.pi[i].log_prob_from_distribution(pi, a)
                actions.append(a.to("cpu").numpy())
                logp_actions.append(logp_a.to("cpu").numpy())
                v = self.ma_ppo_actor_critic.v[i](central_obs)
                values.append(v.to("cpu").numpy())
        return actions, values, logp_actions

    def act_greedy(self, obs_all_agent):
        obs_all_agent = obs_all_agent.to(self.device)
        actions = []
        with torch.no_grad():
            for i in range(self.num_agent):
                a = self.ma_ppo_actor_critic.pi[i].get_greedy(obs_all_agent[i])
                if isinstance(self.action_space_each[i], Box):
                    a = torch.max(torch.min(a, torch.tensor(self.action_space_each[i].high).to(a.device)),
                                  torch.tensor(self.action_space_each[i].low).to(a.device))
                actions.append(a.to("cpu").numpy())
        return actions

    def compute_loss_pi(self, data_all_agent):
        loss_pi_all, approx_kl_all, entropy_all = [], [], []
        for i in range(self.num_agent):
            obs, act, advantage, logp_old = (data_all_agent[i]['obs'],
                                             data_all_agent[i]['act'], data_all_agent[i]['adv'],
                                             data_all_agent[i]['logp'])
            act = act.squeeze(-1)
            # Calculate policy loss for each agent
            pi, logp = self.ma_ppo_actor_critic.pi[i](obs, act)
            ratio = torch.exp(logp - logp_old)
            clipped_advantage = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
            if len(pi.entropy().shape) > 1:
                ent_loss = pi.entropy().sum(-1) * self.alpha  # Normal distribution needs to use sum here
                entropy_all.append(pi.entropy().sum(-1).mean().item())
            else:
                ent_loss = pi.entropy() * self.alpha  # make the agent to explore
                entropy_all.append(pi.entropy().mean().item())
            loss_pi = -(torch.min(ratio * advantage, clipped_advantage) + ent_loss).mean()
            loss_pi_all.append(loss_pi)

            approx_kl = (logp_old - logp).mean().item()
            approx_kl_all.append(approx_kl)
        return loss_pi_all, approx_kl_all, entropy_all

    def compute_loss_v(self, data_all_agent):
        centralized_obs = []
        for i in range(self.num_agent):
            centralized_obs.append(data_all_agent[i]['obs'])
        centralized_obs = torch.concat(centralized_obs, dim=1)
        v_loss = torch.tensor(0, dtype=torch.float32).to(centralized_obs.device)
        for i in range(self.num_agent):
            ret = data_all_agent[i]['ret']
            v_estimate = self.ma_ppo_actor_critic.v[i](centralized_obs)
            v_loss += ((v_estimate - ret) ** 2).mean()  # Value function loss for each agent
        return v_loss

    def learn(self, data_for_all):
        # Train policies
        pi_loss_numpy = [0] * self.num_agent
        kls = [0] * self.num_agent
        entropies = [0] * self.num_agent
        for j in range(self.train_actor_epoch):
            self.pi_optimizer.zero_grad()
            loss_pis, kls, entropies = self.compute_loss_pi(data_for_all)
            if any(kl > 1.5 * self.kl_target for kl in kls):
                break
            for i in range(self.num_agent):
                pi_loss_numpy[i] = loss_pis[i].item()
                loss_pis[i].backward()
            torch.nn.utils.clip_grad_value_(self.pi_params, 1e5)
            self.pi_optimizer.step()

        vi_loss_numpy = 0
        # Train critics
        for i in range(self.train_critic_epoch):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data_for_all)
            vi_loss_numpy = loss_v.item()
            loss_v.backward()
            torch.nn.utils.clip_grad_value_(self.vf_params, 1e5)
            self.vf_optimizer.step()
        return pi_loss_numpy, vi_loss_numpy, kls, entropies

    def save_agent(self):
        if self.save_model:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            param_dict = {}
            for i in range(self.num_agent):
                key = 'agent' + str(i) + '_state_dict'
                param_dict[key] = self.ma_ppo_actor_critic.pi[i].state_dict()
            for i in range(self.num_agent):
                key = 'agent' + str(i) + 'critic_state_dict'
                param_dict[key] = self.ma_ppo_actor_critic.v[i].state_dict()
            param_dict['actor_optimizer_state_dict'] = self.pi_optimizer.state_dict()
            param_dict['critic_optimizer_state_dict'] = self.vf_optimizer.state_dict()
            torch.save(param_dict, os.path.join(self.save_dir, 'model.pth'))
            print(f"Agent model is saved to {os.path.join(self.save_dir, 'model.pth')}")

    def load_model(self):
        model_path = os.path.join(self.save_dir, 'model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            try:
                for i in range(self.num_agent):
                    key = 'agent' + str(i) + '_state_dict'
                    self.ma_ppo_actor_critic.pi[i].load_state_dict(checkpoint[key])
                for i in range(self.num_agent):
                    key = 'agent' + str(i) + 'critic_state_dict'
                    self.ma_ppo_actor_critic.v[i].load_state_dict(checkpoint[key])
                self.pi_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.vf_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                print(f"Agent model is loaded from {os.path.join(self.save_dir, 'model.pth')}")
            except:
                print("The model in the saved path is not matching. The agents are started from scratch.")
        else:
            print("The agents are started from scratch.")

    def train(self, batch_idx):
        futures = []
        for env_runner_rref in self.env_runner_rrefs:
            futures.append(env_runner_rref.rpc_async().get_episode(self.agent_rref))

        # wait all env_runners to finish collecting its own episode for the synchronization
        rets = torch.futures.wait_all(futures)

        # combine all env data
        trajectories_all_env = [ret[0] for ret in rets]
        data_for_all = []
        for i in range(self.num_agent):
            cur_agent_data = dict(
                obs=torch.concat([data[i]["obs"] for data in trajectories_all_env], dim=0).to(self.device),
                act=torch.concat([data[i]["act"] for data in trajectories_all_env], dim=0).to(self.device),
                ret=torch.concat([data[i]["ret"] for data in trajectories_all_env], dim=0).to(self.device),
                adv=torch.concat([data[i]["adv"] for data in trajectories_all_env], dim=0).to(self.device),
                logp=torch.concat([data[i]["logp"] for data in trajectories_all_env], dim=0).to(self.device))
            data_for_all.append(cur_agent_data)

        avg_loss_pi, avg_loss_v, avg_kl_info, entropies = self.learn(data_for_all)

        # training record for avg values of all env_runners
        avg_episode_reward = sum([ret[1] for ret in rets]) / len(rets)
        avg_ep_len = sum([ret[2] for ret in rets]) / len(rets)
        self.add_summary(batch_idx, avg_episode_reward, "train",
                         avg_loss_pi, avg_loss_v, avg_kl_info, avg_ep_len, entropies)
        return avg_episode_reward

    def add_summary(self, global_step, avg_episode_reward, tag="train",
                    avg_loss_pi=None, avg_loss_v=None, avg_kl_info=None, avg_ep_len=None, entropies=None):
        self.writer.add_scalar(f"{tag}/'avg_episode_reward'", avg_episode_reward, global_step)
        if avg_ep_len is not None:
            self.writer.add_scalar(f"{tag}/'avg_ep_len'", avg_ep_len, global_step)
        if avg_loss_v is not None:
            self.writer.add_scalar("avg_loss_v", avg_loss_v, global_step)
        if avg_loss_pi is not None:
            for i in range(self.num_agent):
                self.writer.add_scalar(f"agent_{i}/'avg_loss_pi'", avg_loss_pi[i], global_step)
        if avg_kl_info is not None:
            for i in range(self.num_agent):
                self.writer.add_scalar(f"agent_{i}/'avg_kl_info'", avg_kl_info[i], global_step)
        if entropies is not None:
            for i in range(self.num_agent):
                self.writer.add_scalar(f"agent_{i}/'avg_entropy_info'", entropies[i], global_step)

    def evaluate(self, max_episodes):
        epi_rewards = []
        frames = []
        if self.num_agent > 1:
            self.env_testing.render_mode = "rgb_array"  # in vmas, render_mode can be set after the env's registration
        for epi in tqdm(range(max_episodes),
                        desc="Testing the agents for layout {}".format(self.env_name), file=sys.stdout):
            cur_epi_reward = 0
            state = EnvRunner.process_reset(self.num_agent, self.env_testing)
            ep_len = 0
            for t in range(self.testing_steps):
                # Obtain observations for each agent
                obs = EnvRunner.process_obs(self.num_agent, state)
                acts = self.act_greedy(torch.as_tensor(obs, dtype=torch.float32).to(self.device))
                actions = EnvRunner.process_act(self.num_agent, acts, self.action_space_each)
                next_state, r, done = EnvRunner.process_act_step(self.env_testing, self.num_agent, actions)

                cur_epi_reward += np.sum(r)
                ep_len += 1
                state = next_state
                timeout = (ep_len == self.testing_steps)
                terminal = done or timeout
                if epi == 0:
                    # record evaluation results
                    frame = self.env_testing.render()
                    frames.append(frame)
                if terminal:
                    break
            epi_rewards.append(cur_epi_reward)
            self.add_summary(epi, cur_epi_reward, "test", avg_ep_len=ep_len)

        print('\rAverage rewards in this testing set: {}'.format(np.mean(np.array(epi_rewards))))
        if len(frames) > 0:
            from moviepy import ImageSequenceClip
            fps = 30
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_gif(os.path.join(self.saved_logs_path, "testing.gif"), fps=fps)
        return epi_rewards
