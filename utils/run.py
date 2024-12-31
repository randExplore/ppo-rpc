import os
from tqdm import tqdm
import sys
import time

import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from model.ppo.mappo import PPOTrainer, ENV_RUNNER_NAME


def run_worker(rank, world_size, total_num_episodes, num_agent, env_name,
               num_steps_per_episode=1000, device="gpu", testing_episodes=100):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '30000'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc("PPOAgent", rank=rank, world_size=world_size)

        agents = PPOTrainer(world_size, num_agent, env_name, num_steps_per_episode, device)
        for epi in tqdm(range(total_num_episodes),
                        desc="Training the ppo_agent for env {}".format(env_name), file=sys.stdout):
            avg_episode_reward = agents.train(epi)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(epi + 1, avg_episode_reward))
            if (epi + 1) % 200 == 0 or (epi + 1) == total_num_episodes:
                agents.save_agent()
        agents.evaluate(testing_episodes)
    else:
        # other ranks are the environment runners passively waiting for start from the agent
        rpc.init_rpc(ENV_RUNNER_NAME.format(rank), rank=rank, world_size=world_size)
    rpc.shutdown()


def evaluate_agents(agents: PPOTrainer, testing_episodes: int):
    agents.evaluate(testing_episodes)


def main_run(world_size, num_agent_in_env, env_name,
             num_steps_per_episode, device,
             only_evaluate, training_episodes, testing_episodes):

    start_time = time.time()
    if only_evaluate:
        agents = PPOTrainer(world_size, num_agent_in_env, env_name, num_steps_per_episode, device,
                                      only_evaluate=True)
        evaluate_agents(agents, testing_episodes)
    else:
        assert world_size >= 2, "It requires at least one trainer and one env_runner!"
        mp.spawn(
            run_worker,
            args=(world_size, training_episodes, num_agent_in_env, env_name,
                  num_steps_per_episode, device, testing_episodes),
            nprocs=world_size,
            join=True
        )
    end_time = time.time()
    print("It takes {} seconds to finish running!".format(end_time - start_time))
