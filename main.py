import os
import argparse
from utils.run import main_run


def get_parser():
    parser = argparse.ArgumentParser(prog="Training PPO agent")
    parser.add_argument("--world_size", type=int, default=4,
                        help="The world_size of processes")
    parser.add_argument("--device", type=str, default="gpu",
                        help="The device of agents' networks")
    parser.add_argument("--layout_index", type=int, default=0,
                        help="The layout index for the RL environments")
    parser.add_argument("--only_evaluate", type=lambda x: (str(x).lower() == 'true'),
                        default=False, help='select if using the previously well-trained agents for evaluations only')
    parser.add_argument("--testing_episodes", type=int, default=100,
                        help="number of episodes for evaluation process")

    return parser


if __name__ == "__main__":
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    layouts = ["CartPole-v1",
               "LunarLander-v3",
               "navigation",
               "road_traffic"]
    num_agent_in_env = [1, 1, 3, 5] # index is the same as layouts
    training_episodes_all = [30, 500, 1500, 3000]
    training_num_steps_per_episode_all = [1000, 1000, 500, 100]
    layout_index = args.layout_index  # you can either change the index from the default arg or here
    assert layout_index < len(layouts)
    layout = layouts[layout_index]
    env_name = layout
    only_evaluate = args.only_evaluate
    training_episodes = training_episodes_all[layout_index]
    testing_episodes = args.testing_episodes
    num_steps_per_episode = training_num_steps_per_episode_all[layout_index]
    world_size = args.world_size
    device = args.device
    root_dir = os.path.join(os.getcwd())
    num_agent_in_env = num_agent_in_env[layout_index]
    main_run(world_size, num_agent_in_env, env_name,
             num_steps_per_episode, device,
             only_evaluate, training_episodes, testing_episodes)
