import argparse
import os

import gym
import torch
import torch.nn as nn
import torch.optim as optim

import random

from reinforcement.DQNAgent import DQNAgent


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=lambda x: bool(int(x)),
                        default=True,
                        # default=False
                        )
    parser.add_argument('--visualize', type=lambda x: bool(int(x)),
                        default=True
                        # default=False
                        )
    parser.add_argument('--load_from_checkpoint', type=lambda x: bool(int(x)),
                        # default=True
                        default=False
                        )
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints'
                        )
    parser.add_argument('--checkpoint_frequency', type=int,
                        default=10
                        )

    # Training parameters

    # Number of games to play
    parser.add_argument('--num_episodes', type=int,
                        default=1000
                        )
    # Max dataset size
    parser.add_argument('--replay_memory_size', type=int,
                        default=100000
                        )
    # Agent won't start training until memory is big enough
    parser.add_argument('--replay_memory_warmup_size', type=int,
                        default=1
                        )

    parser.add_argument('--batch_size', type=int,
                        default=512
                        )
    parser.add_argument('--learning_rate', type=float,
                        default=0.001
                        )
    parser.add_argument('--momentum', type=float,
                        default=0.95
                        )

    # Total reward discount
    parser.add_argument('--gamma', type=float,
                        default=0.99
                        )

    # epsilon-greedy parameters:
    # chance to perform random action decreases from eps_start to eps_end over the course of eps_decay steps
    parser.add_argument('--eps_start', type=float,
                        default=0.9
                        )
    parser.add_argument('--eps_end', type=float,
                        default=0.05
                        )
    parser.add_argument('--eps_decay', type=int,
                        default=10000
                        )

    # target_net update frequency
    parser.add_argument('--target_update_frequency', type=int,
                        default=100
                        )
    # Number of training steps per observation step
    parser.add_argument('--repeat_update', type=int,
                        default=1
                        )

    # random seed
    parser.add_argument('--seed', type=int,
                        default=42
                        )

    args = parser.parse_args()
    return args


def create_agent(args, state_size, num_actions):

    def network_builder():

        net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, num_actions)
        )

        return net

    optimizer_builder = lambda parameters: optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum)

    agent = DQNAgent(network_builder, optimizer_builder,
                     device, num_actions,
                     gamma=args.gamma,

                     replay_memory_size=args.replay_memory_size,
                     replay_memory_warmup_size=args.replay_memory_warmup_size,
                     batch_size=args.batch_size,

                     eps_start=args.eps_start,
                     eps_end=args.eps_end,
                     eps_decay=args.eps_decay,

                     target_update_frequency=args.target_update_frequency,

                     repeat_update=args.repeat_update
                     )
    return agent


if __name__ == '__main__':
    args = get_args()

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    env_name = 'LunarLander-v2'
    environment = gym.make(env_name)

    # Set fixed random seeds so your experiments are reproducible
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    environment.seed(args.seed)

    print('Observation space: {}'.format(environment.observation_space))
    print('Action space: {}'.format(environment.action_space))

    state_size = environment.observation_space.shape[0]
    num_actions = environment.action_space.n
    agent = create_agent(args, state_size, num_actions)

    if args.load_from_checkpoint:
        agent.load_weights(os.path.join(args.checkpoint_dir, env_name))

    agent.set_training(args.train)

    total_steps = 0
    for episode in range(args.num_episodes):
        last_state = environment.reset()
        total_reward = 0
        while True:
            if args.visualize:
                environment.render()

            action = agent.act(last_state)
            next_state, reward, is_terminal, info = environment.step(action)
            agent.observe_and_learn(last_state, action, reward, next_state, is_terminal)

            total_steps += 1
            total_reward += reward
            last_state = next_state

            if is_terminal:
                print('Episode {}, steps: {}, reward: {} '.format(episode, total_steps, total_reward))

                if args.train and episode % args.checkpoint_frequency == 0:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    agent.save_weights(os.path.join(args.checkpoint_dir, env_name))
                    print('Checkpoint created')
                break
