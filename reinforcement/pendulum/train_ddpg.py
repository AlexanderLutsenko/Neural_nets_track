import argparse
import os

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from reinforcement.DDPGAgent import DDPGAgent


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
                        default=256*3
                        )

    parser.add_argument('--batch_size', type=int,
                        default=256
                        )
    parser.add_argument('--actor_learning_rate', type=float,
                        default=0.0001
                        )
    parser.add_argument('--critic_learning_rate', type=float,
                        default=0.001
                        )
    parser.add_argument('--momentum', type=float,
                        default=0.8
                        )

    # Total reward discount
    parser.add_argument('--gamma', type=float,
                        default=0.99
                        )

    # epsilon-greedy parameters:
    # chance to perform random action decreases from eps_start to eps_end over the course of eps_decay steps
    parser.add_argument('--eps_start', type=float,
                        default=1
                        # default=0.9
                        )
    parser.add_argument('--eps_end', type=float,
                        # default=0.3
                        default=0.05
                        )
    parser.add_argument('--eps_decay', type=int,
                        default=50*200
                        )

    # target_net update frequency
    parser.add_argument('--target_update_frequency', type=int,
                        default=200
                        )
    # Number of training steps per observation step
    parser.add_argument('--repeat_update', type=int,
                        default=1
                        )

    # random seed
    parser.add_argument('--seed', type=int,
                        default=42
                        # default=100500
                        )

    args = parser.parse_args()
    return args


class Actor(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        num_hidden = 64
        self.net = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_actions),
            nn.Sigmoid()
        )
        self.min_val = float(environment.action_space.low)
        self.max_val = float(environment.action_space.high)

    def forward(self, states):
        actions_unscaled = self.net(states)
        return self.min_val + actions_unscaled * (self.max_val - self.min_val)


class Critic(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        num_hidden = 64
        self.net = nn.Sequential(
            nn.Linear(state_size + num_actions, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=1))


def create_agent(args, state_size, num_actions):

    def actor_builder():
        return Actor(state_size, num_actions)

    def critic_builder():
        return Critic(state_size, num_actions)

    # optimizer_builder = lambda parameters: optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum)
    actor_optimizer_builder = lambda parameters: optim.Adam(parameters, lr=args.actor_learning_rate)
    critic_optimizer_builder = lambda parameters: optim.Adam(parameters, lr=args.critic_learning_rate)

    agent = DDPGAgent(actor_builder, critic_builder, actor_optimizer_builder, critic_optimizer_builder,
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

    env_name = 'Pendulum-v0'
    environment = gym.make(env_name)

    # Set fixed random seeds so your experiments are reproducible
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    environment.seed(args.seed)

    print('Observation space: {}'.format(environment.observation_space))
    print('Action space: {}'.format(environment.action_space))

    state_size = environment.observation_space.shape[0]
    num_actions = environment.action_space.shape[0]
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

            actions = agent.act(last_state)

            next_state, reward, is_terminal, info = environment.step(actions)
            reward = reward / 1000

            agent.observe_and_learn(last_state, actions, reward, next_state, is_terminal)

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
