import argparse
import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torch.distributions import Normal

from reinforcement.agent.A2CAgent import A2CAgent
from reinforcement.util.parallel import EnvironmentsParallelBeam


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=lambda x: bool(int(x)),
                        default=True,
                        # default=False
                        )
    parser.add_argument('--visualize', type=lambda x: bool(int(x)),
                        # default=True
                        default=False
                        )
    parser.add_argument('--load_from_checkpoint', type=lambda x: bool(int(x)),
                        # default=True
                        default=False
                        )
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoints'
                        )
    parser.add_argument('--checkpoint_frequency', type=int,
                        default=100000
                        )

    # Training parameters

    # Number of games to play
    parser.add_argument('--num_episodes', type=int,
                        default=100000000
                        )

    # Max dataset size
    # parser.add_argument('--replay_memory_size', type=int,
    #                     default=100000
    #                     )

    parser.add_argument('--batch_size', type=int,
                        # default=256
                        # default=1024
                        default=128
                        )
    parser.add_argument('--num_trajectories_in_batch', type=int,
                        # default=1
                        default=32
                        )
    parser.add_argument('--policy_learning_rate', type=float,
                        default=1e-4
                        )
    parser.add_argument('--value_learning_rate', type=float,
                        default=1e-4
                        )
    parser.add_argument('--value_num_updates_per_steps', type=int,
                        default=1
                        )

    # Total reward discount
    parser.add_argument('--gamma', type=float,
                        default=0.99
                        # default=1
                        )

    # random seed
    parser.add_argument('--seed', type=int,
                        default=42
                        )

    args = parser.parse_args()
    return args


class PolicyNet(nn.Module):
    def __init__(self, state_size, num_actions, environment):
        super().__init__()
        num_hidden = 64
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, num_actions*2)
        )
        self.min_val = float(environment.action_space.low)
        self.max_val = float(environment.action_space.high)

    def forward(self, states):
        action_params_flat = self.net(states)
        action_params = action_params_flat.view(-1, num_actions, 2)
        action_means = action_params[:, :, 0]
        action_vars = action_params[:, :, 1]

        # action_means = self.min_val + torch.sigmoid(action_means) * (self.max_val - self.min_val)
        action_means = torch.clip(action_means, self.min_val, self.max_val)
        action_vars = F.softplus(action_vars)

        action_distribution = Normal(action_means, action_vars)
        return action_distribution


class ValueNet(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        num_hidden = 64
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, 1),
        )

    def forward(self, states):
        return self.net(states)


def create_agent(args, state_size, num_actions):

    policy_net = PolicyNet(state_size, num_actions, environment).to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.policy_learning_rate)

    value_net = ValueNet(state_size).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_learning_rate)

    agent = A2CAgent(policy_net, value_net,
                     policy_optimizer, value_optimizer,
                     device, num_actions,
                     gamma=args.gamma,
                     batch_size=args.batch_size,
                     num_trajectories_in_batch=args.num_trajectories_in_batch,
                     value_num_updates_per_steps=args.value_num_updates_per_steps,
                     entropy_bonus_coeff=0
                     )
    return agent


if __name__ == '__main__':
    args = get_args()

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    env_name = 'Pendulum-v1'
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
            # actions = action.clip(policy_net.min_val, policy_net.max_val)


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
