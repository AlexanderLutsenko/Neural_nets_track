import argparse
import os

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.distributions import Categorical

from reinforcement.agent.REINFORCEAgent import REINFORCEAgent


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=lambda x: bool(int(x)),
                        # default=True,
                        default=False
                        )
    parser.add_argument('--visualize', type=lambda x: bool(int(x)),
                        default=True
                        # default=False
                        )
    parser.add_argument('--load_from_checkpoint', type=lambda x: bool(int(x)),
                        default=True
                        # default=False
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
                        default=100000
                        )

    parser.add_argument('--batch_size', type=int,
                        default=256*100
                        # default=5
                        )
    parser.add_argument('--num_trajectories_in_batch', type=int,
                        default=1
                        )
    parser.add_argument('--learning_rate', type=float,
                        default=0.01
                        )
    # Total reward discount
    parser.add_argument('--gamma', type=float,
                        # default=0.99
                        default=1
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
        num_hidden = 32
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.Sigmoid(),
            nn.Linear(num_hidden, num_actions),
        )

    def forward(self, states):
        logits = self.net(states)
        action_distribution = Categorical(logits=logits)
        return action_distribution


def create_agent(args, state_size, num_actions):

    policy_net = PolicyNet(state_size, num_actions, environment).to(device)

    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)

    agent = REINFORCEAgent(policy_net,
                           optimizer,
                           device, num_actions,
                           gamma=args.gamma,
                           batch_size=args.batch_size,
                           num_trajectories_in_batch=args.num_trajectories_in_batch
                           )
    return agent


if __name__ == '__main__':
    args = get_args()

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    env_name = 'CartPole-v1'
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

            actions = agent.act(last_state)

            next_state, reward, is_terminal, info = environment.step(actions)
            reward = reward / 500

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
