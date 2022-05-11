import argparse
import os

import gym
import pybulletgym
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torch.distributions import Normal

from reinforcement.agent.SOPAgent import SOPAgent


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
    parser.add_argument('--checkpoint_name', type=str,
                        # default=None
                        default='HumanoidPyBulletEnv-v0'
                        )
    parser.add_argument('--checkpoint_frequency', type=int,
                        default=100
                        )

    # Training parameters

    # Number of games to play
    parser.add_argument('--num_episodes', type=int,
                        default=1000000
                        )
    # Max dataset size
    parser.add_argument('--replay_memory_size', type=int,
                        default=500000
                        )
    # Agent won't start training until memory is big enough
    parser.add_argument('--replay_memory_warmup_size', type=int,
                        default=10000
                        )

    parser.add_argument('--batch_size', type=int,
                        # default=64
                        default=256
                        # default=4096
                        )
    parser.add_argument('--actor_learning_rate', type=float,
                        # default=0.00005
                        default=1e-6
                        )
    parser.add_argument('--critic_learning_rate', type=float,
                        # default=0.00005
                        default=1e-5
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

    parser.add_argument('--action_eps', type=float,
                        default=0.29
                        # default=0.5
                        # default=0.15
                        )

    # Number of training steps per observation step
    parser.add_argument('--repeat_update', type=int,
                        default=1
                        )
    parser.add_argument('--target_update_tau', type=float,
                        default=0.005
                        )

    # random seed
    parser.add_argument('--seed', type=int,
                        default=42
                        # default=43
                        )

    args = parser.parse_args()
    return args


class Actor(nn.Module):
    def __init__(self, state_size, num_actions, environment):
        super().__init__()
        num_hidden = 256
        self.net = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_actions),
        )
        self.min_val = float(environment.action_space.low[0])
        self.max_val = float(environment.action_space.high[1])

    def forward(self, states, with_noise):
        actions_raw = self.net(states)

        if with_noise:
            zeros = torch.zeros(actions_raw.size()).to(actions_raw.device)
            std = torch.zeros(actions_raw.size()).to(actions_raw.device) + args.action_eps
            noise = Normal(zeros, std).sample()
            actions_raw = actions_raw + noise

        # return self.min_val + torch.sigmoid(actions_raw) * (self.max_val - self.min_val)
        return torch.tanh(actions_raw)


class Critic(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        num_hidden = 256
        self.net = nn.Sequential(
            nn.Linear(state_size + num_actions, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=1))


def create_agent(args, state_size, num_actions):

    def actor_builder():
        return Actor(state_size, num_actions, environment)

    def critic_builder():
        return Critic(state_size, num_actions)

    actor_optimizer_builder = lambda parameters: optim.Adam(parameters, lr=args.actor_learning_rate)
    critic_optimizer_builder = lambda parameters: optim.Adam(parameters, lr=args.critic_learning_rate)

    agent = SOPAgent(actor_builder, critic_builder, actor_optimizer_builder, critic_optimizer_builder,
                     device, num_actions,
                     gamma=args.gamma,

                     replay_memory_size=args.replay_memory_size,
                     replay_memory_warmup_size=args.replay_memory_warmup_size,
                     batch_size=args.batch_size,

                     eps_start=args.eps_start,
                     eps_end=args.eps_end,
                     eps_decay=args.eps_decay,

                     repeat_update=args.repeat_update,
                     target_update_tau=args.target_update_tau
                     )
    return agent


if __name__ == '__main__':
    args = get_args()

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.train else 'cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    env_name = 'HumanoidPyBulletEnv-v0'
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
        agent.load_weights(os.path.join(args.checkpoint_dir, args.checkpoint_name if args.checkpoint_name is not None else env_name))

    agent.set_training(args.train)

    total_steps = 0
    for episode in range(1, args.num_episodes + 1):
        if args.visualize:
            environment.render()

        last_state = environment.reset()
        total_reward = 0
        step = 0
        while True:
            actions = agent.act(last_state, with_noise=args.train)

            next_state, reward, is_terminal, info = environment.step(actions)

            reward = reward / 10000
            if step == (environment.spec.max_episode_steps - 1):
                is_terminal = False

            agent.observe_and_learn(last_state, actions, reward, next_state, is_terminal)

            last_state = next_state
            step += 1
            total_steps += 1
            total_reward += reward

            if is_terminal:
                print('Episode {}, steps: {}, reward: {} '.format(episode, total_steps, total_reward))

                if args.train and episode % args.checkpoint_frequency == 0:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    agent.save_weights(os.path.join(args.checkpoint_dir, env_name))
                    print('Checkpoint created')
                break
