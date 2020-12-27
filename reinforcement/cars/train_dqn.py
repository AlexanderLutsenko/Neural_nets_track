import argparse
import os

import numpy as np
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
                        default=1
                        )

    # Training parameters

    # Number of games to play
    parser.add_argument('--num_episodes', type=int,
                        default=100000
                        )
    # Max dataset size
    parser.add_argument('--replay_memory_size', type=int,
                        default=20000
                        )
    # Agent won't start training until memory is big enough
    parser.add_argument('--replay_memory_warmup_size', type=int,
                        # default=10000
                        default=0
                        )

    parser.add_argument('--batch_size', type=int,
                        default=128
                        )
    parser.add_argument('--optimizer', type=str,
                        default='adam',
                        # default='sgd',
                        )
    parser.add_argument('--learning_rate', type=float,
                        default=0.001
                        )
    parser.add_argument('--momentum', type=float,
                        default=0.9
                        )

    # Total reward discount
    parser.add_argument('--gamma', type=float,
                        default=0.99
                        )

    # epsilon-greedy parameters:
    # chance to perform random action decreases from eps_start to eps_end over the course of eps_decay steps
    parser.add_argument('--eps_start', type=float,
                        default=0.2
                        # default=0.05
                        )
    parser.add_argument('--eps_end', type=float,
                        default=0.05
                        # default=0.01
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


def create_agent(args, input_channels, num_actions):

    def network_builder():

        class Normalize(nn.Module):
            def forward(self, input):
                return input / 255

        class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)

        net = nn.Sequential(
            Normalize(),
            # 96 x 96
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 48 x 48
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 24x24
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12x12
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 6x6
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3x3
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 1x1
            Flatten(),
            # nn.Dropout(p=0.2),
            nn.Linear(256, num_actions)
        )

        return net

    if args.optimizer == 'sgd':
        optimizer_builder = lambda parameters: optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer_builder = lambda parameters: optim.Adam(parameters, lr=args.learning_rate)
    else:
        raise Exception('Unsupported optimizer')

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


class DiscreteToContinuousActionMapper:
    # Since DQN can only make discrete actions,
    # so there are 5 of them which we then transform into 3 original continuous actions
    num_discrete_actions = 5

    def make_continuous(self, action_idx):
        assert 0 <= action_idx < self.num_discrete_actions

        # discrete action space: one_hot of [steer left, steer right, gas, brakes, do nothing]
        # continuous actions space: steer [-1, 1], gas [0, 1], brakes [0, 1]
        actions = np.zeros((3,), dtype=np.float32)
        # Make car always running
        actions[1] = 0.1
        # Steer left
        if action_idx == 1:
            actions[0] = -1
        # Steer right
        elif action_idx == 2:
            actions[0] = 1
        # Gas
        elif action_idx == 3:
            actions[1] = 1
        # Brakes
        elif action_idx == 4:
            actions[2] = 1
        # else action_idx == 0: do nothing
        return actions


class StateKeeper:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.states = []
        # Each input frame is reduced to one channel, and there are sequence_length frames in each state
        self.num_channels = 1 * self.sequence_length

    def append_state(self, state, reset=False):
        # PyTorch image channel order is different from that of the environment
        # (h, w, c) -> (c, h, w)
        state = state.copy().transpose((2, 0, 1))
        # Take green channel
        state = state[1:2]

        if reset:
            self.states = [state] * self.sequence_length
        else:
            self.states.append(state)
            self.states = self.states[1:]

        return np.concatenate(self.states, axis=0)


class OffRoadAnalyzer:
    def __init__(self):
        self.off_road_time = 0

    def calc_off_road_time(self, frame):
        # Car is off-road when there's a lot of green stuff around it
        car_area = frame[65:79, 44:52].astype(np.float32)
        green_score = np.mean(2 * car_area[:, :, 1] > car_area[:, :, 0] + car_area[:, :, 2])
        if green_score > 0.5:
            self.off_road_time += 1
        else:
            self.off_road_time = 0
        return self.off_road_time


if __name__ == '__main__':
    args = get_args()

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

    env_name = 'CarRacing-v0'
    environment = gym.make(env_name)

    # Set fixed random seeds so your experiments are reproducible
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    environment.seed(args.seed)

    print('Observation space: {}'.format(environment.observation_space))
    print('Action space: {}'.format(environment.action_space))

    # Stack 4 consecutive frames into single state
    state_keeper = StateKeeper(sequence_length=4)
    # Since DQN can only make discrete actions,
    # so there are 5 of them which we then transform into 3 original continuous actions
    action_mapper = DiscreteToContinuousActionMapper()
    # We want to modify original environment's reward when car goes off the track
    off_road_analyzer = OffRoadAnalyzer()

    agent = create_agent(args, state_keeper.num_channels, action_mapper.num_discrete_actions)

    if args.load_from_checkpoint:
        agent.load_weights(os.path.join(args.checkpoint_dir, env_name))

    agent.set_training(args.train)

    total_steps = 0
    for episode in range(args.num_episodes):

        # FIXME: CarRacing-v0 leaks memory,
        #  so instead of simply calling environment.reset(), we create new environment at the start of each episode
        environment.close()
        environment = gym.make(env_name)
        environment.seed(args.seed + episode)

        last_state = environment.reset()
        last_state = state_keeper.append_state(last_state, reset=True)

        # Skip initial zoom-in animation (first ~40 frames)
        zero_actions = np.zeros((3,), dtype=np.float32)
        for _ in range(40):
            environment.step(zero_actions)

        total_reward = 0
        while True:
            if args.visualize:
                environment.render()

            action_idx = agent.act(last_state)
            continuous_actions = action_mapper.make_continuous(action_idx)
            next_state_rgb, reward, is_terminal, info = environment.step(continuous_actions)

            off_road_time = off_road_analyzer.calc_off_road_time(next_state_rgb)
            # Penalize agent for going off-road
            if off_road_time > 0:
                reward -= 0.1
            # Been off-road for too long; finish episode
            if args.train and off_road_time > 100:
                is_terminal = True

            next_state_stacked = state_keeper.append_state(next_state_rgb, reset=False)
            agent.observe_and_learn(last_state, action_idx, reward, next_state_stacked, is_terminal)

            total_steps += 1
            total_reward += reward
            last_state = next_state_stacked

            if is_terminal:
                print('Episode {}, steps: {}, reward: {} '.format(episode, total_steps, total_reward))

                if args.train and episode % args.checkpoint_frequency == 0:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    agent.save_weights(os.path.join(args.checkpoint_dir, env_name))
                    print('Checkpoint created')
                break
