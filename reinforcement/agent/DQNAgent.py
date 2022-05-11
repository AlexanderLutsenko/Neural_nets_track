import random

import numpy as np
from collections import namedtuple

import torch
import torch.nn.functional as F


Sample = namedtuple('Sample',
                    ('state', 'action', 'reward', 'next_state', 'non_terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Saves a training sample
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Sample(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


class DQNAgent(object):

    def __init__(self,
                 network_builder,
                 optimizer_builder,
                 device,

                 num_actions,

                 replay_memory_size,
                 replay_memory_warmup_size,
                 batch_size,

                 gamma,
                 eps_start,
                 eps_end,
                 eps_decay,

                 target_update_frequency,
                 repeat_update=1
                 ):

        self.device = device
        self.num_actions = num_actions

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_frequency = target_update_frequency
        self.repeat_update = repeat_update

        self.policy_net = network_builder().to(device)

        self.target_net = network_builder().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Some layers (like Dropout, Batch normalization, etc.) exhibit different behaviour depending on the mode (training or evaluation)
        # target_net (used in loss calculation) is a frozen copy of policy_net and never trains, so set it in evaluation mode right away
        self.target_net.eval()

        self.optimizer = optimizer_builder(self.policy_net.parameters())

        # Here we'll be maintaining our "dataset", i.e. agent's past experience
        self.memory = ReplayMemory(replay_memory_size)
        self.replay_memory_warmup_size = replay_memory_warmup_size

        self.steps_done = 0
        self.eps_threshold = self.eps_start

        self.is_training = True

    def set_training(self, is_training):
        self.is_training = is_training

    def save_weights(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def act(self, state):
        self.policy_net.eval()

        sample = random.random()
        self.eps_threshold = self.eps_start - (self.eps_start - self.eps_end) * min(1, self.steps_done / self.eps_decay)

        self.steps_done += 1

        # epsilon-greedy policy:
        # with probability eps_threshold, act randomly
        if self.is_training and sample < self.eps_threshold:
            return random.randrange(self.num_actions)
        # otherwise, make the "best" action
        else:
            with torch.no_grad():
                th_state = torch.from_numpy(state).unsqueeze(0).type(torch.get_default_dtype()).to(self.device)
                action_scores = self.policy_net(th_state)
                action = action_scores.max(1)[1]
                return np.asscalar(action.cpu().numpy())

    def observe_and_learn(self, state, action, reward, next_state, is_terminal):
        if not self.is_training:
            return

        self.memory.push(
            torch.tensor(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.get_default_dtype()),
            torch.tensor(next_state),
            torch.tensor(0 if is_terminal else 1, dtype=torch.get_default_dtype())
        )

        self.learn()

    def learn(self):
        self.policy_net.train()

        if len(self.memory) < max(self.batch_size, self.replay_memory_warmup_size):
            return

        for _ in range(self.repeat_update):
            samples = self.memory.sample(self.batch_size)
            # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Sample(*zip(*samples))

            # Compute a mask of non-final states and concatenate the batch elements
            state_batch = torch.stack(batch.state).type(torch.get_default_dtype()).to(self.device)
            action_batch = torch.stack(batch.action).to(self.device)
            reward_batch = torch.stack(batch.reward).to(self.device)
            next_state_batch = torch.stack(batch.next_state).type(torch.get_default_dtype()).to(self.device)
            non_terminal_mask = torch.stack(batch.non_terminal).to(self.device)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_q_values = self.policy_net(state_batch)
            state_action_values = state_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # Compute max Q'(s_{t+1})
            # target_net must never be trained, so disable gradient calculation for it
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1)[0]

            # Compute the expected Q values
            expected_state_action_values = reward_batch + (self.gamma * next_state_values * non_terminal_mask)
            loss = F.mse_loss(state_action_values, expected_state_action_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.steps_done % self.target_update_frequency == 0:
            # It's time to update target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
