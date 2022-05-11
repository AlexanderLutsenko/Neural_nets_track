import random
from collections import namedtuple
from copy import deepcopy

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

    def size(self):
        return len(self.memory)


class A2CAgent(object):

    def __init__(self,
                 policy_net, value_net,
                 policy_optimizer, value_optimizer,
                 device,
                 num_actions,
                 batch_size,
                 num_trajectories_in_batch,
                 gamma,
                 value_num_updates_per_steps,
                 entropy_bonus_coeff
                 ):

        self.value_num_updates_per_steps = value_num_updates_per_steps
        self.device = device
        self.num_actions = num_actions

        self.batch_size = batch_size
        self.num_trajectories_in_batch = num_trajectories_in_batch
        self.gamma = gamma

        self.policy_net = policy_net
        self.value_net = value_net

        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.replay_memory_size = self.batch_size

        self.replay_memory = ReplayMemory(self.replay_memory_size)
        self.is_training = True

        self.step = 0
        self.update_tau = 0.01
        self.entropy_bonus_coeff = entropy_bonus_coeff

    def set_training(self, is_training):
        self.is_training = is_training

    def save_weights(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict()
        }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])

    def update_target_net(self, target, source, update_tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - update_tau) + param.data * update_tau
            )

    def act(self, state):
        self.policy_net.eval()
        with torch.no_grad():
            th_state = torch.from_numpy(state).unsqueeze(0).type(torch.get_default_dtype()).to(self.device)
            action_distribution = self.policy_net(th_state)
            action = action_distribution.sample().squeeze(0).cpu().numpy()
            return action

    def observe_and_learn(self, state, action, reward, next_state, is_terminal):
        if not self.is_training:
            return

        self.replay_memory.push(
            torch.tensor(state),
            torch.tensor(action, dtype=torch.get_default_dtype()),
            torch.tensor(reward, dtype=torch.get_default_dtype()),
            torch.tensor(next_state),
            torch.tensor(0 if is_terminal else 1, dtype=torch.get_default_dtype())
        )

        if is_terminal and self.replay_memory.size() >= self.batch_size:
            self.learn()
            self.replay_memory = ReplayMemory(self.replay_memory_size)

    def learn(self):
        self.policy_net.train()
        self.value_net.train()

        self.step += 1

        samples = self.replay_memory.sample(self.batch_size)
        batch = Sample(*zip(*samples))

        states = torch.stack(batch.state).type(torch.get_default_dtype()).to(self.device)
        actions = torch.stack(batch.action).to(self.device)
        rewards = torch.stack(batch.reward).to(self.device)
        next_states = torch.stack(batch.next_state).type(torch.get_default_dtype()).to(self.device)
        non_terminal_masks = torch.stack(batch.non_terminal).to(self.device)

        action_distributions = self.policy_net(states)

        with torch.no_grad():
            total_rewards = rewards + self.gamma * self.value_net(next_states).squeeze(1) * non_terminal_masks
            baselines = self.value_net(states).squeeze(1)
            advantages = total_rewards - baselines

        policy_loss = -(action_distributions.log_prob(actions) * advantages).mean() - self.entropy_bonus_coeff * action_distributions.entropy().mean()

        with torch.no_grad():
            total_rewards = rewards + self.gamma * self.value_net(next_states).squeeze(1) * non_terminal_masks

        baselines = self.value_net(states).squeeze(1)
        advantages = total_rewards - baselines
        value_loss = advantages**2

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
