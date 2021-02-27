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


class DDPGAgent(object):

    def __init__(self,
                 actor_builder, critic_builder,
                 actor_optimizer_builder, critic_optimizer_builder,
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

        self.actor_net = actor_builder().to(device)
        self.critic_net = critic_builder().to(device)

        self.actor_target_net = actor_builder().to(device)
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.actor_target_net.eval()

        self.critic_target_net = critic_builder().to(device)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        self.critic_target_net.eval()

        self.actor_optimizer = actor_optimizer_builder(self.actor_net.parameters())
        self.critic_optimizer = critic_optimizer_builder(self.critic_net.parameters())

        # Here we'll be maintaining our "dataset", i.e. agent's past experience
        self.memory = ReplayMemory(replay_memory_size)
        self.replay_memory_warmup_size = replay_memory_warmup_size

        self.steps_done = 0
        self.eps_threshold = self.eps_start

        self.is_training = True

        self.update_tau = 0.01

    def set_training(self, is_training):
        self.is_training = is_training

    def save_weights(self, path):
        torch.save({
           'actor_state_dict': self.actor_net.state_dict(),
           'critic_state_dict': self.critic_net.state_dict()
        }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_state_dict'])

        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

    def update_target_net(self, target, source, update_tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - update_tau) + param.data * update_tau
            )

    def act(self, state):
        self.actor_net.eval()

        self.steps_done += 1
        with torch.no_grad():
            th_state = torch.from_numpy(state).unsqueeze(0).type(torch.get_default_dtype()).to(self.device)
            action = self.actor_net(th_state).detach().cpu().numpy().squeeze(axis=0)

            sample = random.random()
            self.eps_threshold = self.eps_start - (self.eps_start - self.eps_end) * min(1, self.steps_done / self.eps_decay)
            if self.is_training and sample < self.eps_threshold:
                action = np.random.uniform(-2, 2, size=action.shape)

            # if self.is_training and self.steps_done < self.eps_decay:
            #     p = self.steps_done / self.eps_decay
            #     action = action * p + (1 - p) * next(self.noise) * 2

            # if self.is_training and self.steps_done < self.eps_decay:
            #     p = self.steps_done / self.eps_decay
            #     action = np.random.normal(loc=action, scale=p, size=action.shape)

            return action

    def observe_and_learn(self, state, action, reward, next_state, is_terminal):
        if not self.is_training:
            return

        self.memory.push(
            torch.tensor(state),
            torch.tensor(action, dtype=torch.get_default_dtype()),
            torch.tensor(reward, dtype=torch.get_default_dtype()),
            torch.tensor(next_state),
            torch.tensor(0 if is_terminal else 1, dtype=torch.get_default_dtype())
        )

        self.learn()

    def learn(self):
        self.actor_net.train()
        self.critic_net.train()

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

            # Compute Q(s_t, mu(s_t))
            Q_values = self.critic_net(state_batch, action_batch).squeeze(dim=1)
            # print(action_batch)
            # print(state_batch)

            # Compute Q'(s_{t+1}, mu'(s_{t+1}))
            with torch.no_grad():
                next_Q_values = self.critic_target_net(next_state_batch, self.actor_target_net(next_state_batch)).squeeze(dim=1)
                expected_Q_values = (next_Q_values * non_terminal_mask * self.gamma) + reward_batch

            # print(Q_values.shape, expected_Q_values.shape)

            critic_loss = F.mse_loss(Q_values, expected_Q_values)
            # print(critic_loss.detach().cpu().numpy() / reward_batch.mean().cpu().numpy())

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic_net(state_batch, self.actor_net(state_batch)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # self.update_target_net(self.actor_target_net, self.actor_net, self.update_tau)
        # self.update_target_net(self.critic_target_net, self.critic_net, self.update_tau)

        if self.steps_done % self.target_update_frequency == 0:
            self.actor_target_net.load_state_dict(self.actor_net.state_dict())
            self.critic_target_net.load_state_dict(self.critic_net.state_dict())
