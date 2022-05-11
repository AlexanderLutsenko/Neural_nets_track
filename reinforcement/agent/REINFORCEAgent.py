import random
from collections import namedtuple

import torch

Sample = namedtuple('Sample',
                    ('state', 'action', 'reward'))


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def calc_rewards_to_go(self, rewards, gamma):
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = cumulative_reward * gamma + r
            discounted_rewards.append(cumulative_reward)

        return reversed(discounted_rewards)

    def get_samples(self, gamma):
        disounted_rewards = self.calc_rewards_to_go(self.rewards, gamma)
        return [Sample(state=state, action=action, reward=reward) for state, action, reward in zip(self.states, self.actions, disounted_rewards)]


class TrajectoryMemory:

    def __init__(self):
        self.trajectories = [Trajectory()]

    def push(self, state, action, reward, is_terminal):
        self.trajectories[-1].add(state, action, reward)
        if is_terminal:
            self.trajectories.append(Trajectory())

    def sample(self, gamma, batch_size):
        samples = []
        for trajectory in self.trajectories:
            samples += trajectory.get_samples(gamma)
        return random.sample(samples, min(len(samples), batch_size))

    def get_num_trajectories(self):
        return len(self.trajectories)


class REINFORCEAgent(object):

    def __init__(self,
                 policy_net,
                 optimizer,
                 device,
                 num_actions,
                 batch_size,
                 num_trajectories_in_batch,
                 gamma,
                 ):

        self.device = device
        self.num_actions = num_actions

        self.batch_size = batch_size
        self.num_trajectories_in_batch = num_trajectories_in_batch
        self.gamma = gamma

        self.policy_net = policy_net
        self.optimizer = optimizer

        self.memory = TrajectoryMemory()
        self.is_training = True

    def set_training(self, is_training):
        self.is_training = is_training

    def save_weights(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict)

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

        self.memory.push(
            torch.tensor(state),
            torch.tensor(action, dtype=torch.get_default_dtype()),
            torch.tensor(reward, dtype=torch.get_default_dtype()),
            is_terminal
        )

        if is_terminal and self.memory.get_num_trajectories() > self.num_trajectories_in_batch:
            self.learn()
            self.memory = TrajectoryMemory()

    def learn(self):
        self.policy_net.train()

        samples = self.memory.sample(self.gamma, self.batch_size)
        batch = Sample(*zip(*samples))

        state_batch = torch.stack(batch.state).type(torch.get_default_dtype()).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward2go_batch = torch.stack(batch.reward).to(self.device)

        action_distribution_batch = self.policy_net(state_batch)
        baseline = reward2go_batch.mean()
        loss = -(action_distribution_batch.log_prob(action_batch) * (reward2go_batch - baseline)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
