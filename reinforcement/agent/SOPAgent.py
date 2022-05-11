from collections import namedtuple

import torch
import torch.nn.functional as F


Sample = namedtuple('Sample',
                    ('state', 'action', 'reward', 'next_state', 'non_terminal'))


class ReplayMemory:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device

        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.non_terminals = None

        self.step = 0

    def push(self, state, action, reward, next_state, non_terminal):
        if self.states is None:
            self.states = torch.zeros(size=(self.capacity, *state.shape), dtype=state.dtype, device=self.device)
            self.actions = torch.zeros(size=(self.capacity, *action.shape), dtype=action.dtype, device=self.device)
            self.rewards = torch.zeros(size=(self.capacity, *reward.shape), dtype=reward.dtype, device=self.device)
            self.next_states = torch.zeros(size=(self.capacity, *next_state.shape), dtype=next_state.dtype, device=self.device)
            self.non_terminals = torch.zeros(size=(self.capacity, *non_terminal.shape), dtype=non_terminal.dtype, device=self.device)

        i = self.step % self.capacity
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.non_terminals[i] = non_terminal

        self.step += 1

    def sample(self, batch_size):
        batch_indices = torch.randint(low=0, high=self.real_capacity, size=(batch_size,), dtype=torch.int64, device=self.device)
        state_batch = self.states[batch_indices]
        action_batch = self.actions[batch_indices]
        reward_batch = self.rewards[batch_indices]
        next_state_batch = self.next_states[batch_indices]
        non_terminal_batch = self.non_terminals[batch_indices]
        return Sample(state=state_batch, action=action_batch, reward=reward_batch, next_state=next_state_batch, non_terminal=non_terminal_batch)

    @property
    def real_capacity(self):
        return min(self.step, self.capacity)

    def __len__(self):
        return self.real_capacity


class SOPAgent:
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

                 repeat_update=1,
                 target_update_tau=0.01,
                 ):

        self.device = device
        self.num_actions = num_actions

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.repeat_update = repeat_update
        self.target_update_tau = target_update_tau

        self.actor_net = actor_builder().to(device)
        self.actor_optimizer = actor_optimizer_builder(self.actor_net.parameters())
        self.actor_target_net = actor_builder().to(device)
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.actor_target_net.eval()

        self.critic_nets = []
        self.critic_optimizers = []
        self.critic_target_nets = []
        for i in range(2):
            net = critic_builder().to(device)
            net.train()
            optimizer = critic_optimizer_builder(net.parameters())
            target_net = critic_builder().to(device)
            target_net.load_state_dict(net.state_dict())
            target_net.eval()
            self.critic_nets.append(net)
            self.critic_optimizers.append(optimizer)
            self.critic_target_nets.append(target_net)

        self.memory = ReplayMemory(replay_memory_size, self.device)
        self.replay_memory_warmup_size = replay_memory_warmup_size

        self.steps_done = 0
        self.eps_threshold = self.eps_start

        self.is_training = True

    def set_training(self, is_training):
        self.is_training = is_training

    def save_weights(self, path):
        torch.save({
           'actor_state_dict': self.actor_net.state_dict(),
           'critic_state_dict_0': self.critic_nets[0].state_dict(),
           'critic_state_dict_1': self.critic_nets[1].state_dict()
        }, path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.actor_net.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target_net.load_state_dict(checkpoint['actor_state_dict'])

        for i in range(len(self.critic_nets)):
            self.critic_nets[i].load_state_dict(checkpoint['critic_state_dict_{}'.format(i)])
            self.critic_target_nets[i].load_state_dict(checkpoint['critic_state_dict_{}'.format(i)])

    def update_target_net(self, target, source, update_tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - update_tau) + param.data * update_tau
            )

    def act(self, state, with_noise=True):
        self.actor_net.eval()
        with torch.no_grad():
            th_state = torch.from_numpy(state).unsqueeze(0).type(torch.get_default_dtype()).to(self.device)
            action = self.actor_net(th_state, with_noise=with_noise).detach().cpu().numpy().squeeze(axis=0)
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

        if len(self.memory) < max(self.batch_size, self.replay_memory_warmup_size):
            return

        for _ in range(self.repeat_update):
            # start = time.time()
            state_batch, action_batch, reward_batch, next_state_batch, non_terminal_mask = self.memory.sample(self.batch_size)
            # print('Batch time: {}'.format(time.time() - start))

            with torch.no_grad():
                # Compute Q'(s_{t+1}, mu'(s_{t+1}))
                next_actions = self.actor_net(next_state_batch, with_noise=True)
                next_Q_values_0 = self.critic_target_nets[0](next_state_batch, next_actions).squeeze(dim=1)
                next_Q_values_1 = self.critic_target_nets[1](next_state_batch, next_actions).squeeze(dim=1)
                next_Q_values = torch.minimum(next_Q_values_0, next_Q_values_1)
                expected_Q_values = reward_batch + self.gamma * next_Q_values * non_terminal_mask

            critic_losses = []
            for critic_net in self.critic_nets:
                # Compute Q(s_t, mu(s_t))
                Q_values = critic_net(state_batch, action_batch).squeeze(dim=1)
                critic_loss = F.mse_loss(Q_values, expected_Q_values)
                critic_losses.append(critic_loss)

            actions = self.actor_net(state_batch, with_noise=False)
            q0 = self.critic_nets[0](state_batch, actions)
            q1 = self.critic_nets[1](state_batch, actions)
            actor_loss = -torch.minimum(q0, q1).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Optimize the critics
            for i in range(len(self.critic_nets)):
                self.critic_optimizers[i].zero_grad()
                critic_losses[i].backward()
                self.critic_optimizers[i].step()

            # self.update_target_net(self.actor_target_net, self.actor_net, self.target_update_tau)
            for critic_net, critic_target_net in zip(self.critic_nets, self.critic_target_nets):
                self.update_target_net(critic_target_net, critic_net, self.target_update_tau)
