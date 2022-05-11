import random


class EnvironmentsParallelBeam:

    class EnvironmentWrapper:
        def __init__(self, environment):
            self.environment = environment
            self.last_state = environment.reset()

            self.cumulative_reward = 0
            self.total_reward = 0
            self.total_steps = 0

        def step(self, actions):
            next_state, reward, is_terminal, info = self.environment.step(actions)
            self.cumulative_reward += reward
            if is_terminal:
                self.total_reward = self.cumulative_reward
                self.cumulative_reward = 0
                next_state = self.environment.reset()
            self.last_state = next_state
            self.total_steps += 1
            return next_state, reward, is_terminal, info

    def __init__(self, create_environment_function, size, start_seed):
        self.size = size
        self.environments = []
        self.i = 0
        self.total_steps = 0

        for i in range(self.size):
            env = create_environment_function()
            env.seed(start_seed + i)
            env_wrapper = self.EnvironmentWrapper(env)
            self.environments.append(env_wrapper)

    def _max_i(self):
        return min(self.size, 1 + self.total_steps // 100)

    def step(self, actions):
        ret = self.environments[self.i].step(actions)
        # self.i = (self.i + 1) % self._max_i()
        self.i = random.randint(0, self._max_i() - 1)
        self.total_steps += 1
        return ret

    def get_last_state(self):
        return self.environments[self.i].last_state

    def get_mean_total_reward(self):
        return sum([env.total_reward for env in self.environments[:self._max_i()]]) / self._max_i()