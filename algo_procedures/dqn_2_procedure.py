import numpy as np

from algo_procedures.dqn_2 import DQN2Agent, ReplayMemory, train_helper, epsilon_annealing
from pipeline import calc_mse
from pipeline.experiment import AgentWrapper


class DQN_2_Wrapper(AgentWrapper):
    def __init__(self, max_n_steps, input_dim,
                 hidden_dim, capacity,
                 gamma, batch_size, min_eps,
                 eps_decrease_last_episode, actions):
        self.actions = actions
        self.min_eps = min_eps
        self.eps_decrease_last_episode = eps_decrease_last_episode
        self.agent = DQN2Agent(input_dim, len(actions), hidden_dim)
        self.replay_memory = ReplayMemory(capacity)
        self.max_n_steps = max_n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.episode_counter = 0

    def run_one_train_episode(self, env) -> dict:
        state = env.reset()
        p_actual, p_reference = state
        state = np.array(state)
        eps = epsilon_annealing(self.episode_counter, self.eps_decrease_last_episode, self.min_eps)
        action_index = self.agent.get_action(state, eps)
        action = self.actions[action_index]
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [action]}

        for i in range(self.max_n_steps):
            next_state, reward, done, _ = env.step(action)
            p_actual, p_reference = next_state
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(action)
            next_state = np.array(next_state)
            self.replay_memory.push(state, action_index, reward, next_state, done)

            if len(self.replay_memory) > self.batch_size:
                minibatch = self.replay_memory.pop(self.batch_size)
                train_helper(self.agent, minibatch, self.gamma)

            state = next_state
            eps = epsilon_annealing(self.episode_counter, self.eps_decrease_last_episode, self.min_eps)
            action_index = self.agent.get_action(state, eps)
            action = self.actions[action_index]

            if done:
                break

        mse = calc_mse(output['p_actual'], output['p_reference'])
        output['mse'] = mse
        self.episode_counter += 1
        return output

    def run_one_test_episode(self, env) -> dict:
        state = env.reset()
        p_actual, p_reference = state

        state = np.array(state)
        eps = self.min_eps
        action_index = self.agent.get_action(state, eps)
        action = self.actions[action_index]
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [action]}

        for i in range(self.max_n_steps):
            next_state, reward, done, _ = env.step(action)
            p_actual, p_reference = next_state
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(action)
            state = np.array(next_state)
            action_index = self.agent.get_action(state, eps)
            action = self.actions[action_index]

            if done:
                break

        mse = calc_mse(output['p_actual'], output['p_reference'])
        output['mse'] = mse
        return output
