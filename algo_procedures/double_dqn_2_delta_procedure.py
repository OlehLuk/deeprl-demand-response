import numpy as np

from algo_procedures.double_dqn_2 import Double_DQN2Agent, ReplayMemory, train_helper, epsilon_annealing
from pipeline import calc_mse
from pipeline.experiment import AgentWrapper


class Double_DQN_2_Delta_Wrapper(AgentWrapper):
    def __init__(self, max_n_steps, input_dim,
                 hidden_dim, capacity,
                 gamma, batch_size, min_eps,
                 eps_decrease_last_episode, actions,
                 target_update):
        self.target_update = target_update
        self.actions = actions
        self.min_eps = min_eps
        self.eps_decrease_last_episode = eps_decrease_last_episode
        self.agent = Double_DQN2Agent(input_dim + 1, len(actions), hidden_dim, target_update)
        self.replay_memory = ReplayMemory(capacity)
        self.max_n_steps = max_n_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.episode_counter = 0

    def run_one_train_episode(self, env) -> dict:
        state = env.reset()
        p_actual, p_reference = state
        state = np.array(state)
        delta = 0
        eps = epsilon_annealing(self.episode_counter, self.eps_decrease_last_episode, self.min_eps)
        action_index = self.agent.get_action(np.array([*state, delta]), eps)
        action = self.actions[action_index]
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [action]}

        for i in range(self.max_n_steps):
            next_state, reward, done, _ = env.step(action)
            next_delta = next_state[0] - state[0]
            p_actual, p_reference = next_state
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(action)
            next_state = np.array(next_state)
            self.replay_memory.push(np.array([*state, delta]),
                                    action_index, reward,
                                    np.array([*next_state, next_delta]), done)

            if len(self.replay_memory) > self.batch_size:
                minibatch = self.replay_memory.pop(self.batch_size)
                train_helper(self.agent, minibatch, self.gamma)

            self.agent.target_update_step()
            state = next_state
            delta = next_delta
            action_index = self.agent.get_action(np.array([*state, delta]), eps)
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
        delta = 0
        state = np.array(state)
        eps = self.min_eps
        action_index = self.agent.get_action(np.array([*state, delta]), eps)
        action = self.actions[action_index]
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [action]}

        for i in range(self.max_n_steps):
            next_state, reward, done, _ = env.step(action)
            p_actual, p_reference = next_state
            delta = next_state[0] - state[0]
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(action)
            state = np.array(next_state)
            action_index = self.agent.get_action(np.array([*state, delta]), eps)
            action = self.actions[action_index]

            if done:
                break

        mse = calc_mse(output['p_actual'], output['p_reference'])
        output['mse'] = mse
        return output
