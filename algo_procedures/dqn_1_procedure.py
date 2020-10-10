from algo_procedures.dqn_1 import DqnAgent
from pipeline.experiment import AgentWrapper


class DQN_1_Wrapper(AgentWrapper):
    def __init__(self, actions, n_state_variables, n_hidden_1, n_hidden_2,
                 buffer_size, batch_size, exploration_rate, expl_rate_decay, expl_rate_final,
                 discount_factor, target_update, expl_decay_step, max_n_steps):
        self.max_n_steps = max_n_steps
        self.agent = DqnAgent(actions, n_state_variables, n_hidden_1, n_hidden_2,
                 buffer_size, batch_size, exploration_rate, expl_rate_decay, expl_rate_final,
                 discount_factor, target_update, expl_decay_step)
        pass

    def run_one_train_episode(self, env) -> dict:
        state = env.reset()
        p_actual, p_reference = state
        action = self.agent.use(state)
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [action]}

        for i in range(self.max_n_steps):
            next_state, reward, done, _ = env.step(action)
            p_actual, p_reference = next_state
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(action)

            action = self.agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break

        return output

    def run_one_test_episode(self, env) -> dict:
        state = env.reset()
        p_actual, p_reference = state
        action = self.agent.use(state)
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [action]}

        for i in range(self.max_n_steps):
            next_state, reward, done, _ = env.step(action)
            p_actual, p_reference = next_state
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(action)

            action = self.agent.use(next_state)
            if done:
                break

        return output
