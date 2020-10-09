from algo_procedures.dqn_1 import DqnAgent
from pipeline.experiment import AgentWrapper


class DQN_2_Wrapper(AgentWrapper):
    def __init__(self, max_n_steps):
        self.max_n_steps = max_n_steps


    def run_one_train_episode(self, env) -> dict:
        output = []
        state = env.reset()
        action = self.agent.use(state)
        for i in range(self.max_n_steps):
            # TODO: save observation to output properly
            next_state, reward, done, _ = env.step(action)
            output.append(next_state)
            action = self.agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                break

        return output

    def run_one_test_episode(self, env) -> dict:
        output = []
        state = env.reset()
        action = self.agent.use(state)
        for i in range(self.max_n_steps):
            # TODO: save observation to output properly
            next_state, reward, done, _ = env.step(action)
            output.append(next_state)
            action = self.agent.use(next_state)
            if done:
                break

        return output
