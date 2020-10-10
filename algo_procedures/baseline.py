from pipeline.experiment import AgentWrapper


class ConstantControlWrapper(AgentWrapper):
    def __init__(self, constant_control=7, max_steps=200):
        self.k = constant_control
        self.max_steps = max_steps

    def run_one_train_episode(self, env) -> dict:
        state = env.reset()
        p_actual, p_reference = state
        output = {'p_actual': [p_actual], 'p_reference': [p_reference], 'action': [self.k]}

        for i in range(self.max_steps):
            obs, rew, done, _ = env.step(self.k)
            p_actual, p_reference = obs
            output['p_actual'].append(p_actual)
            output['p_reference'].append(p_reference)
            output['action'].append(self.k)
            if done:
                break
        return output

    def run_one_test_episode(self, env) -> dict:
        return self.run_one_test_episode(env)

