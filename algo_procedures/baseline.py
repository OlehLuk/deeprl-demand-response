from pipeline.experiment import AgentWrapper


class ConstantControlWrapper(AgentWrapper):
    def __init__(self, constant_control=7, max_steps=200):
        self.k = constant_control
        self.max_steps = max_steps

    def run_one_train_episode(self, env) -> dict:
        output = []
        for i in range(self.max_steps):
            obs, rew, done, _ = env.step(self.k)
            output.append(obs)
            if done:
                break
        return output

    def run_one_test_episode(self, env) -> dict:
        return self.run_one_test_episode(env)

