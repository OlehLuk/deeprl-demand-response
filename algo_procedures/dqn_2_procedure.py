import numpy as np

from algo_procedures.dqn_2 import DQN2Agent, ReplayMemory, train_helper
from pipeline.experiment import AgentWrapper


class DQN_2_Wrapper(AgentWrapper):
    def __init__(self, max_n_steps):
        self.max_n_steps = max_n_steps

    def play_episode(self, env,
                     agent: DQN2Agent,
                     replay_memory: ReplayMemory,
                     eps: float,
                     batch_size: int,
                     gamma=0.99) -> int:
        """Play an epsiode and train
        Args:
            env (gym.Env): gym environment (CartPole-v0)
            agent (DQN2Agent): agent will train and get action
            replay_memory (ReplayMemory): trajectory is saved here
            eps (float): 𝜺-greedy for exploration
            batch_size (int): batch size
        Returns:
            int: reward earned in this episode
        """
        s = env.reset()
        s = np.array(s)
        done = False
        total_reward = 0
        episode_length = 0

        while not done:

            a = agent.get_action(s, eps)
            s2, r, done, info = env.step(a)
            s2 = np.array(s2)

            if done:
                r = -1
            replay_memory.push(s, a, r, s2, done)

            if len(replay_memory) > batch_size:
                minibatch = replay_memory.pop(batch_size)
                train_helper(agent, minibatch, gamma)

            s = s2

        return episode_length

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
