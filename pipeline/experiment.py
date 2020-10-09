import time
import gym
import os
import json
import numpy as np


def save_dict_to_json(dictionary, exp_subfolder, file_name):
    exp_config_file = os.path.join(exp_subfolder, file_name)
    with open(exp_config_file, 'w') as json_file:
        json.dump(dictionary, json_file)


class AgentWrapper(object):
    def run_one_train_episode(self, env) -> dict:
        pass

    def run_one_test_episode(self, env) -> dict:
        pass


def default_save_output(all_outputs, exp_subfolder):
    for i, output in enumerate(all_outputs):
        exp_train_results, exp_test_results = output
        train_name = os.path.join(exp_subfolder, f"train_output_{i+1}.json")
        test_name = os.path.join(exp_subfolder, f"test_output_{i+1}.json")
        with open(train_name, 'w') as json_file:
            json.dump({'output': exp_train_results}, json_file)
        with open(test_name, 'w') as json_file:
            json.dump({'output': exp_test_results}, json_file)


class GymExperiment:
    def __init__(self, env_config: dict, alg_config: dict, exp_config: dict,
                 create_agent_wrapper, save_experiment_output=default_save_output):
        self.env_config = env_config

        self.env_name = self.env_config.pop("env_name", "PSwithFMU-v0")
        self.env_entry_point = self.env_config.pop("entry_point", "environments:JModelicaCSCartPoleEnv")

        self.alg_config = alg_config

        self.exp_config = exp_config

        self.exp_folder = exp_config.get("exp_folder", ".")
        self.exp_id = exp_config.get("exp_id", "0")
        self.exp_repeat = exp_config.get("exp_folder", 1)
        self.n_episodes_train = exp_config.get("n_episodes_train", 200)
        self.n_episodes_test = exp_config.get("n_episodes_test", 100)

        self.create_agent_wrapper = create_agent_wrapper
        self.save_experiment_output = save_experiment_output

    def run(self):
        exp_subfolder = self.create_setup()
        env = self.create_env()

        exec_times = []
        all_outputs = []

        for i in range(self.exp_repeat):
            # TODO: add a progress bar
            start = time.time()

            output = self.run_one_experiment(env, self.alg_config)
            exec_times.append(time.time() - start)
            all_outputs.append(output)

        self.save_experiment_output(all_outputs, exp_subfolder)

        np.savetxt(fname=os.path.join(exp_subfolder, "exec_times.csv"),
                   X=np.transpose(exec_times),
                   delimiter=",",
                   fmt="%d")

        env.close()
        del gym.envs.registry.env_specs[self.env_name]

    def create_env(self):
        from gym.envs.registration import register
        register(
            id=self.env_name,
            entry_point=self.env_entry_point,
            kwargs=self.env_config
        )
        env = gym.make(self.env_name)
        return env

    def create_setup(self):
        # create subfolder
        exp_subfolder = os.path.join(self.exp_folder, self.exp_id)
        if os.path.exists(exp_subfolder):
            raise FileExistsError("Experiment folder exists, not overriding it")
        else:
            os.mkdir(exp_subfolder)
        # save files with experiment config
        save_dict_to_json(self.env_config, exp_subfolder, "env_config.json")
        save_dict_to_json(self.exp_config, exp_subfolder, "exp_config.json")
        save_dict_to_json(self.alg_config, exp_subfolder, "alg_config.json")
        return exp_subfolder

    def run_one_experiment(self, env, alg_config):
        exp_train_output = []
        agent_wrapper: AgentWrapper = self.create_agent_wrapper(alg_config)
        for i in range(self.n_episodes_train):
            # TODO: allow early stopping
            episode_output = agent_wrapper.run_one_train_episode(env)
            exp_train_output.append(episode_output)
            env.reset()

        exp_test_output = []
        for i in range(self.n_episodes_test):
            episode_output = agent_wrapper.run_one_test_episode(env)
            exp_test_output.append(episode_output)
            env.reset()

        return exp_train_output, exp_test_output

    def run_one_episode(self, env, agent):
        pass

    def save_output(self, i, output, exp_subfolder):
        pass
