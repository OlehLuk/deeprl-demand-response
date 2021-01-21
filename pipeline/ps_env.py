import json
import logging
import os
import time

import numpy as np
from gym import spaces

from modelicagym.environment import FMI2CSEnv
from project import Project

logger = logging.getLogger(__name__)


# P_DIFF_THRESHOLD = 1
P_DIFF_THRESHOLD = 10


def calc_mse(a, b):
    return np.mean((np.array(a)-np.array(b))**2)


class PSEnv:
    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return True

    def render(self, mode='human', close=False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Draws cart-pole with the built-in gym tools.

        :param mode: rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        :return: rendering result
        """
        u, p = self.state
        print("Current value of u: {}, p_reference: {}".format(u, p))
        return None

    def reset(self):
        """
        OpenAI Gym API. Restarts environment.
        Cleans saved difference.
        :return: state after restart
        """
        self.p_diffs = []
        return super().reset()

    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size 7, as currently only 7 actions are considered:
        k = 0,1,2,..,6
        """
        return spaces.Box(np.array([0]), np.array([np.inf]))

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements

        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([np.inf, np.inf])

        return spaces.Box(np.array([0, 0]), high)

    def _is_done(self):
        """
        Internal logic that is utilized by parent classes.
        Checks if power demand is not higher than reference + threshold.

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        True, if current value of demand is bigger than p_reference + threshold
        """
        u, p = self.state
        logger.debug("u (actual): {}".format(u))

        if u > p + self.p_diff_threshold:
            return True
        else:
            return False

    def _reward_policy(self):
        """
        Internal logic that is utilized by parent classes. Provides custom reward policy.
        :return: MSE between vector of reference and actual demand
        """
        u, p = self.state
        if self.compute_reward is None:
            return -1000 * (u - p) * (u - p) if not self._is_done() else -1000
        else:
            return self.compute_reward(u, p)


class PSEnvV1(PSEnv, FMI2CSEnv):
    def __init__(self,
                 p_reff,
                 time_step,
                 log_level,
                 compute_reward=None,
                 p_reff_amplitude=0,
                 p_reff_period=200,
                 get_seed=lambda: round(time.time()),
                 p_diff_threshold=P_DIFF_THRESHOLD,
                 path=Project.get_fmu("PS_20TCL_v1.fmu"),
                 simulation_start_time=0,
                 *args, **kwargs):

        logger.setLevel(log_level)
        self.p_reff = p_reff
        self.p_diff_threshold = p_diff_threshold
        self.compute_reward = compute_reward
        self.p_diffs = []
        self.viewer = None
        self.display = None
        self.get_seed = get_seed
        self.time_step = time_step

        config = {
            'model_input_names': ['k_in'],
            'model_output_names': ['controller_New.u2', 'controller_New.p'],
            'model_parameters': {'P_ref.height': p_reff, 'P_ref_change.amplitude': p_reff_amplitude,
                                 'P_ref_change.period': p_reff_period, "globalSeed.useAutomaticSeed": 0,
                                 "globalSeed.fixedSeed": self.get_seed()},
            'time_step': time_step
        }
        super().__init__(path,
                         config, log_level, simulation_start_time=simulation_start_time)

    def reset(self):
        """
        OpenAI Gym API. Restarts environment.
        Cleans saved difference.
        :return: state after restart
        """
        self.model_parameters.update({"globalSeed.useAutomaticSeed": 0})
        self.model_parameters.update({"globalSeed.fixedSeed": self.get_seed()})
        return super().reset()


def save_ps_output(all_outputs, exp_subfolder):
    mses_train = []
    mses_test = []

    raw_subfolder = os.path.join(exp_subfolder, "raw_data")
    if os.path.exists(raw_subfolder):
        raise FileExistsError("Experiment folder exists, not overriding it")
    else:
        os.mkdir(raw_subfolder)

    system_subfolder = os.path.join(exp_subfolder, "system_behavior")
    if os.path.exists(system_subfolder):
        raise FileExistsError("Experiment folder exists, not overriding it")
    else:
        os.mkdir(system_subfolder)

    for i, output in enumerate(all_outputs):
        exp_train_results, exp_test_results = output
        # save raw data
        train_name = os.path.join(raw_subfolder, f"train_output_{i + 1}.json")
        test_name = os.path.join(raw_subfolder, f"test_output_{i + 1}.json")
        with open(train_name, 'w') as json_file:
            json.dump({'output': exp_train_results}, json_file)
        with open(test_name, 'w') as json_file:
            json.dump({'output': exp_test_results}, json_file)

        # save mse data
        mses_train.append([episode_output['mse'] for episode_output in exp_train_results])
        mses_test.append([episode_output['mse'] for episode_output in exp_test_results])

        pa_train = [episode_output['p_actual'] for episode_output in exp_train_results]
        pa_test = [episode_output['p_actual'] for episode_output in exp_test_results]
        pr_train = [episode_output['p_reference'] for episode_output in exp_train_results]
        pr_test = [episode_output['p_reference'] for episode_output in exp_test_results]
        actions_train = [episode_output['action'] for episode_output in exp_train_results]
        actions_test = [episode_output['action'] for episode_output in exp_test_results]

        np.savetxt(fname=os.path.join(system_subfolder, f"actions_train_{i}.csv"),
                   X=np.transpose(actions_train),
                   delimiter=",",
                   fmt="%d")

        np.savetxt(fname=os.path.join(system_subfolder, f"actions_test_{i}.csv"),
                   X=np.transpose(actions_test),
                   delimiter=",",
                   fmt="%d")

        np.savetxt(fname=os.path.join(system_subfolder, f"power_actual_train_{i}.csv"),
                   X=np.transpose(pa_train),
                   delimiter=",",
                   fmt="%.4f")

        np.savetxt(fname=os.path.join(system_subfolder, f"power_actual_test_{i}.csv"),
                   X=np.transpose(pa_test),
                   delimiter=",",
                   fmt="%.4f")
        np.savetxt(fname=os.path.join(system_subfolder, f"power_reference_train_{i}.csv"),
                   X=np.transpose(pr_train),
                   delimiter=",",
                   fmt="%.4f")

        np.savetxt(fname=os.path.join(system_subfolder, f"power_reference_test_{i}.csv"),
                   X=np.transpose(pr_test),
                   delimiter=",",
                   fmt="%.4f")


    np.savetxt(fname=os.path.join(exp_subfolder, "train_mse.csv"),
               X=np.transpose(mses_train),
               delimiter=",",
               fmt="%.4f")

    np.savetxt(fname=os.path.join(exp_subfolder, "test_mse.csv"),
               X=np.transpose(mses_test),
               delimiter=",",
               fmt="%.4f")
