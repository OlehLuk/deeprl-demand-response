import logging
import time

from pipeline import PSTestSeed
from pipeline.ps_env import save_ps_output
from project import Project

from pipeline.experiment import GymExperiment
from algo_procedures.baseline import ConstantControlWrapper


if __name__ == '__main__':
    env_config = {
        "env_name": "PSwithFMU-v0",
        "entry_point": "pipeline:PSEnvV1",
        'p_reff': 1.2,
        'time_step': 1,
        'log_level': logging.INFO,
        'compute_reward': None,
        'p_reff_amplitude': 0,
        'p_reff_period': 200,
        'get_seed': lambda: round(time.time()),
        'path': Project.get_fmu("PS_20TCL_v1.fmu"),
        'simulation_start_time': 0
    }
    fixed_test_set = PSTestSeed(100)
    test_env_config = {
        "env_name": "PSwithFMU-v0",
        "entry_point": "pipeline:PSEnvV1",
        'p_reff': 1.2,
        'time_step': 1,
        'log_level': logging.INFO,
        'compute_reward': None,
        'p_reff_amplitude': 0,
        'p_reff_period': 200,
        'get_seed': lambda: fixed_test_set.get_seed(),
        'path': Project.get_fmu("PS_20TCL_v1.fmu"),
        'simulation_start_time': 0
    }

    agent_config = {
        'constant_control': 6,
        'max_steps': 200
    }

    experiment_config = {
        "exp_folder": "results/baseline/",
        "exp_id": "k=6",
        "exp_repeat": 2,
        "n_episodes_train": 0,
        "n_episodes_test": 10
    }
    baseline = GymExperiment(env_config, agent_config, experiment_config,
                             lambda x: ConstantControlWrapper(**x),
                             save_experiment_output=save_ps_output)
    baseline.run()
