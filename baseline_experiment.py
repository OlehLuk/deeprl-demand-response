import logging
import time

from project import Project

from pipeline.experiment import GymExperiment
from algo_procedures.baseline import ConstantControlWrapper


if __name__ == '__main__':
    env_config = {
        "env_name": "PSwithFMU-v0",
        "entry_point": "pipeline:OMCSPSStochasticEnv",
        'p_reff': 1.2,
        'time_step': 1,
        'log_level': logging.INFO,
        'compute_reward': None,
        'p_reff_amplitude': 0,
        'p_reff_period': 200,
        'get_seed': lambda: round(time.time()),
        'path': Project.get_fmu("PS_1.fmu"),
        'simulation_start_time': 0
    }

    agent_config = {
        'constant_control': 5,
        'max_steps': 200
    }

    experiment_config = {
        "exp_folder": "results/baseline",
        "exp_id": "k=5",
        "exp_repeat": 5,
        "n_episodes_train": 200,
        "n_episodes_test": 100
    }
    baseline = GymExperiment(env_config, agent_config, experiment_config,
                             lambda x: ConstantControlWrapper(**x), )
    baseline.run()
