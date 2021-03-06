from project import Project
from pyfmi import load_fmu
model = load_fmu(Project.get_fmu("PS_40TCL_om12.fmu"))
import logging
import time

from pipeline.ps_env import save_ps_output
from project import Project

from pipeline.experiment import GymExperiment
from algo_procedures.baseline import ConstantControlWrapper


def baseline_experiment(folder, ks, p_ref=1.2):
    for k in ks:
        env_config = {
            "env_name": "PSwithFMU-v0",
            "entry_point": "pipeline:PSEnvV1",
            'p_reff': p_ref,
            'time_step': 1,
            'log_level': logging.INFO,
            'compute_reward': None,
            'p_reff_amplitude': 0,
            'p_reff_period': 200,
            'get_seed': lambda: round(time.time()),
            'path': Project.get_fmu("PS_40TCL_om12.fmu"),
            'simulation_start_time': 0
        }
        test_env_config = {
            "entry_point": "pipeline:PSEnvV1",
            'p_reff': p_ref,
            'time_step': 1,
            'log_level': logging.INFO,
            'compute_reward': None,
            'p_reff_amplitude': 0,
            'p_reff_period': 200,
            # 'path': Project.get_fmu("PS_20TCL_v1.fmu"),
            'path': Project.get_fmu("PS_40TCL_om12.fmu"),
            'simulation_start_time': 0
        }

        agent_config = {
            'constant_control': k,
            'max_steps': 200
        }

        experiment_config = {
            "exp_folder": folder,
            "exp_id": f"k={k}",
            "exp_repeat": 1,
            "n_episodes_train": 0,
            "n_episodes_test": 100
        }
        baseline = GymExperiment(env_config, agent_config, experiment_config,
                                 lambda x: ConstantControlWrapper(**x),
                                 save_experiment_output=save_ps_output,
                                 test_env_config=test_env_config)
        baseline.run()


if __name__ == '__main__':
    # baseline_experiment("results/baseline_tcl_40/p_ref=2.3", ks=[0, 1, 2, 3, 4, 5, 6, 7], p_ref=2.3)
    # baseline_experiment("results/baseline_tcl_40/p_ref=1.2", ks=[0])
    baseline_experiment("results/baseline_tcl_40/p_ref=1.2_", ks=[5])
    # baseline_experiment("results/baseline/p_ref=1.2", ks=[3])
    # baseline_experiment("results/baseline", ks=[3])
