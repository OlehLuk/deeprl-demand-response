from project import Project
from pyfmi import load_fmu
model = load_fmu(Project.get_fmu("PS_20TCL_v1.fmu"))
import logging
import time

from algo_procedures.dqn_2_delta_procedure import DQN_2_Delta_Wrapper
from algo_procedures.dqn_2_procedure import DQN_2_Wrapper
from pipeline.experiment import GymExperiment
from pipeline.ps_env import save_ps_output

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
    test_env_config = {
        "entry_point": "pipeline:PSEnvV1",
        'p_reff': 1.2,
        'time_step': 1,
        'log_level': logging.INFO,
        'compute_reward': None,
        'p_reff_amplitude': 0,
        'p_reff_period': 200,
        'path': Project.get_fmu("PS_20TCL_v1.fmu"),
        'simulation_start_time': 0
    }
    agent_config = {
        'max_n_steps': 200,
        'input_dim': 2,
        'hidden_dim': 64,
        'capacity': 512,
        'gamma': 0.99,
        'batch_size': 64,
        'min_eps': 0.01,
        'eps_decrease_last_episode': 50,
        'actions': [0, 1, 2, 4, 8, 16, 32]
    }

    experiment_config = {
        "exp_folder": "results\\dqn_2",
        "exp_id": f"1",
        "exp_repeat": 5,
        "n_episodes_train": 200,
        "n_episodes_test": 100
    }

    # baseline = GymExperiment(env_config, agent_config, experiment_config,
    #                         lambda x: DQN_2_Wrapper(**x),
    #                         save_experiment_output=save_ps_output,
    #                         test_env_config=test_env_config)
    # baseline.run()
    experiment_config['exp_id'] = "3"
    dqn_3 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_2_Delta_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    # dqn_3.run()

    env_config['compute_reward'] = lambda u, v: 1/abs(u-v)
    env_config['reward_fct_str'] = "lambda u, v: 1 / abs(u - v)"
    experiment_config['exp_id'] = "2"
    dqn_2 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_2_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    # dqn_2.run()

    experiment_config['exp_id'] = "4"
    dqn_4 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_2_Delta_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    dqn_4.run()
