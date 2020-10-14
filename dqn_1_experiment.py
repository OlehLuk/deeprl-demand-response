from project import Project
from pyfmi import load_fmu
model = load_fmu(Project.get_fmu("PS_20TCL_v1.fmu"))
import logging
import time

from algo_procedures.dqn_1_delta_procedure import DQN_1_Delta_Wrapper
from algo_procedures.dqn_1_procedure import DQN_1_Wrapper
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
        "max_n_steps": 200,
        'actions': [0, 1, 2, 3, 4, 5, 6, 7],
        'n_state_variables': 2,
        'n_hidden_1': 64,
        'n_hidden_2': 64,
        'buffer_size': 512,
        'batch_size': 64,
        'exploration_rate': 0.5,
        'expl_rate_decay': 0.999,
        'expl_rate_final': 0.05,
        'discount_factor': 0.99,
        'target_update': 1000,
        'expl_decay_step': 1
    }

    experiment_config = {
        "exp_folder": "results\\dqn_1",
        "exp_id": f"1",
        "exp_repeat": 5,
        "n_episodes_train": 200,
        "n_episodes_test": 100
    }
    dqn_1 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_1_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    # dqn_1.run()

    experiment_config['exp_id'] = "3"
    dqn_3 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_1_Delta_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    # dqn_3.run()

    env_config['compute_reward'] = lambda u, v: 1/abs(u-v)
    env_config['reward_fct_str'] = "lambda u, v: 1 / abs(u - v)"
    experiment_config['exp_id'] = "2"
    dqn_2 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_1_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    # dqn_2.run()

    experiment_config['exp_id'] = "4"
    dqn_4 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_1_Delta_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    # dqn_4.run()

    env_config['compute_reward'] = None
    env_config['reward_fct_str'] = "default"
    experiment_config['exp_id'] = "5"
    experiment_config['n_episodes_train'] = 500
    experiment_config['exp_repeat'] = 3
    dqn_5 = GymExperiment(env_config, agent_config, experiment_config,
                          lambda x: DQN_1_Wrapper(**x),
                          save_experiment_output=save_ps_output,
                          test_env_config=test_env_config)
    dqn_5.run()
