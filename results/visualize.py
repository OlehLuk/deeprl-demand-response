import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from IPython.core.pylabtools import figsize

FIRST_U = 0.8869
FIRST_P = 1.2


def report_experiment_summary(title, mses, perfs, sigs, labels, plot_interval_sigma=-1):
    figsize(9, 6)
    styles = ['r--', 'y-.', 'k', 'b.']
    color = ['r', 'y', 'k', 'b']
    for i, mse in enumerate(mses):
        plt.plot(mse, styles[i], label=labels[i])
    if plot_interval_sigma >= 0:
        plt.hlines(perfs, xmin=mses[0].index.min(), xmax=mses[0].index.max(), colors=color[:len(mses)],
                   label="Avg exploit performance")
    if plot_interval_sigma > 0:
        plt.hlines(np.array(perfs) + np.array(sigs), xmin=mses[0].index.min(), xmax=mses[0].index.max(),
                   colors=color[:len(mses)], linestyles="dashed")
        plt.hlines(np.array(perfs) - np.array(sigs), xmin=mses[0].index.min(), xmax=mses[0].index.max(),
                   colors=color[:len(mses)], linestyles="dashed")

    plt.title("Average smoothed last episode MSE vs training time. {}".format(title))
    plt.xlabel("# of episode")
    plt.ylabel("Average smoothed MSE")
    plt.legend()


def report_experiment(name, plot_perf_sigma=-1, fig_size=(9, 6), exp_name="", var_name="MSE", n_exp=5, stochastic=True):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    df_ep_ls = None # pd.read_csv("./{}/episodes_lengths.csv".format(name), header=None)
    df_ex_ts = pd.read_csv("./{}/exec_times.csv".format(name),
                           header=None)
    df_mses = pd.read_csv("./{}/train_mse.csv".format(name),
                          header=None)
    df_expl_perf = pd.read_csv("./{}/test_mse.csv".format(name),
                               header=None)
    df_avgsm_mses = plot_smoothed(df_mses, exp_name=exp_name, var_name=var_name, n_exp=n_exp, fig_size=fig_size)
    perf, sigma = print_exec_summary(df_ep_ls, df_ex_ts, df_expl_perf, stochastic)
    if plot_perf_sigma == 0:
        plt.hlines([perf], xmin=df_avgsm_mses.index.min(), xmax=df_avgsm_mses.index.max(),
                   colors=['k'], linestyles='dashed', label="Avg exploit. performance")
    elif plot_perf_sigma > 0:
        plt.hlines([perf, perf + plot_perf_sigma * sigma, perf - plot_perf_sigma * sigma],
                   xmin=df_avgsm_mses.index.min(),
                   xmax=df_avgsm_mses.index.max(), colors=['k', 'k', 'k'], linestyles='dashed',
                   label="Avg exploit. performance with {}-sigma interval bounds".format(plot_perf_sigma))
    plt.legend()
    return df_avgsm_mses, perf, sigma

def mse_vs_ref(labels, bas_folders, contr_folders, palette=["c", "lightgray"], fig_size=(9, 6), y_label = 'Performance (MSE)', x_label = 'Reference power', group_label = 'Control type'):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    
    contr_mses = []
    bas_mses = []
    
    
    df = pd.DataFrame(columns=[y_label, group_label, x_label])
    
    for i in range(len(labels)):
        perfs_c = pd.read_csv("{}/exploit_performance.csv".format(contr_folders[i]), header=None).values.flatten()
        df_temp_c = pd.DataFrame(data={y_label:perfs_c, group_label:"RL-driven controller", x_label:labels[i]})
        perfs_b = pd.read_csv("{}/mses.csv".format(bas_folders[i]), header=None).values.flatten()
        df_temp_b = pd.DataFrame(data={y_label:perfs_b, group_label:"Optimal constant control", x_label:labels[i]})
        df = pd.concat([df, df_temp_c, df_temp_b])
    
    sns.boxplot(data=df, x=x_label, y=y_label, hue=group_label, palette=palette)
    return bas_mses, contr_mses

def perf_box_plots(labels, bas_folders, contr_folders, color='lightgray', fig_size=(9, 6), y_label = 'Performance (MSE)', x_label = 'Reference power level'):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    
    contr_mses = []
    bas_mses = []
    
    df = pd.DataFrame(columns=[y_label, x_label])
    
    for i in range(len(contr_folders)):
        perfs_c = pd.read_csv("{}/exploit_performance.csv".format(contr_folders[i]), header=None).values.flatten()
        df_temp_c = pd.DataFrame(data={y_label:perfs_c, x_label:labels[i]})
        df = pd.concat([df, df_temp_c])
        
    for i in range(len(bas_folders)):
        perfs_b = pd.read_csv("{}/mses.csv".format(bas_folders[i]), header=None).values.flatten()
        df_temp_b = pd.DataFrame(data={y_label:perfs_b, x_label:"Optimal constant control"})
        df = pd.concat([df, df_temp_b])
    
    sns.boxplot(data=df, x=x_label, y=y_label, color=color)
    return bas_mses, contr_mses

def perf_violins(labels, bas_folders, contr_folders, color='lightgray', fig_size=(9, 6), y_label = 'Performance (MSE)', x_label = 'Reference power'):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    
    contr_mses = []
    bas_mses = []
    
    df = pd.DataFrame(columns=[y_label, x_label])
    
    for i in range(len(contr_folders)):
        perfs_c = pd.read_csv("{}/exploit_performance.csv".format(contr_folders[i]), header=None).values.flatten()
        df_temp_c = pd.DataFrame(data={y_label:perfs_c, x_label:labels[i]})
        df = pd.concat([df, df_temp_c])
        
    for i in range(len(bas_folders)):
        perfs_b = pd.read_csv("{}/mses.csv".format(bas_folders[i]), header=None).values.flatten()
        df_temp_b = pd.DataFrame(data={y_label:perfs_b, x_label:"Constant"})
        df = pd.concat([df, df_temp_b])
    
    sns.violinplot(data=df, x=x_label, y=y_label, color=color)
    return bas_mses, contr_mses

def perf_viol_split(labels, bas_folders, contr_folders, palette=["c", "lightgray"], fig_size=(9, 6), y_label = 'Performance (MSE)', x_label = 'Reference power', group_label = 'Control type'):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    
    contr_mses = []
    bas_mses = []
    
    
    df = pd.DataFrame(columns=[y_label, group_label, x_label])
    
    for i in range(len(labels)):
        perfs_c = pd.read_csv("{}/exploit_performance.csv".format(contr_folders[i]), header=None).values.flatten()
        df_temp_c = pd.DataFrame(data={y_label:perfs_c, group_label:"RL-driven controller", x_label:labels[i]})
        perfs_b = pd.read_csv("{}/mses.csv".format(bas_folders[i]), header=None).values.flatten()
        df_temp_b = pd.DataFrame(data={y_label:perfs_b, group_label:"Optimal constant control", x_label:labels[i]})
        df = pd.concat([df, df_temp_c, df_temp_b])
    
    sns.violinplot(data=df, x=x_label, y=y_label, hue=group_label, palette=palette, split=True)
    return bas_mses, contr_mses


def plot_smoothed(df, exp_name, var_name, n_exp=5, n_smoothing=20, fig_size=(9, 6)):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    transformed = df.rolling(window=n_smoothing, min_periods=n_smoothing, axis=0).mean()[n_smoothing - 1:]
    transformed['avg'] = transformed.mean(axis=1)
    plt.plot(transformed.index, transformed[[i for i in range(n_exp)]], 'gray',
             alpha=0.5)
    plt.plot(transformed.index, transformed['avg'], 'red', label="MSE of episode")
    plt.title("Smoothed {} of episode vs training time. {}".format(var_name, exp_name))
    plt.xlabel("# of episode")
    plt.ylabel(var_name)
    plt.legend()
    print("Last episode performance: {:.2f}".format(transformed['avg'].iloc[-1]))
    return transformed['avg']


def report_const_experiment(name, old=False, fig_size=(12, 20)):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    subfolder_names = list(filter(lambda s: "k=" in s, os.listdir(name)))
    n_variations = len(subfolder_names)
    df_us = []
    df_ps = []
    df_mses = []
    for i, subf in enumerate(subfolder_names):
        df_us.append(pd.read_csv("{}/{}/system_behavior/power_actual_test_0.csv".format(name, subf), header=None))
        df_ps.append(pd.read_csv("{}/{}/system_behavior/power_reference_test_0.csv".format(name, subf), header=None))
        df_mses.append(pd.read_csv("{}/{}/test_mse.csv".format(name, subf), header=None))

    for i, subf in enumerate(subfolder_names):
        plt.subplot(n_variations, 1, i + 1)
        if not old:
            plt.plot(df_us[i].index.values, df_us[i].iloc[:, 0].values, 'red', label="Actual power level")
            plt.plot(df_us[i].index.values, df_ps[i].iloc[:, 0].values, 'gray', label="Reference power level")
        else:
            index = list(range(df_us[i].index.shape[0] + 1))
            plt.plot(index, np.insert(df_us[i].iloc[:, 0].values, 0, FIRST_U), 'red', label="Actual power level")
            plt.plot(index, np.insert(df_ps[i].iloc[:, 0].values, 0, FIRST_P), 'gray', label="Reference power level")

        avg_mse = df_mses[i].mean()[0]
        std_mse = df_mses[i].std()[0]
        median_mse = np.median(df_mses[i])
        plt.title("Constant action {}. MSE: avg={:.4f}, std={:.4f}, median={:.4f}".format(subf, avg_mse, std_mse, median_mse))
        plt.legend()


def print_exec_summary(df_ep_ls, df_ex_ts, df_expl_perf, stochastic):
    n_steps_in_exp = 200*100
    t_per_step = df_ex_ts[0].values / n_steps_in_exp
    tot_ex_t = df_ex_ts.values.sum()
    tot_n_steps = n_steps_in_exp * 5
    avg_per_step = tot_ex_t / tot_n_steps
    print("Time per simulation step in each experiment: {} s".format(t_per_step))
    print("Mean time per simulation step: {:.3f} s, std: {:.4f} s".format(t_per_step.mean(), t_per_step.std()))
    print(
        "Total execution time: {:.3f} s for {} steps -> {:.3f} s per step".format(tot_ex_t, tot_n_steps, avg_per_step))

    performance = np.mean(df_expl_perf.mean(axis=0))
    if stochastic:
        sigma = np.mean(df_expl_perf.std(axis=0))
    else:
        sigma = df_expl_perf.std(axis=1)[0]
    median_perf = np.median(df_expl_perf.values.flatten())
    print("\nMedian performance (MSE) in exploitation mode {:.4f}".format(median_perf))
    print("\nAverage MSE in exploitation mode {:.4f} +- {:.4f}".format(performance, sigma))
    print("Average exploitation performance of each agent: \n{}".format(df_expl_perf.mean(axis=0).values))
    return performance, sigma


def plot_sys_behaviour(name, n_runs=5, steps=[0, -1], old=False, fig_size=(15, 20)):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    df_us = [pd.read_csv("./{}/system_trajectory/us_run_{}.csv".format(name, i), header=None) for i in range(n_runs)]
    df_ps = [pd.read_csv("./{}/system_trajectory/ps_run_{}.csv".format(name, i), header=None) for i in range(n_runs)]
    df_mses = pd.read_csv("./{}/mses.csv".format(name), header=None)
    for i in range(n_runs):
        for j, step in enumerate(steps):
            index = list(range(df_us[i].index.shape[0] + 1))
            plt.subplot(n_runs, len(steps), len(steps) * i + j + 1)
            if old:
                plt.plot(index, np.insert(df_us[i].iloc[:, step].values, 0, FIRST_U), 'red', label="Actual power level")
                plt.plot(index, np.insert(df_ps[i].iloc[:, step].values, 0, FIRST_P), 'gray',
                         label="Reference power level")
            else:
                plt.plot(df_us[i].index.values, df_us[i].iloc[:, step].values, 'red', label="Actual power level")
                plt.plot(df_us[i].index.values, df_ps[i].iloc[:, step].values, 'gray', label="Reference power level")
            if step != -1:
                plt.title("Training step #{}. MSE={}".format(step, df_mses.iloc[step, i]))
            else:
                plt.title("Last training step. MSE={}".format(df_mses.iloc[step, i]))
            plt.legend()
            
def plot_agent_sys_exploit_behaviour(name, n_runs=5, old=False, fig_size=(15, 20)):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    df_us = [pd.read_csv("./{}/system_trajectory/us_run_{}.csv".format(name, i), header=None) for i in range(n_runs)]
    df_ps = [pd.read_csv("./{}/system_trajectory/ps_run_{}.csv".format(name, i), header=None) for i in range(n_runs)]
    df_actions = [pd.read_csv("./{}/agent/action_run_{}.csv".format(name, i), header=None) for i in range(n_runs)]
    df_mses = pd.read_csv("./{}/mses.csv".format(name), header=None)
    for i in range(n_runs):
        
        plt.subplot(n_runs, 2, 2 * i + 1)
        if old:
            index = list(range(df_us[i].index.shape[0] + 1))
            plt.plot(index, np.insert(df_us[i].iloc[:, -1].values, 0, FIRST_U), 'red', label="Actual power level")
            plt.plot(index, np.insert(df_ps[i].iloc[:, -1].values, 0, FIRST_P), 'gray',
                     label="Reference power level")
        else:
            plt.plot(df_us[i].index.values, df_us[i].iloc[:, -1].values, 'red', label="Actual power level")
            plt.plot(df_us[i].index.values, df_ps[i].iloc[:, -1].values, 'gray', label="Reference power level")  
        plt.title("Last training step system behaviour. MSE={}".format(df_mses.iloc[-1, i]))
        
        plt.subplot(n_runs, 2, 2 * i + 2)
        plt.plot(df_actions[i].index.values, df_actions[i].iloc[:, -1].values, 'o', label="Action performed (k)")
        plt.title("Last training step agent behaviour. MSE={}".format(df_mses.iloc[-1, i]))
        
        plt.legend()


def plot_agent_behaviour(name, n_runs=5, steps=[0, -1], fig_size=(15, 20)):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    df_actions = [pd.read_csv("./{}/agent/action_run_{}.csv".format(name, i), header=None) for i in range(n_runs)]
    df_mses = pd.read_csv("./{}/mses.csv".format(name), header=None)
    for i in range(n_runs):
        for j, step in enumerate(steps):
            plt.subplot(n_runs, len(steps), len(steps) * i + j + 1)
            plt.plot(df_actions[i].index.values, df_actions[i].iloc[:, step].values, 'o', label="Action performed (k)")
            if step != -1:
                plt.title("Step #{}. MSE={}".format(step, df_mses.iloc[step, i]))
            else:
                plt.title("Last step. MSE={}".format(df_mses.iloc[step, i]))
            plt.legend()

def plot_two_perf_dist(baseline, controller, fig_size=(12, 25)):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    x = pd.read_csv(controller + "/test_mse.csv", header=None)
    y = pd.read_csv(baseline + "/test_mse.csv", header=None)[0].values
    y_mean = y.mean()
    y_std = y.std()
    y_median = np.median(y)
    n = x.shape[1]
    for i in range(n):
        plt.subplot(n+1, 1, i+1)
        plt.subplots_adjust(hspace=0.4)
        x_mean = x[i].mean()
        x_std = x[i].std()
        plt.title("Agent performance: mean={:.4f}, std={:.4f} \n Constant control performance:  mean={:.4f}, std={:.4f}".format(
            x_mean, x_std, y_mean, y_std))
        ax = sns.distplot(x[i].values, color='red', label="Agent#{}".format(i+1))
        sns.distplot(y, ax=ax, color='gray', label="Optimal constant control")
        ax.legend()
    
    plt.subplot(n+1, 1, n+1)
    performance = np.mean(x.mean(axis=0))
    sigma = np.mean(x.std(axis=0))
    median = np.median(x.values.flatten())
    
    
    ax = sns.distplot(x.values.flatten(), color='red', label="RL-driven controller")
    plt.title("Summarized agents performance.\nRL-driven controller: median={:.4f}, mean={:.4f}, std={:.4f} \n Optimal constant control:  median={:.4f}, mean={:.4f}, std={:.4f}".format(median, performance, sigma, y_median, y_mean, y_std))
    sns.distplot(y, ax=ax, color='gray', label="Optimal constant control", axlabel="Performance (MSE)")
    ax.legend()
    return 


def plot_perf_dist(baseline, controllers, labels, fig_size=(12, 25), x_label=""):
    fig_x, fig_y = fig_size
    figsize(fig_x, fig_y)
    x = [pd.read_csv(controller + "/test_mse.csv", header=None) for controller in controllers]
    y = pd.read_csv(baseline + "/test_mse.csv", header=None)[0].values
    y_mean = y.mean()
    y_std = y.std()
    n = x[0].shape[1]
    
    for i in range(n):
        plt.subplot(n+1, 1, i+1)
        plt.subplots_adjust(hspace=0.4)
        
        x_mean = [x_i[i].mean() for x_i in x]
        x_std = [x_i[i].std() for x_i in x]
        plt.title("Constant control performance:  mean={:.4f}, std={:.4f}".format(y_mean, y_std))
        
        ax = sns.distplot(y, color='gray', label="Optimal constant control")
        for j, x_i in enumerate(x):
            sns.distplot(x_i[i].values, ax=ax, label="{} Agent#{}".format(labels[j], i+1))
        ax.legend()
        ax.set_xlabel(x_label)
    
    plt.subplot(n+1, 1, n+1)
    performance = [np.mean(x_i.mean(axis=0)) for x_i in x]
    sigma = [np.mean(x_i.std(axis=0)) for x_i in x]
    ax = sns.distplot(y, color='gray', label="Optimal constant control", axlabel="Performance (MSE)")
    
    for j, x_i in enumerate(x):
        sns.distplot(x_i[i].values, ax=ax, label="{}".format(labels[j]))
    
    
    
    ax.legend()
    return 
