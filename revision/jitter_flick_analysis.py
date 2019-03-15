import numpy as np
import connection_matrices as cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import os
import bluepy
from bluepy.v2 import Cell

import data_access_shuffling

import initial_analysis_final


def plot_psths():

    df_neurons = data_access_shuffling.get_selected_L456_gids()

    gids = np.array(df_neurons.index)
    print gids.shape

    bcs = data_access_shuffling.get_jitter_flick_blueconfigs()
    df_network = data_access_shuffling.get_spike_times_multiple(bcs).loc[gids]

    fig, axs = plt.subplots(6, figsize=(14, 14))
    t_start = 1000
    t_end = 8000
    df_network = df_network[(df_network['spike_time'] < t_end) & (df_network['spike_time'] >= t_start)]

    for k in range(30):
        times = df_network[df_network['spike_trial'] == k]['spike_time']
        axs[0].hist(times, bins=np.arange(t_start, t_end + 5, 5), color='red', histtype='step', alpha=0.5,
                    weights=np.zeros(times.size) + (1000 / 5.0) / gids.size)

        axs[0].set_xlim([t_start, t_end])
        axs[0].set_ylim([0, 15])

        axs[0].set_xlabel('t (ms)')

    gids_special = [74419, 75270, 93272]

    for i, gid in enumerate(gids_special):
        df_neuron = df_network.loc[gid]

        ax = axs[i + 1]
        ax.vlines(df_neuron['spike_time'], df_neuron['spike_trial'], df_neuron['spike_trial'] + 1, rasterized=True, lw=1)
        ax.set_xlim([t_start, t_end])
        ax.set_ylim([0, 30])
        ax.set_xlabel('t (ms)')
        ax.set_title(gid)
    plt.tight_layout()
    plt.savefig('figures/population_psths_jitter_flick.pdf', dpi=300)


def plot_fano_factors():
    n = 14

    fig, axs = plt.subplots(n, n, figsize=(28, 28))

    df_neurons = data_access_shuffling.get_selected_L456_gids()
    # df_neurons = data_access_shuffling.get_mc_2_gids()

    df_neurons_exc = df_neurons[df_neurons['synapse_class'] == 'EXC']

    means = np.zeros(n)
    errs = np.zeros(means.size)

    times = np.arange(-4000, 3000, 500)

    for i in range(n):
        print i
        ax = axs[0, i]
        counts_df = data_access_shuffling.get_spike_counts_jitter_flick(bin_id=i).loc[df_neurons_exc.index]

        ax.plot([0, 10], [0, 10], '--', color='red')

        ax.scatter(counts_df['mean'], counts_df['variance'], rasterized=True, marker='.', color='black',
                   s=0.5)
        ax.set_title(times[i])
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])

        ax.set_xlabel('Mean')
        ax.set_ylabel('Variance')

        ax = axs[1, i]
        ax.hist(counts_df['ff'], bins=np.arange(0, 5.2, 0.2), histtype='stepfilled', color='gray')
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 600])

        means[i] = counts_df['ff'].mean()
        errs[i] = counts_df['ff'].std()
        print counts_df['ff'].mean()

    ax = axs[2, 0]
    ax.plot(times, means, marker='.')
    ax.plot(times, means + errs, color='blue')
    ax.plot(times, means - errs, color='blue')
    ax.set_xticks(times)
    ax.set_xticklabels(times)

    ax.plot([-250, -250], [0, 3], '--', color='red')

    plt.savefig('figures/fano-factor_jitter_flick.pdf', dpi=300)


def plot_psths_control():

    df_neurons = data_access_shuffling.get_selected_L456_gids()

    gids = np.array(df_neurons.index)
    print gids.shape

    bcs = data_access_shuffling.get_jitter_flick_control_blueconfigs()
    df_network = data_access_shuffling.get_spike_times_multiple(bcs).loc[gids]

    fig, axs = plt.subplots(6, figsize=(14, 14))
    t_start = 1000
    t_end = 8000
    df_network = df_network[(df_network['spike_time'] < t_end) & (df_network['spike_time'] >= t_start)]

    for k in range(30):
        times = df_network[df_network['spike_trial'] == k]['spike_time']
        axs[0].hist(times, bins=np.arange(t_start, t_end + 5, 5), color='red', histtype='step', alpha=0.5,
                    weights=np.zeros(times.size) + (1000 / 5.0) / gids.size)

        axs[0].set_xlim([t_start, t_end])
        axs[0].set_ylim([0, 15])

        axs[0].set_xlabel('t (ms)')

    gids_special= [74419, 75270, 93272]

    for i, gid in enumerate(gids_special):
        df_neuron = df_network.loc[gid]

        ax = axs[i + 1]
        ax.vlines(df_neuron['spike_time'], df_neuron['spike_trial'], df_neuron['spike_trial'] + 1, rasterized=True, lw=1)
        ax.set_xlim([t_start, t_end])
        ax.set_ylim([0, 30])
        ax.set_xlabel('t (ms)')
        ax.set_title(gid)
    plt.tight_layout()
    plt.savefig('figures/population_psths_jitter_flick-control.pdf', dpi=300)


def plot_fano_factors_control():
    n = 14

    fig, axs = plt.subplots(n, n, figsize=(28, 28))

    df_neurons = data_access_shuffling.get_selected_L456_gids()
    # df_neurons = data_access_shuffling.get_mc_2_gids()

    df_neurons_exc = df_neurons[df_neurons['synapse_class'] == 'EXC']

    means = np.zeros(n)
    errs = np.zeros(means.size)

    times = np.arange(-4000, 3000, 500)

    for i in range(n):
        print i
        ax = axs[0, i]
        counts_df = data_access_shuffling.get_spike_counts_jitter_flick_control(bin_id=i).loc[df_neurons_exc.index]

        ax.plot([0, 10], [0, 10], '--', color='red')

        ax.scatter(counts_df['mean'], counts_df['variance'], rasterized=True, marker='.', color='black',
                   s=0.5)
        ax.set_title(times[i])
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])

        ax.set_xlabel('Mean')
        ax.set_ylabel('Variance')

        ax = axs[1, i]
        ax.hist(counts_df['ff'], bins=np.arange(0, 5.2, 0.2), histtype='stepfilled', color='gray')
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 600])

        means[i] = counts_df['ff'].mean()
        errs[i] = counts_df['ff'].std()
        print counts_df['ff'].mean()

    ax = axs[2, 0]
    ax.plot(times, means, marker='.')
    ax.plot(times, means + errs, color='blue')
    ax.plot(times, means - errs, color='blue')
    ax.set_xticks(times)
    ax.set_xticklabels(times)


    ax.plot([-250, -250], [0, 3], '--', color='red')


    plt.savefig('figures/fano-factor_jitter_flick-control.pdf', dpi=300)

plot_psths()
plot_fano_factors()
plot_isis()
plot_psths_control()
plot_fano_factors_control()