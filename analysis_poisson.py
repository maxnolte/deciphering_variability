import initial_analysis_final as iaf
from analysis_exp_25_decoupled import load_spike_times
from analysis_exp_25_decoupled import get_spike_times_experiment_25
from analysis_exp_25_decoupled import get_selected_L456_gids


import os
import bluepy
from bluepy.v2 import Cell

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42


def get_input_spikes_exp_25(n=0):
    variable_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_variable/seed%d/experiment_25.dat'
    variable_seeds = np.arange(200, 230, dtype=np.int)
    file_path_orig = variable_path % variable_seeds[n]
    data = pd.read_table(file_path_orig, header=0, names=('spikes', 'gids'))
    return data

def get_all_input_spikes_exp_25():
    trial_dfs = []
    for j in range(30):
        # df = simulation.v2.spikes.get(gids=gids)
        df = get_input_spikes_exp_25(n=j)
        spike_series = pd.Series(df.spikes.values, index=df.gids, dtype=float)
        print spike_series
        trial_series = pd.Series(j * np.ones(len(spike_series)), index=spike_series.index, dtype=int)
        trial_dfs.append(pd.concat([spike_series, trial_series], axis=1))
    df = pd.concat(trial_dfs, axis=0)
    df.columns = ['time', 'trial']
    return df

def load_spike_times_spontaneous():
    bcs = iaf.get_continue_bcs(n=40)
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df


def get_spike_times_spontaneous():
    directory = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/'
    spike_file = 'spontaneous_spikes.pkl'
    file_name = os.path.join(directory, spike_file)
    if os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
    else:
        df = load_spike_times_spontaneous()
        df.to_pickle(file_name)
    return df


def compute_spike_counts(df, n=30, vpm=False):
    bcs = iaf.get_continue_bcs(n=2)
    sim = bluepy.Simulation(bcs[0])
    if not vpm:
        gids = np.sort(np.array(list(sim.get_circuit_target())))
    else:
        gids = np.arange(221042, 221042 + 310)

    spikes_df = df.drop(columns=['time'])
    counts_all = np.zeros((gids.size, n), dtype=np.float64)
    for i in range(n):
        x = spikes_df[spikes_df['trial'] == i]
        counts_all[:, i] = np.bincount(np.array(x.index).astype(int), minlength=gids.max()+1)[-gids.size:]
    means = counts_all.mean(axis=1)
    vars = np.var(counts_all, ddof=1, axis=1)
    ffs = vars/means
    ffs[means == 0] = 0
    #print means
    df_counts = pd.DataFrame({'mean':means, 'variance':vars, 'ff':ffs}, index=gids)
    return df_counts


def get_spike_counts_vpm(t_start=0, t_end=7000):
    df = get_all_input_spikes_exp_25()
    df = df[df['time'] >= t_start]
    df = df[df['time'] < t_end]
    return compute_spike_counts(df, n=30, vpm=True)


def get_spike_counts_spontaneous(t_start=0, t_end=2000):
    df = get_spike_times_spontaneous()
    df = df[df['time'] >= t_start]
    df = df[df['time'] < t_end]
    return compute_spike_counts(df, n=40)


def get_spike_counts_spontaneous_binned(bins):
    # averaging over variances. not good.
    df_counts_list = [get_spike_counts_spontaneous(t_start=bins[j], t_end=bins[j + 1]) for j in
                      range(bins.size - 1)]
    df_concat = pd.concat(df_counts_list)
    by_row_index = df_concat.groupby(df_concat.index)
    return by_row_index.mean()


def get_spike_counts_spontaneous_binned_concatenated(bins):
    # computing variance over all bins
    df = get_spike_times_spontaneous()
    dfs = []
    for j in range(bins.size - 1):
        print j
        t_start = bins[j]
        t_end = bins[j + 1]
        dfx = df[df['time'] >= t_start]
        dfx = dfx[dfx['time'] < t_end]
        dfx['trial'] += j * 40
        dfs.append(dfx)
    df = pd.concat(dfs)
    return compute_spike_counts(df, t_end - t_start, n=40*(bins.size -1))


def get_spike_counts_evoked(t_start=0, t_end=7000, variable=False):
    df = get_spike_times_experiment_25(variable=variable)
    df = df[df['time'] >= t_start]
    df = df[df['time'] < t_end]
    return compute_spike_counts(df, n=30)


def plot_count_variance():
    """

    :return:
    """
    circuit = bluepy.Simulation(iaf.get_continue_bcs(n=2)[0]).circuit
    df_neurons = circuit.v2.cells({Cell.HYPERCOLUMN: 2})

    fig, axs = plt.subplots(6, 4, figsize=(10, 12))
    mean_fr = []

    for i, n in enumerate([40, 20, 10, 4, 2, 1]):
        df_counts = get_spike_counts_spontaneous_binned_concatenated(np.linspace(0, 2000, n+1))

        df_counts_exc = df_counts[df_neurons['synapse_class'] == 'EXC']
        df_counts_inh = df_counts[df_neurons['synapse_class'] == 'INH']
        mean_fr.append(df_counts['mean'].loc[84815])
        ax = axs[0 + i, 0]
        ax.scatter(df_counts_exc['mean'], df_counts_exc['variance'], marker='.', color='red', s=3, rasterized=True)

        xs = np.linspace(0, df_counts_exc['mean'].max(), 1000)
        ax.plot(xs, xs, '--', color='black')
        ax.plot(xs, (xs % 1) * (1 - (xs % 1)), '-', color='black')
        ax.set_ylim([0, xs[-1]])

        ax = axs[0 + i, 2]
        ax.hist(df_counts_exc['ff'], bins=np.linspace(0, 2, 20), histtype='step', color='red')

        #ax.set_xlim([0, 10])

        ax = axs[0 + i, 1]
        ax.scatter(df_counts_inh['mean'], df_counts_inh['variance'], marker='.', color='blue', s=3, rasterized=True)
        xs = np.linspace(0, df_counts_inh['mean'].max(), 1000)
        ax.plot(xs, xs, '--', color='black')
        ax.plot(xs, (xs % 1) * (1 - (xs % 1)), '-', color='black')
        ax.set_ylim([0, xs[-1]])

        ax = axs[0 + i, 3]
        ax.hist(df_counts_inh['ff'], bins=np.linspace(0, 2, 20), histtype='step', color='blue')
        #ax.set_xlim([0, 80])
    print mean_fr
    for ax in axs[:, :2].flatten():
        ax.set_xlabel('Spike count')
        ax.set_ylabel('Variance')
    for ax in axs[:, 2:].flatten():
        ax.set_xlabel('Fano factor')
        ax.set_ylabel('Neurons')
    for k, ax in enumerate(axs[:, 0]):
        ax.set_title(['50 ms', '100 ms', '200 ms', '500 ms', '1000 ms', '2000 ms'][k])
    plt.tight_layout()
    plt.savefig('figures_poisson/spont_variance_poisson.pdf', dpi=300)

def plot_count_variance_exp_25_summary():
    bins = np.linspace(1000, 7000, 13)

    print bins
    circuit = bluepy.Simulation(iaf.get_continue_bcs(n=2)[0]).circuit
    # df_neurons = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    df_neurons = get_selected_L456_gids()

    fig, axs = plt.subplots(5, len(bins) - 1, figsize=(20, 6))
    for k, variable in enumerate([False, True]):
        df_counts_list = [get_spike_counts_evoked(t_start=bins[j], t_end=bins[j + 1],
                                                  variable=variable) for j in range(len(bins) - 1)]

        df_counts_exc_list = []
        for i, df_counts in enumerate(df_counts_list):
            df_counts_exc = df_counts.loc[df_neurons.index.unique()][df_neurons['synapse_class'] == 'EXC']
            df_counts_exc_list.append(df_counts_exc)
            df_counts_inh = df_counts.loc[df_neurons.index.unique()][df_neurons['synapse_class'] == 'INH']

            ax = axs[0 + k * 2, i]
            ax.scatter(df_counts_exc['mean'], df_counts_exc['variance'], marker='.', color='red', s=3, rasterized=True)
            xs = np.linspace(0, df_counts_exc['mean'].max(), 1000)
            ax.plot(xs, xs, '--', color='black')
            ax.plot(xs, (xs % 1) * (1 - (xs % 1)), '-', color='black')
            ax.set_xlim([0, xs.max()])
            ax.set_ylim([0, xs.max()])

            ax = axs[1 + k * 2, i]
            ax.scatter(df_counts_inh['mean'], df_counts_inh['variance'], marker='.', color='blue', s=3, rasterized=True)
            xs = np.linspace(0, df_counts_inh['mean'].max(), 1000)
            ax.plot(xs, xs, '--', color='black')
            ax.plot(xs, (xs % 1) * (1 - (xs % 1)), '-', color='black')
            ax.set_xlim([0, xs.max()])
            ax.set_ylim([0, xs.max()])
    df_counts_list_vpm = [get_spike_counts_vpm(t_start=bins[j], t_end=bins[j + 1]) for j in range(2, len(bins) - 1)]
    for i, df_counts in enumerate(df_counts_list_vpm):

        ax = axs[-1, i+2]
        ax.scatter(df_counts['mean'], df_counts['variance'], marker='.', color='green', s=3, rasterized=True)
        xs = np.linspace(0, df_counts['mean'].max(), 1000)
        ax.plot(xs, xs, '--', color='black')
        ax.plot(xs, (xs % 1) * (1 - (xs % 1)), '-', color='black')
        ax.set_xlim([0, xs.max()])
        ax.set_ylim([0, xs.max()])
    plt.savefig('figures_poisson/exp_25_variance_poisson_summary.pdf', dpi=300)


def plot_psths():
    df_1 = get_spike_times_experiment_25(variable=False)
    df_neurons = get_selected_L456_gids()
    df_1 = df_1.loc[df_neurons.index.unique()]
    fig, axs = plt.subplots(2, 1, figsize=(20, 4))
    ax = axs[0]
    for i in range(30):
        ax.hist(df_1[df_1['trial'] == i]['time'], bins=np.arange(1000, 2020, 20), histtype='step')

    df_2 = get_spike_times_experiment_25(variable=True)
    df_2 = df_2.loc[df_neurons.index.unique()]

    ax = axs[1]
    for i in range(30):
        ax.hist(df_2[df_2['trial'] == i]['time'], bins=np.arange(1000, 2020, 20), histtype='step')
    plt.savefig('figures_poisson/exp_25_hist_variable.pdf', dpi=300)
    return df_1, df_2


def plot_ei_balance():
    df = get_spike_times_experiment_25(variable=False)
    print np.isnan(df.time.values).sum()
    # circuit = bluepy.Simulation(get_configs_network()[0]).circuit
    # df_neurons_other = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    df_neurons = get_selected_L456_gids()
    df_neurons = df_neurons.loc[np.unique(df.index)]
    exc_index = df_neurons[df_neurons['synapse_class'] == 'EXC'].index
    inh_index = df_neurons[df_neurons['synapse_class'] == 'INH'].index
    print inh_index
    print exc_index
    print df.loc[inh_index].time.values
    print np.isnan(df.loc[inh_index].time.values).sum()


    fig, ax = plt.subplots(figsize=(100, 4))
    bins = np.arange(1980, 6185, 5)
    ax.hist(df.loc[exc_index].time.values, bins=bins, normed=True, histtype='step', color='red')
    ax.hist(df.loc[inh_index].time.values, bins=bins, normed=True, histtype='step', color='blue')
    plt.savefig('figures_poisson/spikes_ei.pdf')


def plot_input_spikes():

    df = get_all_input_spikes_exp_25()
    print "---"
    print df
    min_vpm = df.index.min()
    print min_vpm

    fig, axs = plt.subplots(5)

    for i in range(5):
        sub_df = df[df['trial'] == i]
        spike_times = np.array(sub_df['time'])
        spike_gids = np.array(sub_df.index)
        ax = axs[i]
        ax.vlines(spike_times, spike_gids - min_vpm, spike_gids - min_vpm + 1, rasterized=True, lw=1)
        ax2 = ax.twinx()
        ax2.hist(spike_times, bins=np.linspace(2000, 2500, 201), histtype='step',
                     weights=np.zeros(spike_times.size) + (200 / 5.0) / 310)
        ax2.set_ylabel('FR (Hz)')
        ax.set_ylabel('Neurons')
        ax.set_xlabel('t (ms)')
        ax.set_ylim([0, 310])
        ax.set_xlim([2000, 2500])
    plt.savefig('figures_poisson/vpm_raster.pdf')


    fig, axs = plt.subplots(20, figsize=(10, 30))

    for i in range(20):
        print min_vpm + i
        sub_df = df.loc[min_vpm + i]
        spike_times = np.array(sub_df['time'])
        spike_trials = np.array(sub_df['trial'])
        ax = axs[i]
        ax.vlines(spike_times, spike_trials, spike_trials+1, rasterized=True, lw=1)
        # ax2 = ax.twinx()
        # ax2.hist(spike_times, bins=np.linspace(2000, 7000, 1001), histtype='step',
        #              weights=np.zeros(spike_times.size) + (1000 / 5.0) / 310)
        # ax2.set_ylabel('FR (Hz)')
        ax.set_ylabel('Trials')
        ax.set_xlabel('t (ms)')
        ax.set_ylim([0, 30])
        ax.set_xlim([2000, 2500])
    plt.savefig('figures_poisson/vpm_raster_2.pdf', dpi=300)


if __name__ == "__main__":
    plot_count_variance()
    plot_count_variance_exp_25_summary()
    plot_input_spikes()
    plot_count_variance_exp_25_summary()
    df_1, df_2 = plot_psths()
    data = get_all_input_spikes_exp_25()
