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
import scipy

cas = ['1p05',  '1p1', '1p15', '1p2', '1p25', '1p3', '1p35', '1p4', '1p45', '1p5']


def plot_reliabilities_exp_25_only_normal():

    spike_count_dict = data_access_shuffling.get_all_spike_counts_experiment_25()
    df_neurons = data_access_shuffling.get_selected_L456_gids()
    df_neurons_m2 = data_access_shuffling.get_mc_2_gids()

    rel_dict = data_access_shuffling.get_all_reliabilities_experiment_25()
    sim_type = 'control'

    cas_numeric = np.array([float(x.replace('p', '.')) for x in data_access_shuffling.cas_exp_25[sim_type]])
    means = np.zeros((cas_numeric.size, 6))
    errs = np.zeros((cas_numeric.size, 6))
    for j, ca in enumerate(data_access_shuffling.cas_exp_25[sim_type]):
        key = sim_type + '_' + ca

        # Computing reliabilities
        rel_df = rel_dict[key]
        values_exc = rel_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'EXC'].index)]
        values_inh = rel_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'INH'].index)]
        means[j, 0] = values_exc.mean().mean()
        errs[j, 0] = mean_confidence_interval(values_exc.mean())
        means[j, 1] = values_inh.mean().mean()
        errs[j, 1] = mean_confidence_interval(values_inh.mean())

        # Computing firing rates for neurons used for reliable computation
        spike_df = spike_count_dict[key]
        values_exc = spike_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'EXC'].index)]
        values_inh = spike_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'INH'].index)]
        means[j, 2] = values_exc['mean'].sum()/5.0
        errs[j, 2] = np.sqrt(values_exc['variance'].sum())/5.0
        means[j, 3] = values_inh['mean'].sum()/5.0
        errs[j, 3] = np.sqrt(values_inh['mean'].sum())/5.0

        # # Computing firing rates for all neurons
        values_exc = spike_df.loc[df_neurons_m2[df_neurons_m2['synapse_class'] == 'EXC'].index]
        values_inh = spike_df.loc[df_neurons_m2[df_neurons_m2['synapse_class'] == 'INH'].index]
        means[j, 4] = values_exc['mean'].sum()/5.0
        errs[j, 4] = np.sqrt(values_exc['variance'].sum())/5.0
        means[j, 5] = values_inh['mean'].sum()/5.0
        errs[j, 5] = np.sqrt(values_inh['mean'].sum())/5.0

    fig, axs = plt.subplots(2, 2)
    means = means[1:, :]
    errs = errs[1:, :]
    cas_numeric = cas_numeric[1:]

    ax = axs[0, 0]
    ax.errorbar(cas_numeric, means[:, 0], yerr=errs[:, 0], color = 'red', marker='o', markersize=4)
    ax.errorbar(cas_numeric, means[:, 1], yerr=errs[:, 1], color = 'blue', marker='o', markersize=4)
    ax.set_ylim([0, 0.5])
    ax.set_ylabel('r_spike')

    ax = axs[1, 0]
    ax.errorbar(cas_numeric, means[:, 2]/means[:, 3], color = 'black', marker='^', markersize=4)
    ax.errorbar(cas_numeric, means[:, 4]/means[:, 5], color = 'black', marker='d', markersize=4, linestyle='--')
    ax.set_ylabel('E/I-balance')


    ax = axs[0, 1]
    ax.errorbar(cas_numeric, means[:, 2], yerr=errs[:, 2], color = 'red', marker='^', markersize=4)
    ax.errorbar(cas_numeric, means[:, 3], yerr=errs[:, 3], color = 'blue', marker='^', markersize=4)
    ax.set_ylabel('Pop. FR (Hz)')


    ax = axs[1, 1]
    ax.errorbar(cas_numeric, means[:, 4], yerr=errs[:, 4], color = 'red', marker='d', markersize=4, linestyle='--')
    ax.errorbar(cas_numeric, means[:, 5], yerr=errs[:, 5], color = 'blue', marker='d', markersize=4, linestyle='--')
    ax.set_ylabel('Pop. FR (Hz)')


    for ax in axs.flatten():
        ax.set_xticks(np.arange(1.1, 1.4, 0.05))
        ax.set_xlabel('[Ca2+] (mM)')

    plt.savefig('figures/rels_criticality.pdf')
    for key in rel_dict.keys():
        print key
        print np.isnan(rel_dict[key].sum()).sum()
        print np.shape(rel_dict[key])
    return rel_dict[rel_dict.keys()[0]]


def example_raster_plot():
    l4 = [71211, 70922, 71042,  71338, 74266, 74419, 74882, 71610,]
    l5 = [75998, 81442, 82079, 93653, 93902, 81703, 81443, 82202]
    l6 = [93655, 75759 ,76251, 80451, 80762, 81068, 81255, 81316]

    gids_to_plot = [74419, 81442, 93272] #81068] #[l4[n], l5[n], l6[n]]75270
    # gids_to_plot = [85422, 90721]
    fig, axs = plt.subplots(3, 3, figsize=(14, 7))

    cas = ['1p15', '1p25', '1p35']

    for i, ca in enumerate(cas):
        bcs = data_access_shuffling.get_exp_25_blueconfigs_cloud(n=30, ca=ca, sim_type='control')
        df_network = data_access_shuffling.get_spike_times_multiple(bcs).loc[gids_to_plot]

        t_start = 2500
        t_end = 4500
        df_network = df_network[(df_network['spike_time'] < t_end) & (df_network['spike_time'] >= t_start)]

        for k, gid in enumerate(gids_to_plot):
            print k
            axs[i, k].fill_between([t_start, t_end], [0, 0], [30, 30], color='red', alpha=0.1, linewidth=0.0)
            plot_times = df_network.loc[gid]['spike_time']
            plot_trials = df_network.loc[gid]['spike_trial']
            axs[i, k].vlines(plot_times, plot_trials, plot_trials + 1, linewidth=0.5)


    for ax in axs[:, :].flatten():
        ax.set_ylim([0, 30])
        ax.set_yticks([0.5, 4.5, 9.5, 14.5, 19.5, 25.5, 29.5])
        ax.set_yticklabels([1, 5, 10, 15, 20, 25, 30])
        ax.set_ylabel('Trials')

    for ax in axs.flatten():
        ax.set_xlim([t_start, t_end])
        ax.set_xlabel('t (ms)')

    for i in range(3):
        axs[0, i].set_title(gids_to_plot[i])
    plt.tight_layout()
    plt.savefig('figures/rasterplot_example.pdf')


def plot_reliabilities_exp_25_shuffle_comparison():
    spike_count_dict = data_access_shuffling.get_all_spike_counts_experiment_25()
    df_neurons = data_access_shuffling.get_selected_L456_gids()
    df_neurons_m2 = data_access_shuffling.get_mc_2_gids()

    rel_dict = data_access_shuffling.get_all_reliabilities_experiment_25()
    means_dict = {}
    errs_dict = {}
    cas_dict = {}
    for sim_type in ['cloud_synapse_type', 'cloud_mtype', 'control', 'cloud_mtype_exc']:

        cas_numeric = np.array([float(x.replace('p', '.')) for x in data_access_shuffling.cas_exp_25[sim_type]])
        means = np.zeros((cas_numeric.size, 6))
        errs = np.zeros((cas_numeric.size, 6))
        for j, ca in enumerate(data_access_shuffling.cas_exp_25[sim_type]):
            key = sim_type + '_' + ca

            # Computing reliabilities
            rel_df = rel_dict[key]
            values_exc = rel_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'EXC'].index)]
            values_inh = rel_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'INH'].index)]
            means[j, 0] = values_exc.mean().mean()
            errs[j, 0] = mean_confidence_interval(values_exc.mean())
            means[j, 1] = values_inh.mean().mean()
            errs[j, 1] = mean_confidence_interval(values_inh.mean())

            # Computing firing rates for neurons used for reliable computation
            spike_df = spike_count_dict[key]
            values_exc = spike_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'EXC'].index)]
            values_inh = spike_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'INH'].index)]
            means[j, 2] = values_exc['mean'].sum() / 5.0
            errs[j, 2] = np.sqrt(values_exc['variance'].sum()) / 5.0
            means[j, 3] = values_inh['mean'].sum() / 5.0
            errs[j, 3] = np.sqrt(values_inh['mean'].sum()) / 5.0

            # # Computing firing rates for all neurons
            values_exc = spike_df.loc[df_neurons_m2[df_neurons_m2['synapse_class'] == 'EXC'].index]
            values_inh = spike_df.loc[df_neurons_m2[df_neurons_m2['synapse_class'] == 'INH'].index]
            means[j, 4] = values_exc['mean'].sum() / 5.0
            errs[j, 4] = np.sqrt(values_exc['variance'].sum()) / 5.0
            means[j, 5] = values_inh['mean'].sum() / 5.0
            errs[j, 5] = np.sqrt(values_inh['mean'].sum()) / 5.0

        means_dict[sim_type] = means
        errs_dict[sim_type] = errs
        cas_dict[sim_type] = cas_numeric


        fig, axs = plt.subplots(2, 2)
        n_start = 0
        means = means[n_start:, :]
        errs = errs[n_start:, :]
        cas_numeric = cas_numeric[n_start:]

        ax = axs[0, 0]
        ax.errorbar(cas_numeric, means[:, 0], yerr=errs[:, 0], color='red', marker='o', markersize=4)
        ax.errorbar(cas_numeric, means[:, 1], yerr=errs[:, 1], color='blue', marker='o', markersize=4)
        #ax.set_ylim([0, 0.5])
        ax.set_ylabel('r_spike')

        ax = axs[1, 0]
        ax.errorbar(cas_numeric, means[:, 2] / means[:, 3], color='black', marker='^', markersize=4)
        ax.errorbar(cas_numeric, means[:, 4] / means[:, 5], color='black', marker='d', markersize=4, linestyle='--')
        ax.set_ylabel('E/I-balance')

        ax = axs[0, 1]
        ax.errorbar(cas_numeric, means[:, 2], yerr=errs[:, 2], color='red', marker='^', markersize=4)
        ax.errorbar(cas_numeric, means[:, 3], yerr=errs[:, 3], color='blue', marker='^', markersize=4)
        ax.set_ylabel('Pop. FR (Hz)')

        ax = axs[1, 1]
        ax.errorbar(cas_numeric, means[:, 4], yerr=errs[:, 4], color='red', marker='d', markersize=4, linestyle='--')
        ax.errorbar(cas_numeric, means[:, 5], yerr=errs[:, 5], color='blue', marker='d', markersize=4, linestyle='--')
        ax.set_ylabel('Pop. FR (Hz)')

        for ax in axs.flatten():
            ax.set_xticks(np.arange(1.05, 1.4, 0.05))
            ax.set_xlabel('[Ca2+] (mM)')

        plt.savefig('figures/rels_criticality_%s.pdf' % sim_type)

    fig, axs = plt.subplots(2, 2)
    for sim_type in ['cloud_synapse_type', 'cloud_mtype', 'control', 'cloud_mtype_exc']:
        cas_numeric = cas_dict[sim_type]
        means = means_dict[sim_type]
        errs = errs_dict[sim_type]
        ax = axs[0, 0]
        ax.errorbar(cas_numeric, means[:, 0], yerr=errs[:, 0], marker='.', markersize=4, label=sim_type)
    ax.legend(prop={'size': 5})
    for ax in axs.flatten():
        ax.set_xticks(np.arange(1.05, 1.4, 0.05))
        ax.set_xlabel('[Ca2+] (mM)')
    plt.savefig('figures/rels_criticality_comparison.pdf')
    

def plot_reliabilities_jitter():

    spike_count_dict = data_access_shuffling.get_all_spike_counts_jitter()
    df_neurons = data_access_shuffling.get_selected_L456_gids()
    df_neurons_m2 = data_access_shuffling.get_mc_2_gids()

    rel_dict = data_access_shuffling.get_all_reliabilities_jitter()

    ids = data_access_shuffling.ids_jitter[[5, 0, 2, 4, 1, 3]]
    jitters = np.array([2, 50, 5, 200, 20, 0])[[5, 0, 2, 4, 1, 3]]


    means = np.zeros((ids.size, 6))
    errs = np.zeros((ids.size, 6))
    fig, axs = plt.subplots(2)
    for j, id in enumerate(ids):
        key = id
        # Computing reliabilities
        rel_df = rel_dict[key]
        values_exc = rel_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'EXC'].index)]
        values_inh = rel_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'INH'].index)]
        axs[0].hist(np.mean(values_exc, axis=1), bins=np.linspace(0, 1, 21), histtype='step')
        means[j, 0] = values_exc.mean().mean()
        errs[j, 0] = mean_confidence_interval(values_exc.mean())
        means[j, 1] = values_inh.mean().mean()
        errs[j, 1] = mean_confidence_interval(values_inh.mean())

        # Computing firing rates for neurons used for reliable computation
        spike_df = spike_count_dict[key]
        values_exc = spike_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'EXC'].index)]
        values_inh = spike_df.loc[np.intersect1d(rel_df.index, df_neurons[df_neurons['synapse_class'] == 'INH'].index)]
        means[j, 2] = values_exc['mean'].sum()/5.0
        errs[j, 2] = np.sqrt(values_exc['variance'].sum())/5.0
        means[j, 3] = values_inh['mean'].sum()/5.0
        errs[j, 3] = np.sqrt(values_inh['mean'].sum())/5.0

        axs[1].hist(values_exc['mean']/5.0, bins=np.linspace(0, 10, 21), histtype='step')

        # # Computing firing rates for all neurons
        values_exc = spike_df.loc[df_neurons_m2[df_neurons_m2['synapse_class'] == 'EXC'].index]
        values_inh = spike_df.loc[df_neurons_m2[df_neurons_m2['synapse_class'] == 'INH'].index]
        means[j, 4] = values_exc['mean'].sum()/5.0
        errs[j, 4] = np.sqrt(values_exc['variance'].sum())/5.0
        means[j, 5] = values_inh['mean'].sum()/5.0
        errs[j, 5] = np.sqrt(values_inh['mean'].sum())/5.0
    plt.savefig('figures/jitter_hists.pdf')

    fig, axs = plt.subplots(2, 2)

    ax = axs[0, 0]
    ax.errorbar(np.arange(6), means[:, 0], yerr=errs[:, 0], color = 'red', marker='o', markersize=4)
    ax.errorbar(np.arange(6), means[:, 1], yerr=errs[:, 1], color = 'blue', marker='o', markersize=4)
    #ax.set_ylim([0, 0.5])
    ax.set_ylabel('r_spike')

    ax = axs[1, 0]
    ax.errorbar(np.arange(6), means[:, 2]/means[:, 3], color = 'black', marker='^', markersize=4)
    ax.errorbar(np.arange(6), means[:, 4]/means[:, 5], color = 'black', marker='d', markersize=4, linestyle='--')
    ax.set_ylabel('E/I-balance')


    ax = axs[0, 1]
    ax.errorbar(np.arange(6), means[:, 2], yerr=errs[:, 2], color = 'red', marker='^', markersize=4)
    ax.errorbar(np.arange(6), means[:, 3], yerr=errs[:, 3], color = 'blue', marker='^', markersize=4)
    ax.set_ylabel('Pop. FR (Hz)')


    ax = axs[1, 1]
    ax.errorbar(np.arange(6), means[:, 4], yerr=errs[:, 4], color = 'red', marker='d', markersize=4, linestyle='--')
    ax.errorbar(np.arange(6), means[:, 5], yerr=errs[:, 5], color = 'blue', marker='d', markersize=4, linestyle='--')
    ax.set_ylabel('Pop. FR (Hz)')


    for ax in axs.flatten():
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(jitters)
        ax.set_xlabel('jitter')

    plt.savefig('figures/rels_jitter.pdf')


def example_raster_plot_jitter():
    l4 = [71211, 70922, 71042,  71338, 74266, 74419, 74882, 71610,]
    l5 = [75998, 81442, 82079, 93653, 93902, 81703, 81443, 82202]
    l6 = [93655, 75759 ,76251, 80451, 80762, 81068, 81255, 81316]

    gids_to_plot = [74419, 81442, 93272] #81068] #[l4[n], l5[n], l6[n]]75270
    # gids_to_plot = [85422, 90721]
    fig, axs = plt.subplots(6, 3, figsize=(14, 14))

    ids = data_access_shuffling.ids_jitter[[5, 0, 2, 4, 1, 3]]
    jitters = np.array([2, 50, 5, 200, 20, 0])[[5, 0, 2, 4, 1, 3]]

    for i, id in enumerate(ids):
        bcs = data_access_shuffling.get_jitter_blueconfigs(n=30, id_jitter=id)
        df_network = data_access_shuffling.get_spike_times_multiple(bcs).loc[gids_to_plot]

        t_start = 2500
        t_end = 4500
        df_network = df_network[(df_network['spike_time'] < t_end) & (df_network['spike_time'] >= t_start)]

        for k, gid in enumerate(gids_to_plot):
            print k
            axs[i, k].fill_between([t_start, t_end], [0, 0], [30, 30], color='red', alpha=0.1, linewidth=0.0)
            plot_times = df_network.loc[gid]['spike_time']
            plot_trials = df_network.loc[gid]['spike_trial']
            axs[i, k].vlines(plot_times, plot_trials, plot_trials + 1, linewidth=0.5)

    for ax in axs[:, :].flatten():
        ax.set_ylim([0, 30])
        ax.set_yticks([0.5, 4.5, 9.5, 14.5, 19.5, 25.5, 29.5])
        ax.set_yticklabels([1, 5, 10, 15, 20, 25, 30])
        ax.set_ylabel('Trials')

    for ax in axs.flatten():
        ax.set_xlim([t_start, t_end])
        ax.set_xlabel('t (ms)')

    for i in range(3):
        axs[0, i].set_title(gids_to_plot[i])
    plt.tight_layout()
    plt.savefig('figures/rasterplot_example_jitter.pdf')


def populations_psths_jitter():
    df_neurons = data_access_shuffling.get_selected_L456_gids()

    gids = np.array(df_neurons.index)
    print gids.shape

    # spike_count_dict = data_access_shuffling.get_all_spike_counts_jitter()

    fig, axs = plt.subplots(6, figsize=(14, 14))
    ids = data_access_shuffling.ids_jitter[[5, 0, 2, 4, 1, 3]]
    jitters = np.array([2, 50, 5, 200, 20, 0])[[5, 0, 2, 4, 1, 3]]

    for i, id in enumerate(ids):
        bcs = data_access_shuffling.get_jitter_blueconfigs(n=30, id_jitter=id)
        df_network = data_access_shuffling.get_spike_times_multiple(bcs).loc[gids]
        print df_network
        t_start = 1000
        t_end = 7000
        df_network = df_network[(df_network['spike_time'] < t_end) & (df_network['spike_time'] >= t_start)]

        for k in range(30):
            print k
            times = df_network[df_network['spike_trial'] == k]['spike_time']
            axs[i].hist(times, bins=np.arange(t_start, t_end + 5, 5), color='red', histtype='step', alpha=0.5)

    for ax in axs.flatten():
        ax.set_xlim([t_start, t_end])
        ax.set_xlabel('t (ms)')

    plt.tight_layout()
    plt.savefig('figures/population_psths_jitter.pdf')


if __name__ == "__main__":
    plot_reliabilities_exp_25_only_normal()
    example_raster_plot()
    plot_reliabilities_exp_25_shuffle_comparison()
    example_raster_plot_jitter()
    populations_psths_jitter()




