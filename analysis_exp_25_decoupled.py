import correlations
import bluepy
from bluepy.v2 import Cell
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import magicspike.distances as distances
import os
from itertools import combinations
import gc
import initial_analysis_final as iaf
from scipy import stats
from connection_matrices.generate.gen_con_mats import connections


# SIMS part I ----------------------
# Original simulations: normal with VPM stimulus (30), and the replayed versions of those sims (30 x 5)
network_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30/seed%d/BlueConfig'
decoupled_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_replays_fixed/replay_seed%d/original_seed%d/BlueConfig'
network_seeds = np.arange(100, 130, dtype=np.int)
decoupled_seeds = np.arange(1, 6, dtype=np.int)

# Short simulations of spontaneous activity (40)
spont_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_base_seeds/seed%d/BlueConfig'
spont_seeds = np.arange(170, 210, dtype=np.int)

# Same as original simulations above, but only with ab noise sources and only one decoupled replay per simulation(30, 30 x 1)
network_path_ab = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_ab/seed%d/BlueConfig'
decoupled_path_ab = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_replays_fixed_ab/replay_seed%d/original_seed%d/BlueConfig'
network_seeds_ab = np.arange(70, 100, dtype=np.int)
decoupled_seeds_ab = np.arange(1, 6, dtype=np.int)


# SIMS part II ----------------------
# Same VPM input as above, but replayed spontaneous activity from below, Ca for VPM synapses set to 2p0 (30, 30)
decoupled_path_vpm = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_replays_fixed_spontaneous/replay_seed1/original_seed%d/BlueConfig'
decoupled_path_vpm_ca2p0 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_replays_fixed_spontaneous_vpmca2p0/replay_seed1/original_seed%d/BlueConfig'
network_seeds_vpm = np.arange(130, 159, dtype=np.int)
network_seeds_vpm_ca2p0 = np.arange(160, 189, dtype=np.int)
decoupled_seeds_vpm = [1]


# SIMS part III -------------------
# variable input
variable_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/experiment_25/new_exp_25_id0_n30_variable/seed%d/BlueConfig'
variable_seeds = np.arange(200, 230, dtype=np.int)


# Long simulations of spontaneous activity replayed in examples above (30)
spont_path_long = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_abcd_long/seed%d/BlueConfig'
spont_seeds_long = np.arange(50, 80, dtype=np.int)


def get_config_pairs_network():
    bcs = get_configs_network()
    bc_combs = list(combinations(bcs, 2))
    print len(bc_combs)
    bcs_1 = []
    bcs_2 = []
    for pair in bc_combs:
        bcs_1.append(pair[0])
        bcs_2.append(pair[1])
    return bcs_1, bcs_2


def get_configs_network():
    bcs_1 = [network_path % s for s in network_seeds]
    return bcs_1

def get_configs_variable():
    bcs_1 = [variable_path % s for s in variable_seeds]
    return bcs_1

def get_configs_network_ab():
    bcs_1 = [network_path_ab % s for s in network_seeds_ab]
    return bcs_1


def get_configs_network_nr_vpmca2p0():
    bcs_1 = [decoupled_path_vpm_ca2p0 % s for s in network_seeds_vpm_ca2p0]
    return bcs_1


def get_configs_network_nr():
    bcs_1 = [decoupled_path_vpm % s for s in network_seeds_vpm]
    return bcs_1


def get_config_pairs_decoupled():
    bcs_1 = []
    bcs_2 = []
    for ns in network_seeds:
        bcs = [decoupled_path % (ds, ns) for ds in decoupled_seeds]
        bc_combs = list(combinations(bcs, 2))
        for pair in bc_combs:
            bcs_1.append(pair[0])
            bcs_2.append(pair[1])
    return bcs_1, bcs_2


def get_configs_decoupled():
    bcs = []
    for ns in network_seeds:
        bcs_seed = []
        for decoupled_seed in decoupled_seeds:
            bcs_seed.append(decoupled_path % (decoupled_seed, ns))
        bcs.append(bcs_seed)
    return bcs

def get_configs_decoupled_ab():
    bcs = []
    for ns in network_seeds_ab:
        bcs_seed = []
        for decoupled_seed in decoupled_seeds_ab:
            bcs_seed.append(decoupled_path_ab % (decoupled_seed, ns))
        bcs.append(bcs_seed)
    return bcs


def get_correlations(dt=10.0, decouple=False):
    t_start = 1000
    t_end = 6500
    folder = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations_stim' + '/corrs_exp_25_all'

    file = folder + '_dt%d' % dt
    if decouple:
        file += '_decouple'

    file += '.npz'
    if not os.path.isfile(file):

        if not decouple:
            bcs_1, bcs_2 = get_config_pairs_network()
        if decouple:
            bcs_1, bcs_2 = get_config_pairs_decoupled()

        corrs, bins = compute_soma_correlations(bcs_1, bcs_2, dt=dt, t_start=t_start, t_end=t_end)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def get_soma_time_series(blueconfig, t_start=None, t_end=None, gids=None):
    soma = bluepy.Simulation(blueconfig).v2.reports['soma']
    data = soma.data(t_start=t_start, t_end=t_end, gids=gids) #, gids=np.arange(72788, 72800)) #
    return data, data.axes[1]/1000.0


def compute_soma_correlations(bcs_1, bcs_2, dt=5.0, t_start=1000, t_end=6500.0):
    """
    :param parameter_continue:
    :param parameter_change:
    :return:
    """
    for j, (bc_1, bc_2) in enumerate(zip(bcs_1, bcs_2)):
        print j
        print bc_1
        print bc_2
        vm_continue, _ = get_soma_time_series(bc_1, t_start=t_start, t_end=t_end)
        vm_change, times = get_soma_time_series(bc_2, t_start=t_start, t_end=t_end)
        vm_continue = np.array(vm_continue)
        vm_change = np.array(vm_change)

        for i, corr_func in enumerate([correlations.voltage_rmsd_from_data, correlations.voltage_correlation_from_data]):
            corr, bins = corr_func(vm_continue, vm_change, times, dt=dt)
            if j == 0:
                corrs_all = np.zeros(corr.shape + (2, len(bcs_1)), dtype=np.float32)
            corrs_all[:, :, i, j] = corr
            print corrs_all.shape
    return corrs_all, bins


def plot_correlations():
    index_correlation = 1
    corrs, bins = get_correlations(dt=10.0, decouple=False)
    print corrs.shape
    means_nw = corrs[:, :, index_correlation, :].mean(axis=(-1))
    mean_nw = corrs[:, :, index_correlation, :].mean(axis=(0, -1))
    errs_nw = corrs[:, :, index_correlation, :].std(axis=-1)/np.sqrt(435)
    err_nw = corrs[:, :, index_correlation, :].mean(axis=0).std(axis=-1)/np.sqrt(435)

    corrs = 0
    gc.collect()
    print "network loaded"

    corrs, bins = get_correlations(dt=10.0, decouple=True)
    print corrs.shape
    means_dc = corrs[:, :, index_correlation, :].mean(axis=(-1))
    mean_dc = corrs[:, :, index_correlation, :].mean(axis=(0, -1))
    errs_dc = corrs[:, :, index_correlation, :].std(axis=-1)/np.sqrt(300)
    err_dc = corrs[:, :, index_correlation, :].mean(axis=0).std(axis=-1)/np.sqrt(300)

    corrs = 0
    gc.collect()
    print "decoupled loaded"

    t_start = 10
    t_end = 110

    fig, ax = plt.subplots(figsize=(12, 4))

    for i, (mean, err) in enumerate(zip([mean_nw, mean_dc], [err_nw, err_dc])):
        ax.fill_between(bins[t_start:t_end], mean[t_start:t_end] - err[t_start:t_end],
                                   mean[t_start:t_end] + err[t_start:t_end],
                                    linewidth=0.8,  color=['#d9d9d9', '#a6bddb'][i])

        ax.plot(bins, mean[t_start:t_end], linewidth=0.8, color=['#737373', '#3690c0'][i], marker='.', ms=2,)
    plt.savefig('figures_stim/correlation.pdf')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(all_means[0], all_means[1], linewidth=0.8)
    plt.savefig('figures_stim/correlation_2.pdf')

    circuit = bluepy.Simulation(get_config_pairs_network()[0][0]).circuit
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    gids_all = np.array(cells.axes[0])

    gids_to_plot = [70558, 83721]
    print gids_all.min()

    fig, axs = plt.subplots(2, 1, figsize=(12, 4))
    for i, corrs in enumerate([corrs_network, corrs_dec]):
        for j, gid in enumerate(gids_to_plot):
            time = bins[40:90]
            means = corrs[gid - gids_all.min(), 40:90, index_correlation, :].mean(axis=-1)
            errs = corrs[gid - gids_all.min(), 40:90, index_correlation, :].std(axis=-1)/np.sqrt(30)
            axs[j].fill_between(time, means - errs,
                                   means + errs,
                                    linewidth=0.8,  color=['#d9d9d9', '#a6bddb'][i])

            axs[j].plot(time, means, linewidth=0.8, color=['#737373', '#3690c0'][i], marker='.', ms=2,)
    plt.savefig('figures_stim/correlation.pdf')


def compute_reliabilities(sigma_conv=5, dt_conv=0.5, decouple=False, spont=False,
                          network_removal=False, vpm_ca2p0=False, ab=False, variable=False):
    # original sigma_conv = 3, dt = 0.5

    t_start = 2000
    t_end = 7000
    if spont:
        t_start = 1500
        t_end = 2000

    df_network = get_spike_times_experiment_25(decouple=decouple, network_removal=network_removal, vpm_ca2p0=vpm_ca2p0, ab=ab, variable=variable)
    df_network = df_network[df_network['time'] >= t_start]
    df_network = df_network[df_network['time'] < t_end]
    gids = np.unique(df_network.index)
    gids = np.intersect1d(get_selected_L456_gids().index, gids)
    print "no of spiking gids: %d" % len(gids)
    print gids
    if not decouple:
        df_results = pd.DataFrame(columns=range(435), index=gids, dtype=np.float32)
        print df_results
        for k, gid in enumerate(gids):
            print k
            spikes = np.array(df_network.loc[[gid]]['time'])
            trials = np.array(df_network.loc[[gid]]['trial'])
            mean, err, _distances = distances.compute_pairwise_distances(spikes, trials, [t_start, t_end], 30, 435, method='schreiber', time_sorted=True,
                                                trimmed=True, combi_seed=0, sigma_conv=sigma_conv, dt_conv=dt_conv)
            df_results.loc[gid] = _distances
    elif decouple:
        df_results = pd.DataFrame(columns=range(300), index=gids, dtype=np.float32)
        print df_results
        for k, gid in enumerate(gids):
            print k

            df_gid = df_network.loc[[gid]]
            for original_trial in range(30):
                df_gid_trial = df_gid[df_gid['original_trial'] == original_trial]
                spikes = np.array(df_gid_trial['time'])
                trials = np.array(df_gid_trial['trial'])
                mean, err, _distances = distances.compute_pairwise_distances(spikes, trials, [t_start, t_end], 5, 10, method='schreiber', time_sorted=True,
                                                trimmed=True, combi_seed=0, sigma_conv=sigma_conv, dt_conv=dt_conv)
                df_results.loc[gid, (original_trial * 10):((original_trial * 10) + 9)] = _distances
    return df_results


def get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1, spont=False,
                                    network_removal=False, vpm_ca2p0=False, ab=False, variable=False):
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    directory = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/'
    s_ab = ''
    if variable:
        spike_file = 'reliabilities_exp_25_n30_id0_variable_%d_%.1f.pkl' % (sigma_conv, dt_conv)
    else:
        if ab:
            s_ab = '_ab'
        if spont:
            if decouple:
                spike_file = 'reliabilities_exp_25_n30_id0_decouple_%d_%.1f_spont%s.pkl' % (sigma_conv, dt_conv, s_ab)
            else:
                spike_file = 'reliabilities_exp_25_n30_id0_%d_%.1f_spont%s.pkl' % (sigma_conv, dt_conv, s_ab)
        else:
            if decouple:
                spike_file = 'reliabilities_exp_25_n30_id0_decouple_%d_%.1f%s.pkl' % (sigma_conv, dt_conv, s_ab)
            else:
                if network_removal:
                    if vpm_ca2p0:
                        spike_file = 'reliabilities_exp_25_n30_id0_nr_vpmca2p0_%d_%.1f.pkl' % (sigma_conv, dt_conv)
                    else:
                        spike_file = 'reliabilities_exp_25_n30_id0_nr_%d_%.1f.pkl' % (sigma_conv, dt_conv)
                else:
                    spike_file = 'reliabilities_exp_25_n30_id0_%d_%.1f%s.pkl' % (sigma_conv, dt_conv, s_ab)

    file_name = os.path.join(directory, spike_file)
    if os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
    else:
        df = compute_reliabilities(decouple=decouple, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=spont,
                                   network_removal=network_removal, vpm_ca2p0=vpm_ca2p0, ab=ab, variable=variable)
        df.to_pickle(file_name)
    return df


def load_spike_times(blue_configs, gids):
    """
    Load spike times from several simulations and put them in data frame
    """
    trial_dfs = []
    for j, blue_config in enumerate(blue_configs):
        print blue_config
        simulation = bluepy.Simulation(blue_config)
        # spike_series = simulation.v2.reports['spikes'].data(gids=gids)
        df = simulation.v2.spikes.get(gids=gids)
        # hacky because of bluepy version change mid-project
        spike_series = pd.Series(df.index, index=df.values)
        trial_series = pd.Series(j * np.ones(len(spike_series)), index=spike_series.index, dtype=int)
        trial_dfs.append(pd.concat([spike_series, trial_series], axis=1))
    df = pd.concat(trial_dfs, axis=0)
    df.columns = ['time', 'trial']
    return df


def load_spike_times_variable():
    bcs = get_configs_variable()
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df


def load_spike_times_network():
    bcs = get_configs_network()
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df

def load_spike_times_network_ab():
    bcs = get_configs_network_ab()
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df

def load_spike_times_network_nr_vpmca2p0():
    bcs = get_configs_network_nr_vpmca2p0()
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df

def load_spike_times_network_nr():
    bcs = get_configs_network_nr()
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df


def load_spike_times_spont():
    bcs = [spont_path % i for i in spont_seeds]
    print bcs
    sim = bluepy.Simulation(bcs[0])
    gids = list(sim.get_circuit_target())
    df = load_spike_times(bcs, gids)
    return df

def load_spike_times_decoupled():
    bcs_all = get_configs_decoupled()
    print bcs_all
    sim = bluepy.Simulation(bcs_all[0][0])
    gids = list(sim.get_circuit_target())
    orig_trial_dfs = []
    for i, bcs in enumerate(bcs_all):
        df = load_spike_times(bcs, gids)
        df['original_trial'] = pd.Series(i * np.ones(len(df['time'])), index=df.index, dtype=int)
        orig_trial_dfs.append(df)
    return pd.concat(orig_trial_dfs, axis=0)

def load_spike_times_decoupled_ab():
    bcs_all = get_configs_decoupled_ab()
    print bcs_all
    sim = bluepy.Simulation(bcs_all[0][0])
    gids = list(sim.get_circuit_target())
    orig_trial_dfs = []
    for i, bcs in enumerate(bcs_all):
        df = load_spike_times(bcs, gids)
        df['original_trial'] = pd.Series(i * np.ones(len(df['time'])), index=df.index, dtype=int)
        orig_trial_dfs.append(df)
    return pd.concat(orig_trial_dfs, axis=0)

def get_spike_times_experiment_25(decouple=False, network_removal=False, vpm_ca2p0=False, ab=False, variable=False):
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    directory = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/'
    if variable:
        spike_file = 'times_exp_25_n30_id0_variable.pkl'
    else:
        if decouple:
            if ab:
                spike_file = 'times_exp_25_n30_id0_decouple_ab.pkl'
            else:
                spike_file = 'times_exp_25_n30_id0_decouple.pkl'

        else:
            if network_removal:
                if vpm_ca2p0:
                    spike_file = 'times_exp_25_n30_id0_nr_vpm2p0.pkl'
                else:
                    spike_file = 'times_exp_25_n30_id0_nr.pkl'
            else:
                if ab:
                    spike_file = 'times_exp_25_n30_id0_ab.pkl'
                else:
                    spike_file = 'times_exp_25_n30_id0.pkl'
    file_name = os.path.join(directory, spike_file)
    if os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
    else:
        if variable:
            df = load_spike_times_variable()
        else:
            if decouple:
                if ab:
                    df = load_spike_times_decoupled_ab()
                else:
                    df = load_spike_times_decoupled()
            else:
                if network_removal:
                    if vpm_ca2p0:
                        df = load_spike_times_network_nr_vpmca2p0()
                    else:
                        df = load_spike_times_network_nr()
                else:
                    if ab:
                        df = load_spike_times_network_ab()
                    else:
                        df = load_spike_times_network()
        df.to_pickle(file_name)
    return df

def get_spike_times_spont():
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    directory = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/'
    spike_file = 'spont.pkl'
    file_name = os.path.join(directory, spike_file)
    if os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
    else:
        df = load_spike_times_spont()
        df.to_pickle(file_name)
    return df

def example_raster_plot():
    l4 = [71211, 70922, 71042,  71338, 74266, 74419, 74882, 71610,]
    l5 = [75998, 81442, 82079, 93653, 93902, 81703, 81443, 82202]
    l6 = [93655, 75759 ,76251, 80451, 80762, 81068, 81255, 81316]

    for n in range(1):
        gids_to_plot = [74419, 75270, 81442, 93272] #81068] #[l4[n], l5[n], l6[n]]
        # gids_to_plot = [85422, 90721]
        fig, axs = plt.subplots(2, 4, figsize=(12, 5))

        df_network = get_spike_times_experiment_25(decouple=False).loc[gids_to_plot]
        df_decoupled = get_spike_times_experiment_25(decouple=True).loc[gids_to_plot]

        t_start = 5200
        t_end = 5400
        # t_start = 1500
        # t_end = 2000
        df_network = df_network[(df_network['time'] < t_end) & (df_network['time'] >= t_start)]
        df_decoupled = df_decoupled[(df_decoupled['time'] < t_end) & (df_decoupled['time'] >= t_start)]
        # df_network['time'] -= t_start
        # df_decoupled['time'] -= t_start
        # t_end -= t_start
        # t_start = 0
        for k, gid in enumerate(gids_to_plot):
            print k
            axs[0, k].fill_between([t_start, t_end], [0, 0], [5, 5], color='red', alpha=0.2, linewidth=0.0)
            axs[0, k].fill_between([t_start, t_end], [5, 5], [30, 30], color='red', alpha=0.1, linewidth=0.0)
            plot_times = df_network.loc[gid]['time']
            plot_trials = df_network.loc[gid]['trial']
            axs[0, k].vlines(plot_times, plot_trials, plot_trials + 1, linewidth=0.5)

            for i in range(5):
                axs[1, k].plot([t_start, t_end], [6*i, 6*i], color='black', linewidth=0.5)
                axs[1, k].fill_between([t_start, t_end], [6*i, 6*i], [6*i+1, 6*i+1], color='red', alpha=0.2, linewidth=0.0)
                axs[1, k].fill_between([t_start, t_end], [6*i+1, 6*i+1], [6*i+6, 6*i+6], color='green', alpha=0.1, linewidth=0.0)
                plot_times_2 = plot_times[plot_trials == i]
                axs[1, k].vlines(plot_times_2, np.zeros(plot_times.size)+i*6, np.zeros(plot_times.size)+i*6+1, linewidth=0.5)

                trial_df = (df_decoupled.loc[gid])
                trial_df = trial_df[trial_df['original_trial'] == i]
                plot_times_dec = trial_df['time']
                plot_trials_dec =  trial_df['trial']
                axs[1, k].vlines(plot_times_dec, plot_trials_dec+ i*6 + 1, plot_trials_dec+i*6 + 2, linewidth=0.5)

        for ax in axs[1, :]:
            ax.set_ylim([0, 30])
            ax.set_yticks(np.arange(0, 30, 6) + 1.5)
            ax.set_yticklabels(range(1, 6))
            ax.set_ylabel('Decoupled trials')

        for ax in axs[0, :]:
            ax.set_ylim([0, 30])
            ax.set_yticks([0.5, 4.5, 9.5, 14.5, 19.5, 25.5, 29.5])
            ax.set_yticklabels([1, 5, 10, 15, 20, 25, 30])
            ax.set_ylabel('Network trials')

        for ax in axs.flatten():
            ax.set_xlim([t_start, t_end])
            ax.set_xlabel('t (ms)')

        for i in range(4):
             axs[0, i].set_title(gids_to_plot[i])
        plt.tight_layout()
        plt.savefig('figures_stim/rasterplot_example.pdf')


def get_selected_L456_gids():
    circuit = bluepy.Simulation(get_configs_network()[0]).circuit
    cells = circuit.v2.cells({Cell.MINICOLUMN: range(620, 650)})
    print len(cells.index)
    return cells[cells['layer'] > 3]


def raster_global():
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    df_network = get_spike_times_experiment_25(decouple=False)
    spikes = np.array(df_network['time'])

    axs[0, 0].hist(spikes, bins=np.linspace(0, 7000, 141))
    axs[0, 1].hist(spikes, bins=np.linspace(500, 7000, 141))

    plt.tight_layout()
    plt.savefig('figures_stim/raster_global.pdf')


def get_soma_correlations(gids, decouple=True):
    index_correlation = 1
    circuit = bluepy.Simulation(get_configs_network()[0]).circuit
    min_gid = np.min(list(circuit.get_target('mc2_Column')))
    corrs, bins = iaf.get_correlations(parameter_continue='', parameter_change='abcd', dt=10.0, decouple=decouple)
    print "checkpoint 2"
    print corrs.shape
    gids = np.array(gids, dtype=int)
    means_corr = corrs.mean(axis=-1)[gids.astype(int) - min_gid, :, index_correlation]
    print "checkpoint 1"
    errs = corrs[gids.astype(int) - min_gid, :, index_correlation].std(axis=-1)
    return means_corr, errs, bins


def reliabilities_analysis(sigma_conv=5, dt_conv=1, spont=False, ab=False):
    df_1 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=spont, ab=ab)
    df_2 = get_reliabilities_experiment_25(decouple=True, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=spont, ab=ab)

    s_ab = ''
    if ab:
        s_ab = '_ab'

    gids = np.intersect1d(df_1.index, df_2.index)
    df_1 = df_1.loc[gids]
    df_2 = df_2.loc[gids]

    df_neurons = get_selected_L456_gids()
    print "Selected L456 shape"
    print df_neurons.shape
    print "Selected L456 exc shape"

    print df_neurons[df_neurons['synapse_class'] == 'EXC'].shape

    df_neurons = df_neurons.loc[gids]
    print "Selected L456 exc + responding shape"
    print df_neurons[df_neurons['synapse_class'] == 'EXC'].shape

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(np.array(df_1[df_neurons['synapse_class'] == 'EXC']).mean(axis=1), color='black', alpha=0.3, bins=np.linspace(0, 1, 31))
    axs[0, 0].hist(np.array(df_2[df_neurons['synapse_class'] == 'EXC']).mean(axis=1), color='black', alpha=1.0, bins=np.linspace(0, 1, 31), histtype='step')
    axs[0, 1].hist(np.array(df_2[df_neurons['synapse_class'] == 'EXC']).mean(axis=1) - np.array(df_1[df_neurons['synapse_class'] == 'EXC']).mean(axis=1), color='black', bins=np.linspace(-0.1, 0.4, 41),
                   hatch='/', histtype='step')

    axs[1, 1].scatter(np.array(df_2[df_neurons['synapse_class'] == 'EXC']).mean(axis=1) - np.array(df_1[df_neurons['synapse_class'] == 'EXC']).mean(axis=1),
                      df_neurons[df_neurons['synapse_class'] == 'EXC']['y'], c=df_neurons[df_neurons['synapse_class'] == 'EXC']['layer']/6.0, s=3.0, cmap='summer')
    axs[0, 0].set_xlabel('S')
    axs[0, 1].set_xlabel('dS')
    axs[0, 0].set_ylabel('Neurons')
    axs[0, 1].set_ylabel('Neurons')
    axs[0, 0].set_xlabel('S')
    axs[0, 1].set_ylabel('dS')
    axs[1, 0].set_xlabel('y')
    axs[1, 1].set_ylabel('y')
    axs[0, 0].set_xlim([0, 1])
    axs[1, 0].set_xlim([0, 1])
    axs[0, 1].set_xlim([-0.05, 0.25])
    axs[1, 1].set_xlim([-0.05, 0.25])

    plt.tight_layout()
    if spont:
        plt.savefig('figures_stim/reliabilities_sigma_%d_spont%s.pdf' % (sigma_conv, s_ab))
    else:
        plt.savefig('figures_stim/reliabilities_sigma_%d%s.pdf' % (sigma_conv, s_ab))


    gids_all = df_neurons[df_neurons['synapse_class'] == 'EXC'].index
    corrs, errs, bins = get_soma_correlations(gids_all, decouple=False)
    corrs_dec, errs, bins = get_soma_correlations(gids_all, decouple=True)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(corrs.mean(axis=-1), color='black', alpha=0.3, bins=np.linspace(0, 1, 21))
    axs[0, 0].hist(corrs_dec.mean(axis=-1), color='black', alpha=1.0, bins=np.linspace(0, 1, 21), histtype='step')
    axs[0, 1].hist(corrs_dec.mean(axis=-1) - corrs.mean(axis=-1), color='black', bins=np.linspace(0, 1.0, 21),
                   hatch='/', histtype='step')
    axs[1, 0].scatter(corrs.mean(axis=-1),
                      df_neurons[df_neurons['synapse_class'] == 'EXC']['y'],  c=df_neurons[df_neurons['synapse_class'] == 'EXC']['layer'], s=3.0, cmap='summer')
    axs[1, 1].scatter(corrs_dec.mean(axis=-1) - corrs.mean(axis=-1),
                      df_neurons[df_neurons['synapse_class'] == 'EXC']['y'], c=df_neurons[df_neurons['synapse_class'] == 'EXC']['layer']/6.0, s=3.0, cmap='summer')


    corrs_gids, errs_gids, bins = get_soma_correlations(gids, decouple=False)
    corrs_dec_gids, errs_gids, bins = get_soma_correlations(gids, decouple=True)
    # axs[1, 1].scatter(corrs_dec_gids.mean(axis=-1) - corrs_gids.mean(axis=-1),
    #                   df_neurons.loc[df_x.index]['y'], marker='^', color='white', edgecolor='black')
    # axs[1, 0].scatter(corrs_gids.mean(axis=-1),
    #                   df_neurons.loc[df_x.index]['y'], marker='^', color='white', edgecolor='black')
    axs[0, 0].set_xlabel('S')
    axs[0, 1].set_xlabel('dS')
    axs[0, 0].set_ylabel('Neurons')
    axs[0, 1].set_ylabel('Neurons')
    axs[0, 0].set_xlabel('S')
    axs[0, 1].set_xlabel('dS')
    axs[1, 0].set_ylabel('y')
    axs[1, 1].set_ylabel('y')
    axs[0, 0].set_xlim([0, 1])
    axs[1, 0].set_xlim([-0.05, 1])
    axs[0, 1].set_xlim([0, 1.0])
    axs[1, 1].set_xlim([0, 1.0])
    plt.tight_layout()
    if spont:
        plt.savefig('figures_stim/reliabilities_sigma_%d_spont_soma%s.pdf' % (sigma_conv, s_ab))
    else:
        plt.savefig('figures_stim/reliabilities_sigma_%d_soma%s.pdf' % (sigma_conv, s_ab))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(corrs.mean(axis=-1), np.array(df_1[df_neurons['synapse_class'] == 'EXC']).mean(axis=1),
                      marker='.', color='blue', alpha=0.5, edgecolor='')
    axs[0, 0].set_xlabel('r')
    axs[0, 0].set_ylabel('S')

    axs[0, 1].scatter(corrs_dec.mean(axis=-1), np.array(df_2[df_neurons['synapse_class'] == 'EXC']).mean(axis=1),
                      marker='.', color='blue', alpha=0.5, edgecolor='')
    axs[0, 1].set_xlabel('r - dec')
    axs[0, 1].set_ylabel('S - dec')



    axs[1, 0].scatter(corrs_dec.mean(axis=-1) - corrs.mean(axis=-1), np.array(df_2[df_neurons['synapse_class'] == 'EXC']).mean(axis=1) - np.array(df_1[df_neurons['synapse_class'] == 'EXC']).mean(axis=1),
                       marker='.', color='blue', alpha=0.5, edgecolor='')
    axs[1, 0].set_xlabel('diff r')
    axs[1, 0].set_ylabel('diff s')

    slope, intercept, r_value, p_value, std_err = stats.linregress(corrs_dec.mean(axis=-1) - corrs.mean(axis=-1), np.array(df_2[df_neurons['synapse_class'] == 'EXC']).mean(axis=1) - np.array(df_1[df_neurons['synapse_class'] == 'EXC']).mean(axis=1))
    x = corrs_dec.mean(axis=-1) - corrs.mean(axis=-1)
    axs[1, 0].plot(x, x * slope + intercept, color='black')
    axs[1, 0].set_title("p=%.3f err=%.3f" % (p_value, std_err))
    print "slope, intercept, r_value, p_value, std_err"
    print slope, intercept, r_value, p_value, std_err

    gids = [74419, 75270, 93272]
    corrs_gids, errs_gids, bins = get_soma_correlations(gids, decouple=False)
    corrs_dec_gids, errs_gids, bins = get_soma_correlations(gids, decouple=True)
    axs[1, 0].scatter(corrs_dec_gids.mean(axis=-1) - corrs_gids.mean(axis=-1), np.array(df_2[df_neurons['synapse_class'] == 'EXC'].loc[gids]).mean(axis=1) - np.array(df_1[df_neurons['synapse_class'] == 'EXC'].loc[gids]).mean(axis=1),
                      marker='^', color='white', edgecolor='black')
    axs[1, 0].set_xlabel('diff r')
    axs[1, 0].set_ylabel('diff s')

    plt.tight_layout()

    if spont:
        plt.savefig('figures_stim/reliabilities_sigma_%d_spont_soma_corr%s.pdf' % (sigma_conv, s_ab))
    else:
        plt.savefig('figures_stim/reliabilities_sigma_%d_soma_corr%s.pdf' % (sigma_conv, s_ab))


def spike_count_analysis():
    df_1 = get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1, spont=True)
    df_2 = get_reliabilities_experiment_25(decouple=True, sigma_conv=5, dt_conv=1, spont=True)

    gids = np.intersect1d(df_1.index, df_2.index)


    spikes_df = get_spike_times_experiment_25(decouple=False)
    spikes_df_dec = get_spike_times_experiment_25(decouple=True)

    df_neurons = get_selected_L456_gids()
    df_neurons = df_neurons.loc[gids]
    df_neurons = df_neurons[df_neurons['synapse_class'] == 'EXC']

    spikes_df = spikes_df.loc[df_neurons.index]
    spikes_df_dec = spikes_df_dec.loc[df_neurons.index]

    gids = np.unique(spikes_df.index)
    print len(gids)
    gids = np.unique(spikes_df_dec.index)
    print len(gids)

    df_counts = pd.DataFrame(index=gids, columns=('mean', 'variance', 'mean_pre', 'variance_pre'), dtype=float)
    df_counts_dec = pd.DataFrame(index=gids, columns=('mean', 'variance', 'mean_pre', 'variance_pre'), dtype=float)

    for gid in gids:
        gid_df = spikes_df.loc[gid]
        counts_pre = np.bincount(gid_df[(gid_df['time'] >= 1500) & (gid_df['time'] < 2000)]['trial'], minlength=30) / 0.5 # this is wrong for fano factor!!
        counts = np.bincount(gid_df[gid_df['time'] >= 2000]['trial'], minlength=30) / 5.0
        print counts_pre
        print counts.shape
        df_counts.set_value(gid, 'mean', counts.mean())
        df_counts.set_value(gid, 'variance', np.var(counts, ddof=1))
        df_counts.set_value(gid, 'mean_pre', counts_pre.mean())
        df_counts.set_value(gid, 'variance_pre', np.var(counts_pre, ddof=1))
        print gid
    for gid in gids:
        print gid
        gid_df_all = spikes_df_dec.loc[gid]
        means = np.zeros(30)
        variances = np.zeros(30)
        means_pre =np.zeros(30)
        variances_pre =np.zeros(30)
        for i in range(30):
            gid_df = gid_df_all[gid_df_all['original_trial'] == i]
            counts_pre = np.bincount(gid_df[(gid_df['time'] >= 1500) & (gid_df['time'] < 2000)]['trial'], minlength=5) / 0.5
            counts = np.bincount(gid_df[gid_df['time'] >= 2000]['trial'], minlength=5) / 5.0
            means[i] = counts.mean()
            variances[i] = np.var(counts, ddof=1)
            means_pre[i] = counts_pre.mean()
            variances_pre[i] = np.var(counts_pre, ddof=1)
        df_counts_dec.set_value(gid, 'mean', means.mean())
        df_counts_dec.set_value(gid, 'variance', variances.mean())
        df_counts_dec.set_value(gid, 'mean_pre', means_pre.mean())
        df_counts_dec.set_value(gid, 'variance_pre', variances_pre.mean())

    df_spont = get_spike_times_spont()
    df_spont = df_spont[df_spont['time'] >= 1500]
    df_spont = df_spont.loc[np.intersect1d(gids, df_spont.index)]
    df_counts_spont = pd.DataFrame(0.0, index=gids, columns=('mean_pre', 'variance_pre'), dtype=float)
    for gid in np.unique(df_spont.index):
        print gid
        gid_df = df_spont.loc[[gid]]
        counts = np.bincount(gid_df['trial'], minlength=40) / 0.5
        df_counts_spont.set_value(gid, 'mean_pre', counts.mean())
        df_counts_spont.set_value(gid, 'variance_pre', np.var(counts, ddof=1))

    fig, axs = plt.subplots(4, 2, figsize=(6, 10))

    axs[0, 0].scatter(df_counts['mean'], df_counts['variance'], s=3.0)
    axs[0, 0].scatter(df_counts['mean_pre'], df_counts['variance_pre'], s=3.0)
    axs[0, 0].set_xlabel('FR (Hz)')
    axs[0, 0].set_ylabel('Var(FR)')
    axs[0, 0].plot([0, 5], [0, 5], 'r--')

    axs[0, 1].scatter(df_counts_dec['mean'], df_counts_dec['variance'], s=3.0)
    axs[0, 1].scatter(df_counts_dec['mean_pre'], df_counts_dec['variance_pre'], s=3.0)
    axs[0, 1].set_xlabel('dec. FR (Hz)')
    axs[0, 1].set_ylabel('dec. Var(FR)')
    axs[0, 1].plot([0, 5], [0, 5], 'r--')

    axs[1, 0].scatter(df_counts['mean'], df_counts_dec['mean'], s=3.0)
   # axs[0, 1].scatter(df_counts_dec['mean_pre'], df_counts_dec['variance_pre'], s=3.0)
    axs[1, 0].set_xlabel('FR (Hz)')
    axs[1, 0].set_ylabel('dec. FR (Hz)')
    axs[1, 0].plot([0, 5], [0, 5], 'r--')

    axs[1, 1].scatter(df_counts['variance'], df_counts_dec['variance'], s=3.0)
   # axs[0, 1].scatter(df_counts_dec['mean_pre'], df_counts_dec['variance_pre'], s=3.0)
    axs[1, 1].set_xlabel('Var(FR)')
    axs[1, 1].set_ylabel('dec. Var(FR)')
    axs[1, 1].plot([0, 0.25], [0, 0.25], 'r--')

    axs[2, 0].scatter(df_counts['mean_pre'], df_counts_dec['mean_pre'], s=3.0, marker='.', edgecolor='')
   # axs[0, 1].scatter(df_counts_dec['mean_pre'], df_counts_dec['variance_pre'], s=3.0)
    axs[2, 0].set_xlabel('FR (Hz)')
    axs[2, 0].set_ylabel('dec. FR (Hz)')
    axs[2, 0].plot([0, 5], [0, 5], 'r--')

    print np.array(df_counts['mean_pre'])
    print np.array(df_counts_dec['mean_pre']).shape

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(df_counts['mean_pre']), np.array(df_counts_dec['mean_pre']))
    x = np.array(df_counts['mean_pre'])
    axs[2, 0].plot(x, x * slope + intercept, color='black')
    axs[2, 0].set_title("p=%.3f err=%.3f" % (p_value, std_err))

    axs[2, 1].scatter(df_counts['variance_pre']/df_counts['mean_pre'], df_counts_dec['variance_pre']/df_counts_dec['mean_pre'], s=3.0,
                      marker='.', edgecolor='')
   # axs[0, 1].scatter(df_counts_dec['mean_pre'], df_counts_dec['variance_pre'], s=3.0)
    axs[2, 1].set_xlabel('FF')
    axs[2, 1].set_ylabel('dec. FF')
    axs[2, 1].set_aspect('equal')
    axs[2, 1].plot([0, 3.5], [0, 3.5], 'r--')

    print np.array(df_counts['variance_pre'])
    print np.array(df_counts_dec['variance_pre'])

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(df_counts['variance_pre'])/np.array(df_counts['mean_pre']), np.array(df_counts_dec['variance_pre'])/np.array(df_counts_dec['mean_pre']))
    x =np.array(df_counts['variance_pre'])/np.array(df_counts['mean_pre'])
    axs[2, 1].plot(x, x * slope + intercept, color='black')
    axs[2, 1].set_title("p=%.3f err=%.3f" % (p_value, std_err))
    print "slope, intercept, r_value, p_value, std_err"
    print slope, intercept, r_value, p_value, std_err

    axs[3, 0].hist(df_counts_dec['mean_pre'] - df_counts['mean_pre'], bins=np.linspace(-0.75, 0.75, 20),
                   color='black', hatch='/', histtype='step')
    axs[3, 1].hist(df_counts_dec['variance_pre'] - df_counts['variance_pre'], bins=np.linspace(-1.2, 1.2, 20),
                   color='black', hatch='/', histtype='step')

    plt.tight_layout()
    plt.savefig('figures_stim/spike_counts.pdf')

    return df_counts


def get_vpm_in_degree():
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/connection_matrix_with_vpm/mc2_with_vpm.npz'
    data = np.load(open(file_path, 'r'))
    matrix_vpm = data['matrix_vpm']
    return matrix_vpm.sum(axis=0)

def get_vpm_in_degree_active():
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/connection_matrix_with_vpm/mc2_with_vpm.npz'
    data = np.load(open(file_path, 'r'))
    matrix_vpm = data['matrix_vpm']
    print matrix_vpm.shape
    return matrix_vpm[620:930, :].sum(axis=0)

def plot_reliablities_network_effect(sigma_conv=5, dt_conv=1):
    vpm_in_degree = pd.Series(get_vpm_in_degree_active(), index=bluepy.Simulation(get_configs_network()[0]).v2.target_gids)

    print vpm_in_degree
    df_0 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=False, vpm_ca2p0=False)
    df_1 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=True, vpm_ca2p0=False)

    # print df_0.shape
    print df_1.shape
    gids = np.intersect1d(df_0.index, df_1.index)

    df_neurons = get_selected_L456_gids()
    #print df_1[(df_neurons['synapse_class'] == 'EXC') & (df_neurons['layer'] == 3)].mean(axis=1)

    df_neurons = df_neurons.loc[gids]
    print df_neurons.shape

    df_0 = df_0.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_1 = df_1.loc[gids][df_neurons['synapse_class'] == 'EXC']
    vpm_in_degree = vpm_in_degree.loc[gids][df_neurons['synapse_class'] == 'EXC']
    print df_0.shape
    print df_1.shape

    fig, axs = plt.subplots(3, 4, figsize=(15, 8))
    colors_bright = ['#66c2a5','#fc8d62','#8da0cb']
    colors_dark = ['#1b9e77','#d95f02','#7570b3']
    for i, l in enumerate([4, 5, 6]):
        colors = colors_dark
        j = 0
        diff_1 = np.array(df_1[df_neurons['layer'] == l].mean(axis=1)) - np.array(df_0[df_neurons['layer'] == l].mean(axis=1))
        axs[j, 0].hist(diff_1, color=colors[i], histtype='step')
        axs[j, 1].scatter(np.array(df_0[df_neurons['layer'] == l].mean(axis=1)), np.array(df_1[df_neurons['layer'] == l].mean(axis=1)), color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')
        axs[j, 1].plot([0, 0.4], [0, 0.4], 'r--')
        axs[j, 1].set_xlim([0, 1])
        axs[j, 1].set_ylim([-0.025, 0.4])
        axs[j, 1].set_ylabel('r - single cell')
        axs[j, 1].set_xlabel('r - network')
        axs[j, 0].set_ylabel('%d Neurons' % (df_0.shape[0]))
        axs[j, 0].set_xlabel('r - s.c. - netw.')

        for n in range(2):
            colors = [colors_dark, colors_bright][n]
            df = [df_0, df_1][n]
            x = vpm_in_degree[df_neurons['layer'] == l]
            y = np.array(df[df_neurons['layer'] == l].mean(axis=1))
            axs[i, 0 + 2].scatter(x, y,
                              color=colors[i], alpha=0.5, marker='.', edgecolor='')
            bins = [5, 10, 15, 20, 25, 30]

            values = np.digitize(x, bins)
            print values
            xs = np.arange(len(bins) + 1)
            means = np.zeros(xs.shape)
            stds = np.zeros(xs.shape)
            for k in xs:
                means[k] = y[values == k].mean()
                stds[k] = y[values == k].std(ddof=1)/np.sqrt(y[values == k].size)
            print means
            bins_2 = np.append(np.append([0], np.repeat(bins, 2)), 60)
            axs[i, 0 + 2].fill_between(bins_2, np.repeat(means + stds, 2), np.repeat(means - stds, 2), color=colors[i],
                                       alpha=0.4, edgecolor='')
            axs[i, 0 + 2].plot(bins_2, np.repeat(means, 2), color=colors[i])
            axs[i, 0 + 2].set_xlabel('Active VPM fibers')
            axs[i, 0 + 2].set_ylabel('r-spike')

        # axs[j, 3].scatter(vpm_in_degree[df_neurons['layer'] == l], np.array(df_1[df_neurons['layer'] == l].mean(axis=1)),
        #                   color=colors[i], alpha=0.5, marker='.', edgecolor='')

    print "Done!"
    plt.tight_layout()
    plt.savefig('figures_stim/reliabilities_network_removed_sigma_%d_spont_soma_corr.pdf' % sigma_conv)


def scatter_reliability(sigma_conv=5, dt_conv=1):
    vpm_in_degree = pd.Series(get_vpm_in_degree_active(), index=bluepy.Simulation(get_configs_network()[0]).v2.target_gids)

    print vpm_in_degree
    df_0 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=False, vpm_ca2p0=False)
    df_1 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=True, vpm_ca2p0=False)
    df_2 = get_reliabilities_experiment_25(decouple=True, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=False, vpm_ca2p0=False)
    df_3 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=True,
                                           network_removal=True, vpm_ca2p0=False)

    # print df_0.shape
    print df_1.shape
    gids = np.intersect1d(np.intersect1d(np.intersect1d(df_0.index, df_1.index), df_2.index), df_3.index)


    df_neurons = get_selected_L456_gids()
    #print df_1[(df_neurons['synapse_class'] == 'EXC') & (df_neurons['layer'] == 3)].mean(axis=1)

    df_neurons = df_neurons.loc[gids]
    print df_neurons.shape

    df_0 = df_0.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_1 = df_1.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_2 = df_2.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_3 = df_3.loc[gids][df_neurons['synapse_class'] == 'EXC']


    vpm_in_degree = vpm_in_degree.loc[gids][df_neurons['synapse_class'] == 'EXC']
    print df_0.shape
    df_neurons = df_neurons[df_neurons['synapse_class'] == 'EXC']
    print df_neurons.shape
    fig, axs = plt.subplots(3, 4, figsize=(15, 8))
    colors_bright = ['#66c2a5','#fc8d62','#8da0cb']
    colors_dark = ['#1b9e77','#d95f02','#7570b3']

    for i, l in enumerate([4, 5, 6]):
        colors = colors_dark
        print df_neurons[df_neurons['layer'] == l].y.shape
        print np.array(df_3[df_neurons['layer'] == l].mean(axis=1)).shape
        axs[0, 0].scatter(np.array(df_3[df_neurons['layer'] == l].mean(axis=1)), df_neurons[df_neurons['layer'] == l].y, color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')
        axs[0, 1].scatter(np.array(df_0[df_neurons['layer'] == l].mean(axis=1)), df_neurons[df_neurons['layer'] == l].y, color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')
        axs[0, 2].scatter(np.array(df_1[df_neurons['layer'] == l].mean(axis=1)), df_neurons[df_neurons['layer'] == l].y, color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')
        axs[0, 3].scatter(np.array(df_2[df_neurons['layer'] == l].mean(axis=1)) - df_0[df_neurons['layer'] == l].mean(axis=1), df_neurons[df_neurons['layer'] == l].y, color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')
        for j in range(4):
            axs[0, j].set_xlim([0, 1])
            axs[0, j].set_ylim([0, 1500])
        #axs[j, 1].set_ylim([-0.025, 0.4])
        axs[0, 0].set_ylabel('r - single cell')
        axs[0, 0].set_xlabel('r - network')
        axs[0, 0].set_ylabel('%d Neurons' % (df_0.shape[0]))
        axs[0, 0].set_xlabel('r - s.c. - netw.')
    axs[1, 0].hist(df_neurons.y, weights=vpm_in_degree/float(df_neurons.y.size), histtype='stepfilled', bins=30)
    axs[1, 0].set_xlim([0, 1500])

    print "Done!"
    plt.tight_layout()
    plt.savefig('figures_stim/scatter_reliability_sigma_%d_spont_soma_corr.pdf' % sigma_conv)

def reliability_variable(sigma_conv=5, dt_conv=1):
    vpm_in_degree = pd.Series(get_vpm_in_degree_active(),
                              index=bluepy.Simulation(get_configs_network()[0]).v2.target_gids)

    print vpm_in_degree
    df_0 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=False, vpm_ca2p0=False)
    df_1 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=True, vpm_ca2p0=False)
    df_2 = get_reliabilities_experiment_25(decouple=True, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=False, vpm_ca2p0=False)
    df_3 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=True,
                                           network_removal=True, vpm_ca2p0=False)
    df_4 = get_reliabilities_experiment_25(decouple=False, sigma_conv=sigma_conv, dt_conv=dt_conv, spont=False,
                                           network_removal=False, vpm_ca2p0=False, variable=True)
    # print df_0.shape
    print df_1.shape
    gids = np.intersect1d(np.intersect1d(np.intersect1d(np.intersect1d(df_0.index, df_1.index), df_2.index), df_3.index), df_4.index)

    df_neurons = get_selected_L456_gids()
    # print df_1[(df_neurons['synapse_class'] == 'EXC') & (df_neurons['layer'] == 3)].mean(axis=1)

    df_neurons = df_neurons.loc[gids]
    print df_neurons.shape

    df_0 = df_0.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_1 = df_1.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_2 = df_2.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_3 = df_3.loc[gids][df_neurons['synapse_class'] == 'EXC']
    df_4 = df_4.loc[gids][df_neurons['synapse_class'] == 'EXC']


    vpm_in_degree = vpm_in_degree.loc[gids][df_neurons['synapse_class'] == 'EXC']
    print df_0.shape
    df_neurons = df_neurons[df_neurons['synapse_class'] == 'EXC']
    print df_neurons.shape
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    colors_bright = ['#66c2a5', '#fc8d62', '#8da0cb']
    colors_dark = ['#1b9e77', '#d95f02', '#7570b3']

    for i, l in enumerate([4, 5, 6]):
        colors = colors_dark
        print df_neurons[df_neurons['layer'] == l].y.shape
        print np.array(df_3[df_neurons['layer'] == l].mean(axis=1)).shape
        axs[0, 0].scatter(np.array(df_3[df_neurons['layer'] == l].mean(axis=1)), df_neurons[df_neurons['layer'] == l].y,
                          color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')
        axs[0, 1].scatter(np.array(df_0[df_neurons['layer'] == l].mean(axis=1)), df_neurons[df_neurons['layer'] == l].y,
                          color=colors[i],
                          alpha=0.5, marker='.', edgecolor='')

        axs[0, 2].scatter(
            np.array(df_4[df_neurons['layer'] == l].mean(axis=1)),
            df_neurons[df_neurons['layer'] == l].y, color=colors[i],
            alpha=0.5, marker='.', edgecolor='')
        for j in range(3):
            axs[0, j].set_xlim([0, 1])
            axs[0, j].set_ylim([0, 1500])
        # axs[j, 1].set_ylim([-0.025, 0.4])
        axs[0, 0].set_ylabel('r - single cell')
        axs[0, 0].set_xlabel('r - network')
        axs[0, 0].set_ylabel('%d Neurons' % (df_0.shape[0]))
        axs[0, 0].set_xlabel('r - s.c. - netw.')

    axs[1, 0].hist(df_neurons.y, weights=vpm_in_degree / float(df_neurons.y.size), histtype='stepfilled', bins=30)
    axs[1, 1].hist(np.array(df_3[df_neurons['layer'] == l].mean(axis=1)), color='red', histtype='step', bins=np.linspace(0, 1, 21))
    axs[1, 1].hist(np.array(df_0[df_neurons['layer'] == l].mean(axis=1)), color='blue', histtype='step', bins=np.linspace(0, 1, 21))
    axs[1, 1].hist(np.array(df_4[df_neurons['layer'] == l].mean(axis=1)), color='gray', histtype='step', bins=np.linspace(0, 1, 21))
    axs[1, 2].scatter(np.array(df_0[df_neurons['layer'] == l].mean(axis=1)), np.array(df_4[df_neurons['layer'] == l].mean(axis=1)), alpha=0.5, marker='.', edgecolor='')
    axs[1, 0].set_xlim([0, 1500])

    print "Done!"
    plt.tight_layout()
    plt.savefig('figures_stim/variable_reliability_sigma_%d_spont_soma_corr.pdf' % sigma_conv)


def plot_selected_gids():
    vpm_in_degree = pd.Series(get_vpm_in_degree(), index=bluepy.Simulation(get_configs_network()[0]).v2.target_gids)
    vpm_in_degree_active = pd.Series(get_vpm_in_degree_active(), index=bluepy.Simulation(get_configs_network()[0]).v2.target_gids)
    circuit = bluepy.Simulation(get_configs_network()[0]).circuit
    df_neurons_other = circuit.v2.cells({Cell.MINICOLUMN: range(650, 930)})
    df_neurons = get_selected_L456_gids()
    df_neurons_all = circuit.v2.cells({Cell.MINICOLUMN: range(620, 930)})
    print df_neurons_all.shape

    xmean = df_neurons_all.x.mean()
    zmean = df_neurons_all.z.mean()

    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    ax.set_aspect('equal')
    ax.scatter(df_neurons_other.x, df_neurons_other.y, color='orange', alpha=0.1, s=1, marker='.')
    ax.scatter(df_neurons.x, df_neurons.y, color='red', alpha=0.3, s=1, marker='.')
    ax = axs[0, 1]
    ax.set_aspect('equal')
    ax.scatter(df_neurons_other.x, df_neurons_other.z, color='orange', alpha=0.1, s=1, marker='.')
    ax.scatter(df_neurons.x, df_neurons.z, color='red', alpha=0.3, s=1, marker='.')
    ax = axs[1, 0]
    ax.scatter(np.sqrt((df_neurons_all.x - xmean)**2 + (df_neurons_all.z - zmean)**2), vpm_in_degree-vpm_in_degree_active)
    ax.set_xlabel('distance from center')
    ax.set_ylabel('non active vpm fibers')

    ax = axs[1, 1]
    ax.scatter(np.sqrt((df_neurons_all.x - xmean)**2 + (df_neurons_all.z - zmean)**2), vpm_in_degree_active)
    ax.set_xlabel('distance from center')
    ax.set_ylabel('active vpm fibers')
    plt.tight_layout()
    plt.savefig('figures_stim/select_gids.pdf')

    return df_neurons


def compute_reliabilities_shuffled(sigma_conv=5, dt_conv=0.5, synapse_class='EXC'):
    # original sigma_conv = 3, dt = 0.5

    t_start = 2000
    t_end = 7000

    df_network = get_spike_times_experiment_25(decouple=False, network_removal=False, vpm_ca2p0=False, ab=False, variable=False)
    df_network = df_network[df_network['time'] >= t_start]
    df_network = df_network[df_network['time'] < t_end]
    gids = np.unique(df_network.index)
    df_neurons = get_selected_L456_gids()

    gids = np.intersect1d(df_neurons[df_neurons['synapse_class'] == synapse_class].index, gids)
    gids = np.intersect1d(df_neurons[df_neurons['layer'] == 6].index, gids)

    print "no of spiking gids: %d" % len(gids)
    print gids
    df_network = df_network.loc[gids]
    firing_rates = np.arange(1, 81, 10, dtype=int)
    df_results = pd.DataFrame(columns=range(435), index=firing_rates, dtype=np.float32)
    print df_results
    for k, fr in enumerate(firing_rates):
            print k
            how_many_spikes_to_pick = 5 * fr
            n_spikes = np.random.poisson(how_many_spikes_to_pick, size=30)
            print n_spikes
            spikes = np.random.choice(np.array(df_network['time']), size=n_spikes.sum())
            trials = np.hstack([i + np.zeros(n_spikes[i].sum(), dtype=int) for i in range(30)])
            mean, err, _distances = distances.compute_pairwise_distances(spikes, trials, [t_start, t_end], 30, 435, method='schreiber', time_sorted=True,
                                                trimmed=True, combi_seed=0, sigma_conv=sigma_conv, dt_conv=dt_conv)
            df_results.loc[fr] = _distances
    return df_results


def get_reliabilities_experiment_25_shuffled(sigma_conv=5, dt_conv=1, synapse_class='EXC'):
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    directory = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/'
    spike_file = 'reliabilities_exp_25_n30_id0_shuffled_%s_3.pkl' % synapse_class

    file_name = os.path.join(directory, spike_file)
    if not os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
    else:
        df = compute_reliabilities_shuffled(sigma_conv=sigma_conv, dt_conv=dt_conv, synapse_class=synapse_class)
        df.to_pickle(file_name)
    return df


if __name__ == "__main__":
    plot_correlations()
    compute_reliabilities()
    df = get_spike_times_experiment_25()
    df_2 = get_spike_times_experiment_25(decouple=True)
    example_raster_plot()
    reliabilities_analysis()
    raster_global()
    df = get_correlations(decouple=False)
    get_selected_L456_gids()
    df_results = compute_reliabilities(decouple=True)
    get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1)
    get_reliabilities_experiment_25(variable=True, sigma_conv=5, dt_conv=1)
    get_reliabilities_experiment_25(decouple=True, sigma_conv=5, dt_conv=1)
    get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1, spont=True)
    get_reliabilities_experiment_25(decouple=True, sigma_conv=5, dt_conv=1, spont=True)
    get_reliabilities_experiment_25(decouple=False, sigma_conv=20, dt_conv=5)
    get_reliabilities_experiment_25(decouple=True, sigma_conv=20, dt_conv=5)
    reliabilities_analysis(sigma_conv=1, dt_conv=0.5)
    reliabilities_analysis(sigma_conv=20, dt_conv=5)
    reliabilities_analysis(spont=True, ab=True)
    reliabilities_analysis(spont=False, ab=True)
    df_counts = spike_count_analysis()
    plot_reliablities_network_effect()
    matrix = get_vpm_in_degree()
    scatter_reliability()
    reliability_variable()
    get_reliabilities_experiment_25_shuffled(sigma_conv=5, dt_conv=1, synapse_class='INH')