#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import bluepy
from magicspike import distances
from bluepy.v2 import Cell
import sys
import multiprocessing as mp
import glob

sim_path_exp_25_cloud = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability_shuffled/exp_25_id0_30_%s/Ca%s/stimulusstim_a0/seed%d/'
seeds_exp_25 = {'control': np.arange(6000, 6030)}
cas_exp_25 = {'control': ['1p05', '1p1', '1p15', '1p2', '1p225', '1p25', '1p275', '1p3', '1p35']}

sim_types = seeds_exp_25.keys()

cas = ['1p05', '1p1', '1p15', '1p2', '1p25', '1p3', '1p35']
cas_comparison = ['1p2', '1p225', '1p25', '1p275', '1p3']

sim_path_jitter = "/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/jittering/stimulusstim_a%d/seed%d/"
ids_jitter = np.arange(6, dtype=int)
seeds_jitter = np.arange(10000, 10030, dtype=int)


sim_path_jitter_flick = "/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/jittering_flick/stimulusstim_a%d/"
seeds_jitter_flick = np.arange(0, 30, dtype=int)


sim_path_jitter_flick_control = "/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/jittering_flick_control/stimulusstim_a0/seed%d/"
seeds_jitter_flick_control = np.arange(1000, 1030, dtype=int)


sim_path_exp_25_original = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/synchrony/experiment_25/seed%d/n_classes%d/grouping_id%d/'
seeds_exp_25_original = np.arange(30)
n_classes = np.array([5, 15, 30])
grouping_ids = np.arange(3)



def get_exp_25_blueconfigs_cloud(n=30, ca='1p1', sim_type='control'):
    return [os.path.join(sim_path_exp_25_cloud % (sim_type, ca, seed), 'BlueConfig')
                for seed in seeds_exp_25[sim_type][:n]]


def get_exp_25_blueconfigs_original(n=30, n_classes=30, grouping_id=0):
    return [os.path.join(sim_path_exp_25_original % (seed, n_classes, grouping_id), 'BlueConfig')
                for seed in seeds_exp_25_original[:n]]


def get_jitter_blueconfigs(n=30, id_jitter=0):
    return [os.path.join(sim_path_jitter % (id_jitter, seed), 'BlueConfig')
                for seed in seeds_jitter[:n]]


def get_jitter_flick_blueconfigs(n=30):
    return [os.path.join(sim_path_jitter_flick % seed, 'BlueConfig')
                for seed in seeds_jitter_flick[:n]]

def get_jitter_flick_control_blueconfigs(n=30):
    return [os.path.join(sim_path_jitter_flick_control % seed, 'BlueConfig')
                for seed in seeds_jitter_flick_control[:n]]

def get_spike_times_multiple(blueconfigs):
    """

    :param blueconfigs:
    :return:
    """
    combined_frame=None
    for i, blueconfig in enumerate(blueconfigs):
        spike_times, spike_gids = get_spike_times_sim(blueconfig)

        trial_frame = pd.DataFrame(data={'spike_time': spike_times, 'spike_trial': np.zeros(spike_times.size, dtype=int) + i},
                     index=spike_gids.astype(int))
        if combined_frame is None:
            combined_frame = trial_frame
        else:
            combined_frame = pd.concat([combined_frame, trial_frame])
    return combined_frame


def get_spike_times_sim(blueconfig, t_start=500):
    """

    :param blueconfig:
    :return:
    """
    sim = bluepy.Simulation(blueconfig)
    spikes = sim.v2.spikes.get(t_start=t_start)
    spike_times = np.array(spikes.index)
    spike_gids = np.array(spikes.values)
    return spike_times, spike_gids


def compute_reliabilities(spike_df, gids, sigma_conv=5, dt_conv=0.5, t_start=1000, t_end=6000):
    """
    Compute reliabilities from 30 trial data frame
    :param spike_df:
    :param sigma_conv:
    :param dt_conv:
    :return:
    """

    spike_df = spike_df[spike_df['spike_time'] >= t_start]
    spike_df = spike_df[spike_df['spike_time'] < t_end]
    df_results = pd.DataFrame(columns=range(435), index=gids, dtype=np.float32)
    spike_gids = np.unique(spike_df.index)
    for k, gid in enumerate(np.setdiff1d(gids, spike_gids)):
        df_results.loc[gid] = np.zeros(435)
    print "%d non-spiking neurons" % np.setdiff1d(gids, spike_gids).size
    for k, gid in enumerate(np.intersect1d(gids, spike_gids)):
        print "%d out of %d" % (k, len(gids))
        spikes = np.array(spike_df.loc[[gid], 'spike_time'])
        trials = np.array(spike_df.loc[[gid], 'spike_trial'])
        mean, err, _distances = distances.compute_pairwise_distances(spikes, trials, [t_start, t_end], 30, 435, method='schreiber', time_sorted=True,
                                                trimmed=True, combi_seed=0, sigma_conv=sigma_conv, dt_conv=dt_conv)
        df_results.loc[gid] = _distances
    return df_results


def get_reliabilities_experiment_25(ca='1p25', sim_type='control'):
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/shuffled_pandas_exp25_%s_rel_v2.pkl'
    file_name = file_path % (sim_type + '_' + ca)
    if not os.path.isfile(file_name):
        bcs = get_exp_25_blueconfigs_cloud(n=30, ca=ca, sim_type=sim_type)
        gids = np.array(get_selected_L456_gids().index)
        print gids
        spike_df_orig = get_spike_times_multiple(bcs)
        reliability_df = compute_reliabilities(spike_df_orig, gids)
        reliability_df.to_pickle(file_name)
    else:
        reliability_df = pd.read_pickle(file_name)
    return reliability_df


def get_spike_counts_experiment_25(n=30, ca='1p25', sim_type='control'):
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/shuffled_pandas_exp25_counts_%s_v2.pkl'
    file_name = file_path % (sim_type + '_' + ca)
    if not os.path.isfile(file_name):
        bcs = get_exp_25_blueconfigs_cloud(n=30, ca=ca, sim_type=sim_type)
        df = get_spike_times_multiple(bcs)
        df = df[df['spike_time'] >= 1000]
        df = df[df['spike_time'] < 6000]
        sim = bluepy.Simulation(bcs[0])
        gids = np.sort(np.array(list(sim.get_circuit_target())))
        spikes_df = df.drop(columns=['spike_time'])
        counts_all = np.zeros((gids.size, n), dtype=np.float64)
        for i in range(n):
            x = spikes_df[spikes_df['spike_trial'] == i]
            counts_all[:, i] = np.bincount(np.array(x.index).astype(int), minlength=gids.max()+1)[-gids.size:]
        means = counts_all.mean(axis=1)
        vars = np.var(counts_all, ddof=1, axis=1)
        ffs = vars/means
        ffs[means == 0] = 0
        #print means
        spike_counts_df = pd.DataFrame({'mean':means, 'variance':vars, 'ff':ffs}, index=gids)
        spike_counts_df.to_pickle(file_name)
    else:
        spike_counts_df = pd.read_pickle(file_name)

    return spike_counts_df


def get_reliabilities_jitter(id_jitter=0):
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/jitter_%d_rel_v2.pkl'
    file_name = file_path % id_jitter
    if not os.path.isfile(file_name):
        bcs = get_jitter_blueconfigs(n=30, id_jitter=id_jitter)
        gids = np.array(get_selected_L456_gids().index)
        print gids
        spike_df_orig = get_spike_times_multiple(bcs)
        reliability_df = compute_reliabilities(spike_df_orig, gids, t_start=2000, t_end=7000)
        reliability_df.to_pickle(file_name)
    else:
        reliability_df = pd.read_pickle(file_name)
    return reliability_df


def get_spike_counts_jitter(n=30, id_jitter=0):
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/jitter_counts_%d_v2.pkl'
    file_name = file_path % id_jitter
    if not os.path.isfile(file_name):
        bcs = get_jitter_blueconfigs(n=30, id_jitter=id_jitter)
        df = get_spike_times_multiple(bcs)
        df = df[df['spike_time'] >= 2000]
        df = df[df['spike_time'] < 7000]
        sim = bluepy.Simulation(bcs[0])
        gids = np.sort(np.array(list(sim.get_circuit_target())))
        spikes_df = df.drop(columns=['spike_time'])
        counts_all = np.zeros((gids.size, n), dtype=np.float64)
        for i in range(n):
            x = spikes_df[spikes_df['spike_trial'] == i]
            counts_all[:, i] = np.bincount(np.array(x.index).astype(int), minlength=gids.max()+1)[-gids.size:]
        means = counts_all.mean(axis=1)
        vars = np.var(counts_all, ddof=1, axis=1)
        ffs = vars/means
        ffs[means == 0] = 0
        #print means
        spike_counts_df = pd.DataFrame({'mean':means, 'variance':vars, 'ff':ffs}, index=gids)
        spike_counts_df.to_pickle(file_name)
    else:
        spike_counts_df = pd.read_pickle(file_name)

    return spike_counts_df


def get_spike_counts_jitter_flick(n=30, bin_id=0):
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/jitter_counts_%d_flick_v2.pkl'
    file_name = file_path % bin_id
    times = np.arange(1000, 8500, 500)
    if not os.path.isfile(file_name):
        bcs = get_jitter_flick_blueconfigs(n=30)
        df = get_spike_times_multiple(bcs)
        df = df[df['spike_time'] >= times[bin_id]]
        df = df[df['spike_time'] < times[bin_id + 1]]
        sim = bluepy.Simulation(bcs[0])
        gids = np.sort(np.array(list(sim.get_circuit_target())))
        spikes_df = df.drop(columns=['spike_time'])
        counts_all = np.zeros((gids.size, n), dtype=np.float64)
        for i in range(n):
            x = spikes_df[spikes_df['spike_trial'] == i]
            counts_all[:, i] = np.bincount(np.array(x.index).astype(int), minlength=gids.max()+1)[-gids.size:]
        means = counts_all.mean(axis=1)
        vars = np.var(counts_all, ddof=1, axis=1)
        ffs = vars/means
        ffs[means == 0] = 0
        #print means
        spike_counts_df = pd.DataFrame({'mean':means, 'variance':vars, 'ff':ffs}, index=gids)
        spike_counts_df.to_pickle(file_name)
    else:
        spike_counts_df = pd.read_pickle(file_name)

    return spike_counts_df


def get_spike_counts_jitter_flick_control(n=30, bin_id=0):
    file_path = '/gpfs/bbp.cscs.ch/project/proj9/nolte/spike_times_variability/jitter_counts_%d_flick_control_v2.pkl'
    file_name = file_path % bin_id
    times = np.arange(1000, 8500, 500)
    if not os.path.isfile(file_name):
        bcs = get_jitter_flick_control_blueconfigs(n=30)
        df = get_spike_times_multiple(bcs)
        df = df[df['spike_time'] >= times[bin_id]]
        df = df[df['spike_time'] < times[bin_id + 1]]
        sim = bluepy.Simulation(bcs[0])
        gids = np.sort(np.array(list(sim.get_circuit_target())))
        spikes_df = df.drop(columns=['spike_time'])
        counts_all = np.zeros((gids.size, n), dtype=np.float64)
        for i in range(n):
            x = spikes_df[spikes_df['spike_trial'] == i]
            counts_all[:, i] = np.bincount(np.array(x.index).astype(int), minlength=gids.max()+1)[-gids.size:]
        means = counts_all.mean(axis=1)
        vars = np.var(counts_all, ddof=1, axis=1)
        ffs = vars/means
        ffs[means == 0] = 0
        #print means
        spike_counts_df = pd.DataFrame({'mean':means, 'variance':vars, 'ff':ffs}, index=gids)
        spike_counts_df.to_pickle(file_name)
    else:
        spike_counts_df = pd.read_pickle(file_name)

    return spike_counts_df


def get_all_spike_counts_experiment_25():
    spike_count_dfs = {}
    for sim_type in sim_types:
        for ca in cas_exp_25[sim_type]:
            print sim_type
            print ca
            spike_count_dfs[sim_type + '_' + ca] = get_spike_counts_experiment_25(ca=ca, sim_type=sim_type)
    return spike_count_dfs


def get_all_reliabilities_experiment_25():
    reliability_dfs = {}
    for sim_type in sim_types:
        for ca in cas_exp_25[sim_type]:
            reliability_dfs[sim_type + '_' + ca] = get_reliabilities_experiment_25(ca=ca, sim_type=sim_type)
    return reliability_dfs


def get_all_reliabilities_experiment_25_mp():
    k = int(sys.argv[1])
    names = []
    n = 0
    for sim_type in sim_types:
    #sim_type = 'cloud_mtype'
        for ca in cas_exp_25[sim_type]:
                names.append([sim_type, ca])
                n += 1
    print k
    print n
    #if k < n:
    get_reliabilities_experiment_25(ca=names[k][1], sim_type=names[k][0])
    #else:
    #    raise Exception("Sim does not exist.")


def get_reliabilities_experiment_25_original(n_classes=30, grouping_id=0):
    """
    Function to load and return spike times for exp 25 n 30 id 0
    """
    file_path = '/gpfs/bbp.cscs.ch/home/nolte/simplified_analysis/rel_data/exp_25_n_classes%d_grouping_id%d_v2.pkl'
    file_name = file_path % (n_classes, grouping_id)
    if not os.path.isfile(file_name):
        bcs = get_exp_25_blueconfigs_original(n=30, n_classes=n_classes, grouping_id=grouping_id)
        gids = np.array(get_selected_L456_gids().index)
        print gids
        spike_df_orig = get_spike_times_multiple(bcs)
        reliability_df = compute_reliabilities(spike_df_orig, gids)
        reliability_df.to_pickle(file_name)
    else:
        reliability_df = pd.read_pickle(file_name)
    return reliability_df


def get_spike_counts_experiment_25_original(n_classes=30, grouping_id=0, n=30):
    bcs = get_exp_25_blueconfigs_original(n=30, n_classes=n_classes, grouping_id=grouping_id)
    df = get_spike_times_multiple(bcs)
    df = df[df['spike_time'] >= 1000]
    df = df[df['spike_time'] < 6000]
    sim = bluepy.Simulation(bcs[0])
    gids = np.sort(np.array(list(sim.get_circuit_target())))
    spikes_df = df.drop(columns=['spike_time'])
    counts_all = np.zeros((gids.size, n), dtype=np.float64)
    for i in range(n):
        x = spikes_df[spikes_df['spike_trial'] == i]
        counts_all[:, i] = np.bincount(np.array(x.index).astype(int), minlength=gids.max()+1)[-gids.size:]
    means = counts_all.mean(axis=1)
    vars = np.var(counts_all, ddof=1, axis=1)
    ffs = vars/means
    ffs[means == 0] = 0
    #print means
    spike_counts_df = pd.DataFrame({'mean':means, 'variance':vars, 'ff':ffs}, index=gids)
    return spike_counts_df


def get_all_reliabilities_experiment_25_original():
    reliability_dfs = {}
    for n_class in n_classes:
        for grouping_id in grouping_ids:
            reliability_dfs['n_classes%d_grouping_id%d' % (n_class, grouping_id)] = get_reliabilities_experiment_25_original(n_classes=n_class, grouping_id=grouping_id)
    return reliability_dfs


def get_all_reliabilities_experiment_25_original_mp():
    k = int(sys.argv[1])
    names = []
    n = 0
    for n_class in n_classes:
        for grouping_id in grouping_ids:
            names.append([n_class, grouping_id])
            n += 1
    print k
    print n
    #if k < n:
    get_reliabilities_experiment_25_original(n_classes=names[k][0], grouping_id=names[k][1])
    #else:
    #    raise Exception("Sim does not exist.")


def get_all_spike_counts_experiment_25_original():
    spike_count_dfs = {}
    for n_class in n_classes:
        for grouping_id in grouping_ids:
            spike_count_dfs['n_classes%d_grouping_id%d' % (n_class, grouping_id)] = get_spike_counts_experiment_25_original(n_classes=n_class, grouping_id=grouping_id)
    return spike_count_dfs


def get_all_reliabilities_jitter_mp():
    k = int(sys.argv[1])
    get_reliabilities_jitter(id_jitter=ids_jitter[int(k)])

def get_all_reliabilities_jitter():
    dfs = {}
    for id_jitter in ids_jitter:
        dfs[id_jitter] = get_reliabilities_jitter(id_jitter=id_jitter)
    return dfs


def get_all_spike_counts_jitter():
    spike_count_dfs = {}
    for x in ids_jitter:
            spike_count_dfs[x] = get_spike_counts_jitter(id_jitter=x)
    return spike_count_dfs

def get_selected_L456_gids():
    circuit = bluepy.Simulation(get_exp_25_blueconfigs_cloud(n=1)[0]).circuit
    cells = circuit.v2.cells({Cell.MINICOLUMN: range(620, 650)})
    print len(cells.index)
    return cells[cells['layer'] > 3]


def get_mc_2_gids():
    circuit = bluepy.Simulation(get_exp_25_blueconfigs_cloud(n=1)[0]).circuit
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    print len(cells.index)
    return cells


def get_all_correlations_experiment_25():
    circuits = ['control', 'cloud_mtype_exc']
    corrs_dict = {}
    for circuit in circuits:
        for ca in cas_comparison:
            print circuit
            print ca
            corrs, gids = get_correlations_experiment_25(ca=ca, sim_type=circuit)
            corrs_dict[circuit + '_' + ca] = (corrs, gids)
    return corrs_dict

def get_all_correlations_experiment_25_shift(t_bin=20):
    circuits = ['control', 'cloud_mtype_exc']
    corrs_dict = {}
    for circuit in circuits:
        for ca in cas_comparison:
            print circuit
            print ca
            corrs, gids = get_correlations_experiment_25_with_shift(ca=ca, sim_type=circuit, t_bin=t_bin)
            corrs_dict[circuit + '_' + ca] = (corrs, gids)
    return corrs_dict


if __name__ == "__main__":
    get_all_reliabilities_experiment_25_mp()
    get_all_reliabilities_experiment_25_original_mp()
    get_all_spike_counts_experiment_25()
    get_all_reliabilities_jitter_mp()
    get_all_spike_counts_jitter()
    get_all_correlations_experiment_25()
    get_all_correlations_experiment_25_shift(t_bin=5)

