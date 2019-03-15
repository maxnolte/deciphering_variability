
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import pandas as pd
import data_access_shuffling
import bluepy
from bluepy.v2 import Cell

def generate_random_spike_train(t_start=2000, t_end=7000, fr=2, n=40, randomseed=33, jitter=0):
    np.random.seed(randomseed)
    spikes = np.random.uniform(low=t_start, high=t_end, size=(t_end - t_start)/1000 * fr)
    gid_start = 221042
    dict_spikes = {}
    for i in range(n):
        dict_spikes[gid_start + i] = spikes + np.random.randn(spikes.size) * jitter
    return dict_spikes


def plot_random_spike_trains():

    jitters = [0, 2, 5, 20, 50, 200]

    fig, axs = plt.subplots(len(jitters))
    for i, jitter, in enumerate(jitters):
        spike_dict = generate_random_spike_train(jitter=jitter)
        ax = axs[i]
        for key in spike_dict.keys():
            print spike_dict[key]
            ax.vlines(spike_dict[key], key, key+1)
        ax.set_xlim([1000, 7000])
    plt.savefig('figures/input_jitter.pdf')


def generate_random_spike_train_with_flick(t_start_1=1000, t_end_1=8000, fr_1=4, n_1=60, jitter_1=50,
                                           t_start_2=5000, t_end_2=8000, fr_2=4, n_2=40, jitter_2=2, randomseed_2=55,
                                           randomseed_1=0):

    np.random.seed(randomseed_2)
    gid_start = 221042
    gids = np.arange(0, n_1 + n_2) + gid_start
    gids = np.random.permutation(gids)
    spikes_2 = np.random.uniform(low=t_start_2, high=t_end_2, size=(t_end_2 - t_start_2)/1000 * fr_2)
    print spikes_2

    np.random.seed(randomseed_1 + 567)
    size=np.random.poisson((t_end_1 - t_start_1)/1000 * fr_1)
    spikes_1 = np.random.uniform(low=t_start_1, high=t_end_1, size=size)

    dict_spikes = {}
    for i in range(n_1):
        dict_spikes[gids[i]] = spikes_1 + np.random.randn(spikes_1.size) * jitter_1
    for i in range(n_2):
        dict_spikes[gids[i + n_1]] = spikes_2 + np.random.randn(spikes_2.size) * jitter_2
    return dict_spikes


def plot_random_spike_trains_with_flick():

    jitters = np.arange(30)

    fig, axs = plt.subplots(len(jitters), figsize=(14, 14))
    for i, jitter, in enumerate(jitters):
        spike_dict = generate_random_spike_train_with_flick(randomseed_1=i)
        ax = axs[i]
        for key in spike_dict.keys():
            print spike_dict[key]
            ax.vlines(spike_dict[key], key, key+1)
        ax.set_xlim([1000, 8000])
    plt.savefig('figures/input_jitter_flick.pdf')


def save_random_spike_trains_with_flick():

    file_name = '/gpfs/bbp.cscs.ch/home/nolte/simplified_analysis/input_spike_trains_jitter_flick/random_jitter_flick_seed_%d.dat'

    for i in range(30):
        spike_dict = generate_random_spike_train_with_flick(randomseed_1=i)
        stim_file = file_name % i
        with open(stim_file, 'w') as f:
            f.write('/scatter\n')
            for key in np.sort(spike_dict.keys()):
                for spike_time in spike_dict[key]:
                    f.write('%4.2f\t%d\n' % (spike_time, key))
    print "Done"


def save_random_spike_trains():

    n = 40
    seed = 33
    jitters = [0, 2, 5, 20, 50, 200]

    file_name = '/gpfs/bbp.cscs.ch/home/nolte/simplified_analysis/input_spike_trains/random_jitter_%d_n_%d_seed_%d.dat'

    for i, jitter, in enumerate(jitters):
        spike_dict = generate_random_spike_train(jitter=jitter, n=40, randomseed=33)
        stim_file = file_name % (jitter, n, seed)
        with open(stim_file, 'w') as f:
            f.write('/scatter\n')
            for key in np.sort(spike_dict.keys()):
                for spike_time in spike_dict[key]:
                    f.write('%4.2f\t%d\n' % (spike_time, key))
    print "Done"


def plot_jittered_rasters():
    path = "/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/jittering/stimulusstim_a%d/seed%d/BlueConfig"
    seeds = np.arange(10000, 10003, dtype=int)

    sim = bluepy.Simulation(data_access_shuffling.get_spontaneous_blueconfigs_cloud(n=1)[0])
    cells = sim.circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    ys = np.array(cells['y'])
    gids = np.array(list(sim.get_circuit_target()))
    sort_idx = np.argsort(ys)
    sort_dict = dict(zip(sort_idx + gids.min(), np.arange(gids.size)))
    stims = np.array([5, 0, 2, 4, 1, 3])
    jitters = np.array([2, 50, 5, 200, 20, 0])[[5, 0, 2, 4, 1, 3]]

    fig, axs = plt.subplots(18, figsize=(10, 30))
    for j, stim in enumerate(stims):

        for i, seed in enumerate(seeds):
            ax = axs[j*3 + i]
            bc = path % (stim, seed)
            spike_times, spike_gids = data_access_shuffling.get_spike_times_sim(bc, t_start=1000)
            print spike_times
            spike_gids = np.vectorize(sort_dict.get)(spike_gids)
            ax.vlines(spike_times, spike_gids, spike_gids + 100, rasterized=True, lw=0.1)
            ax2 = ax.twinx()
            ax2.hist(spike_times, bins=np.linspace(1000, 7000, 601), histtype='step',
                         weights=np.zeros(spike_times.size) + (1000 / 10.0) / gids.size)
            ax2.set_ylabel('FR (Hz)')
            ax.set_ylabel('Neurons')
            ax.set_xlabel('t (ms)')
            ax.set_ylim([0, gids.max() - gids.min()])
            ax.set_xlim([1000, 7000])
            #axs[i, j].set_title('S%db - %s' % (nclasses, labels[i]))
        #break
    plt.tight_layout()
    plt.savefig('figures/raster_jitter.pdf', dpi=300)


dict = generate_random_spike_train_with_flick()
plot_random_spike_trains_with_flick()
save_random_spike_trains_with_flick()
plot_jittered_rasters()
plot_random_spike_trains()
save_random_spike_trains()
plot_random_spike_trains_with_flick()
