import bluepy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
import correlations
import os
import initial_analysis_final as iaf
from bluepy.v2 import Cell

reyes_puerta = '/gpfs/bbp.cscs.ch/project/proj1/simulations/ReNCCv3/InVivo/03_Reyes-Puerta/K5p0/Ca1p25/minicols%d/seed%d/BlueConfig'
exp_25 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/synchrony/experiment_25/seed%d/n_classes%d/grouping_id%d/BlueConfig'


def get_correlations_puerta(n_vpm=60, dt=10.0):
    """
    Corrrelations between whisker flick sims, no restore
    :param n_vpm:
    :param dt:
    :return:
    """

    file = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + ('/corrs_reyes_puerta_%d.npz' % n_vpm)

    if not os.path.isfile(file):
        bcs_1 = [(reyes_puerta % (n_vpm, seed)) for seed in range(20)]
        bcs_2 = [(reyes_puerta % (n_vpm, seed)) for seed in range(20, 40)]

        corrs, bins = compute_soma_correlations(bcs_1, bcs_2, dt=dt, t_start=1400.0, t_end=1900.0)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def get_correlations_exp_25(n_classes=30, grouping=1, dt=10.0):
    """
    correlations between exp 25, no restore
    :param n_classes:
    :param grouping:
    :param dt:
    :return:
    """

    file = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + ('/corrs_exp_25_n%d_g%d.npz' % (n_classes, grouping))

    if not os.path.isfile(file):
        bcs_1 = [(exp_25 % (seed, n_classes, grouping)) for seed in range(15)]
        bcs_2 = [(exp_25 % (seed, n_classes, grouping)) for seed in range(15, 30)]

        corrs, bins = compute_soma_correlations(bcs_1, bcs_2, dt=dt, t_start=950.0, t_end=5900.0)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def compute_all_evoked_correlations():
    for shuffle in [0, 1]:
        for dt in [10, 5]:
            for shift in [0, 20, 200]:
                print shuffle
                print dt
                print shift
                get_correlations_puerta_same(dt=dt, shift=shift, shuffle=shuffle)

def get_correlations_puerta_same(dt=10.0, shift=0, shuffle=0, middle=False):
    return get_correlations_x(dt=dt, shift=shift, shuffle=shuffle, stimulus='reyes_puerta', middle=middle)

def get_correlations_exp_25_same(dt=10.0, shift=0, shuffle=0, middle=False):
    return get_correlations_x(dt=dt, shift=shift, shuffle=shuffle, stimulus='exp_25', middle=middle)

def get_correlations_x(dt=10.0, shift=0, shuffle=0, stimulus='reyes_puerta', middle=False):
    """
    restored variability
    :param dt:
    :return:
    """

    shift_string = ''
    shift_string_2 = ''
    shift_string_3 = ''
    shuffle_string = ''
    if shuffle > 0:
        shuffle_string = '_different_state'
    if stimulus == 'exp_25' or shift > 0:
        shift_string = '_shift'
        shift_string_2 = 'shift%d/' % shift
        shift_string_3 = '_shift%d' % shift
    middle_string = ''
    if middle:
        middle_string = '_middle'
    file = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + '/corrs_' + stimulus + middle_string + shift_string + '_abcd' + shift_string_3 + shuffle_string + ('_dt%d.npz' % dt)
    print file

    if not os.path.isfile(file):
        bcs_1 = [('/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/evoked/continue_change_' + stimulus + middle_string + shift_string + '_abcd/' + shift_string_2 + 'seed%d/BlueConfig') % seed for seed in range(170, 190)]
        bcs_2 = [('/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/evoked/continue_change_'  + stimulus + middle_string + shift_string + '_x/' + shift_string_2 + 'seed%d/BlueConfig') % seed for seed in range(170, 190)]
        if shuffle > 0:
            bcs_2 = [bcs_2[-1]] + bcs_2[:-1]

        corrs, bins = compute_soma_correlations(bcs_1, bcs_2, dt=dt, t_end=3500.0)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def get_correlations_base(dt=10.0, middle=False):
    """
    restored variability
    :param dt:
    :return:
    """

    middle_string = ''
    if middle:
        middle_string = '_middle'
    file = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + '/corrs' + middle_string + '_base_dt%d.npz' % dt
    print file

    folder = '/spontaneous/base_seeds_abcd/'
    if middle:
        folder = '/evoked/base_seeds_exp_25/'

    if not os.path.isfile(file):
        bcs_1 = [('/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability' + folder + 'seed%d/BlueConfig') % seed for seed in range(170, 190)]
        bcs_2 = [bcs_1[-1]] + bcs_1[:-1]

        corrs, bins = compute_soma_correlations(bcs_1, bcs_2, dt=dt, t_start=1950.0, t_end=None)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def get_bcs_x(shift=0, stimulus='reyes_puerta', seed=170, middle=False):
    """
    restored variability
    :param dt:
    :return:
    """

    shift_string = ''
    shift_string_2 = ''
    if stimulus == 'exp_25' or shift > 0:
        shift_string = '_shift'
        shift_string_2 = 'shift%d/' % shift
    middle_string = ''
    base_string = 'spontaneous/base_seeds_abcd'
    if middle:
        middle_string = '_middle'
        base_string = 'evoked/base_seeds_exp_25'
    bc_0 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/' + base_string + '/seed%d/BlueConfig' % seed
    bc_1 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/evoked/continue_change_' + stimulus + middle_string + shift_string + '_abcd/' + shift_string_2 + ('seed%d/BlueConfig' % seed)
    bc_2 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/evoked/continue_change_'  + stimulus + middle_string + shift_string + '_x/' + shift_string_2 + ('seed%d/BlueConfig' % seed)

    return bc_0, bc_1, bc_2


def get_correlations_puerta_different(dt=10.0, shift=0):
    """
    as a control
    :param dt:
    :return:
    """

    return get_correlations_puerta_same(dt=dt, shift=shift, shuffle=1)


def compute_soma_correlations(bcs_1, bcs_2, dt=5.0, t_start=None, t_end=1900.0):

    results = []
    for bc_1, bc_2 in zip(bcs_1, bcs_2):
        vm_continue, _ = correlations.get_soma_time_series(bc_1, t_start=t_start, t_end=t_end)
        vm_change, times = correlations.get_soma_time_series(bc_2, t_start=t_start, t_end=t_end)
        vm_continue = np.array(vm_continue)
        vm_change = np.array(vm_change)

        corrs = []
        for i, corr_func in enumerate([correlations.voltage_rmsd_from_data, correlations.voltage_correlation_from_data]):
            corr, bins = corr_func(vm_continue, vm_change, times, dt=dt)
            corrs.append(corr)
        corrs = np.dstack(corrs)
        print corrs.shape
        results.append(corrs)
    results = np.concatenate([a[..., np.newaxis] for a in results], axis=3)
    return results, bins

def plot_correlations_puerta():
    values, times_1 = get_correlations_puerta_different(dt=10.0)
    values_same, times_2 = get_correlations_puerta_same(dt=10.0)
    values_no_stim, times_no_stim = iaf.get_correlations(parameter_continue='', parameter_change='abcd', dt=10.0)
    print times_2.shape
    print times_1.shape
    print times_no_stim.shape

    fig, axs = plt.subplots(2)
    ax = axs[0]
    ax.set_ylabel('r')

    n = 21
    means = values.mean(axis=0).mean(axis=-1)[:, 1]
    errs = values.mean(axis=0).std(axis=-1, ddof=1)[:, 1]/np.sqrt(20)

    means = np.insert(means, 0, means[-1])
    errs = np.insert(errs, 0, errs[-1])

    means_no_stim = values_no_stim.mean(axis=0).mean(axis=-1)[:, 1]
    errs_no_stim = values_no_stim.mean(axis=0).std(axis=-1, ddof=1)[:, 1]/np.sqrt(20)

    means_no_stim = np.insert(means_no_stim, 0, 1)
    errs_no_stim = np.insert(errs_no_stim, 0, 0)

    ax.errorbar(times_1[:n] - 5.0, means[:n], yerr=errs[:n])

    means_2 = values_same.mean(axis=0).mean(axis=-1)[:, 1]
    errs_2 = values_same.mean(axis=0).std(axis=-1, ddof=1)[:, 1]/np.sqrt(20)

    means_2 = np.insert(means_2, 0, 1)
    errs_2 = np.insert(errs_2, 0, 0)

    ax.errorbar(times_2[:n] - 5.0, means_2[:n], yerr=errs_2[:n])
    ax.errorbar(times_1[:n] - 5.0, means_no_stim[:n], yerr=errs_no_stim[:n], color='black', linestyle='--')

    ax = axs[1]
    ax.set_ylabel('r norm.')

    scale = means_no_stim[100:].mean()
    errs_no_stim = errs_no_stim / (1.0 - scale)
    means_no_stim = (means_no_stim - scale) / (1 - scale)

    scale = (means_2 - means)[100:].mean()
    scale_2 = (means_2 - means)[100:].mean() + 1 - (means_2 - means)[0].mean()
    errs_2 = np.linalg.norm(np.vstack([errs_2, errs]), axis=0) / (1.0 - scale_2)
    means_2 = (means_2 - means - scale) / (1 - scale_2)

    ax.errorbar(times_2[:n] - 5.0, means_2[:n], yerr=errs_2[:n])
    ax.errorbar(times_1[:n] - 5.0, means_no_stim[:n], yerr=errs_no_stim[:n], color='black', linestyle='--')

    ax.plot(times_2[:n] - 5.0, np.zeros(times_2[:n].size))

    for ax in axs:
        ax.set_ylim([0, 1])

    plt.savefig('figures/reyes_puerta.pdf')



def plot_correlations_puerta_multiple(stimulus='puerta', shifts=[0, 20, 200], middles=[False, False, False]):
# def plot_correlations_puerta_multiple(stimulus='exp_25', shifts=[100, 0, 50, 100, 200], middles=[True, False, False, False, False]):

    means = {}

    dt = 5.0
    for m, shift in enumerate(shifts):
        for shuffle in [0, 1]:
            if stimulus == 'puerta':
                corrs, bins = get_correlations_puerta_same(dt=dt, shift=shift, shuffle=shuffle)
            elif stimulus == 'exp_25':
                corrs, bins = get_correlations_exp_25_same(dt=dt, shift=shift, shuffle=shuffle, middle=middles[m])

            bins -= dt/2 + 2000
            bins[0] = 0

            for index_correlation in range(2):
                p = '%d-%d-%d-%d' % (shuffle, shift, index_correlation, middles[m])
                means_p = corrs.mean(axis=0)[:, index_correlation, :]
                first_mean = np.zeros((means_p[0, :].shape)) + index_correlation
                if shuffle:
                    pre_corrs, _ = get_correlations_base(dt=5.0, middle=middles[m])
                    first_mean = pre_corrs.mean(axis=0)[-1, index_correlation, :]
                    first_mean = (means_p[0, :] + first_mean)/2.0
                means[p] = np.vstack([first_mean[None, :], means_p])

    n = 80

    colors = ['#d53e4f', '#3288bd']

    fig, axs = plt.subplots(len(shifts) + 1, 2, figsize=(8, (8/6.0) * (len(shifts) + 1) ))
    n_start = 0
    for index_correlation in range(2):
        for shuffle in [0, 1]:
            for j, shift in enumerate(shifts):
                p = '%d-%d-%d-%d' % (shuffle, shift, index_correlation, middles[j])
                ax = axs[j, index_correlation]

                corrs = means[p]
                mean_corrs = corrs.mean(axis=1)[n_start:(n+1)]
                err_corrs = np.apply_along_axis(iaf.mean_confidence_interval, -1, corrs)[n_start:(n+1)]

                ax.fill_between(bins[n_start:(n+1)],  mean_corrs - err_corrs,
                                mean_corrs + err_corrs,
                                facecolor=colors[shuffle], alpha=0.3)
                ax.plot(bins[n_start:(n+1)], mean_corrs,
                                linewidth=0.8, markersize=3,
                                color=colors[shuffle])
                ax.set_ylim([[0, 15], [0, 1]][index_correlation])
                ax.set_ylabel(['RMS', 'Corr'][index_correlation])

    n = 30
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
    for index_correlation in range(2):
        ax = axs[-1, index_correlation]
        for j, shift in enumerate(shifts):
            p_0 = '%d-%d-%d-%d' % (0, shift, index_correlation, middles[j])
            p_1 = '%d-%d-%d-%d' % (1, shift, index_correlation, middles[j])


            corrs = [-1.0, 1.0][index_correlation]*(means[p_0][n_start:(n+1)]-means[p_1][n_start:(n+1)])
            mean_corrs = corrs.mean(axis=1)[n_start:(n+1)]
            err_corrs = np.apply_along_axis(iaf.mean_confidence_interval, -1, corrs)[n_start:(n+1)]


            mean_corrs_norm = mean_corrs / mean_corrs[0]

            err_corrs_norm = np.abs(mean_corrs_norm) * np.sqrt((err_corrs/mean_corrs)**2 + (err_corrs[0]/mean_corrs[0])**2)

            ax.errorbar(bins[n_start:(n+1)], mean_corrs_norm, yerr=err_corrs_norm,
                                linewidth=0.8, label=shift, color=colors[j])
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/puerta_multiple.pdf')



def plot_correlations_exp_25():
    values, _ = get_correlations_exp_25(n_classes=30, grouping=1, dt=10.0)
    values_20, times = get_correlations_exp_25(n_classes=30, grouping=1, dt=10.0)
    print values.shape
    print times.shape

    means = values.mean(axis=0).mean(axis=-1)[:, 1]
    errs = values.mean(axis=0).std(axis=-1, ddof=1)[:, 1]/np.sqrt(15)

    fig, ax = plt.subplots()
    ax.errorbar(times[1:51] - 5.0, means[:50], yerr=errs[:50])

    plt.savefig('figures/exp_25.pdf')


def plot_soma_reyes_puerta():

    bc = reyes_puerta % (60, 0)
    voltage, times = correlations.get_soma_time_series(bc, t_end=1900, t_start=1400)

    bc = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/evoked/continue_change_reyes_puerta_x/seed170/BlueConfig'
    voltage_20, _ = correlations.get_soma_time_series(bc, t_end=2500)

    voltage_2, times = correlations.get_soma_time_series(exp_25 % (0, 30, 1), t_end=1400, t_start=900)

    print voltage.shape

    fig, ax = plt.subplots()
    ax.plot(times, voltage.mean(axis=0))
    ax.plot(times + 100, voltage_20.mean(axis=0))

    ax.plot(times, voltage_2.mean(axis=0))

    plt.savefig('figures/soma_voltage_reyes.pdf')


def plot_spike_raster_example_evoked():
    """
    plot for paper
    :return:
    """

    # params = [['exp_25', 'exp_25']]
    # shifts = [50, 100] # 50
    # n_seeds = [187, 189]
    # middles = [False, True]

    params = [['reyes_puerta', 'reyes_puerta']]
    shifts = [0, 20] # 50
    n_seeds = [188, 175]
    middles = [False, False]


    t_middle = 2000
    t_window = 350
    t_window_start = 150
    fig, axs = plt.subplots(4, 2, figsize=(10, 6))

    for k in range(2):
        n_seed = n_seeds[k]
        base_bc, continue_bc, change_bc = get_bcs_x(stimulus=params[0][k], shift=shifts[k], seed=n_seed, middle=middles[k])
        print base_bc
        print continue_bc
        print change_bc
        circuit = bluepy.Simulation(base_bc).circuit
        cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
        ys = np.array(cells['y'])
        gids = np.array(list(bluepy.Simulation(base_bc).get_circuit_target()))

        sort_idx = np.argsort(ys)

        sort_dict = dict(zip(sort_idx + gids.min(), np.arange(gids.size)))

       # sort_dict = dict(zip(np.arange(len(sort_idx), dtype=int), ys[::-1]))

        ax = axs[0,k]
        spikes = bluepy.Simulation(base_bc).v2.reports['spikes']
        df = spikes.data(t_start=t_middle - t_window_start)
        gids_spiking = np.array(df.axes[0])
        print "-----"
        print gids_spiking.max()
        print gids_spiking.min()
        times = np.array(df) - t_middle

        print gids_spiking
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        print gids_spiking
        #times =  np.vectorize(sort_dict.get)(times)

        ax.vlines(times, gids_spiking, gids_spiking + 70, rasterized=True, lw=0.15)
        ax2 = ax.twinx()
        ax2.hist(times, bins=np.linspace(-t_window_start, 0, 31), histtype='step', weights=np.zeros(times.size) + (1000.0/5.0)/gids.size)
        ax2.set_ylabel('FR (Hz)')

        spikes = bluepy.Simulation(continue_bc).v2.reports['spikes']
        df = spikes.data(t_end=t_window + t_middle * middles[k])
        gids_spiking = np.array(df.axes[0])
        print gids_spiking
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        print gids_spiking
        times = np.array(df) - t_middle * middles[k]

        ax.vlines(times, gids_spiking, gids_spiking + 70, rasterized=True, lw=0.15)
        ax2.hist(times, bins=np.linspace(0, t_window, 71), histtype='step', weights=np.zeros(times.size) + (1000.0/5.0)/gids.size)

        ax = axs[1,k]
        spikes = bluepy.Simulation(base_bc).v2.reports['spikes']
        df = spikes.data(t_start=t_middle - t_window_start)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)

        times = np.array(df) - t_middle
        ax.vlines(times, gids_spiking, gids_spiking + 70, rasterized=True, lw=0.15)
        ax2 = ax.twinx()
        ax2.hist(times, bins=np.linspace(-t_window_start, 0, 31), histtype='step', weights=np.zeros(times.size) + (1000.0/5.0)/gids.size)
        ax2.set_ylabel('FR (Hz)')

        spikes = bluepy.Simulation(change_bc).v2.reports['spikes']
        df = spikes.data(t_end=t_window + t_middle * middles[k])
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)

        times = np.array(df) - t_middle * middles[k]
        ax.vlines(times, gids_spiking, gids_spiking + 70, rasterized=True, lw=0.15)
        ax2.hist(times, bins=np.linspace(0, t_window, 71), histtype='step', weights=np.zeros(times.size) + (1000.0/5.0)/gids.size)

        ax = axs[2, k]
        bc = base_bc
        soma = bluepy.Simulation(bc).v2.reports['soma']
        time_range = soma.time_range[soma.time_range >= t_middle - t_window] - t_middle
        data = soma.data(t_start=t_middle - t_window)
        ax.plot(time_range, data.mean(axis=0), linewidth=1, alpha=0.5, color='#1f77b4')
        bc = continue_bc
        soma = bluepy.Simulation(bc).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - t_middle
        data_cont = soma.data(t_end=t_middle + t_window)
        ax.plot(time_range_cont, data_cont.mean(axis=0), linewidth=1, alpha=0.5, color='#ff7f0e')
        bc = change_bc
        soma = bluepy.Simulation(bc).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - t_middle
        data_cont = soma.data(t_end=t_middle + t_window)
        ax.plot(time_range_cont, data_cont.mean(axis=0), linewidth=1, alpha=0.5, color='#ff7f0e')



    for ax in axs[:2, :].flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylim([0, 31346])
        ax.set_yticks([0, 10000, 20000, 30000])

        ax.set_ylabel('Neuron')
        ax.set_xlim([-t_window_start, t_window])

    for ax in axs[2, :].flatten():
        ax.set_xlabel('t (ms)')
        #ax.set_ylim([-63, -60])
        #ax.set_yticks([-63, -62, -61, -60])

        ax.set_ylabel('V (mV)')
        ax.set_xlim([-t_window_start, t_window])

    plt.tight_layout()
    plt.savefig('figures/raster_evoked.pdf', dpi=300)



if __name__ == "__main__":
   plot_correlations_puerta()
   plot_correlations_exp_25()
   plot_soma_reyes_puerta()
   plot_correlations_puerta_multiple()
   plot_spike_raster_example_evoked()