import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import multiprocessing as mp
import correlations
import os
import bluepy
from connection_matrices.generate.gen_con_mats import connections
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import scipy.stats
import numpy.linalg
from bluepy.v2 import Cell
import analysis_noise_stim


base_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_%s/seed%d/BlueConfig'
continue_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_base_seeds%s/seed%d/BlueConfig'
change_path = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_change_%s/seed%d/BlueConfig'

base_path_decouple = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/base_seeds_abcd/seed%d/BlueConfig'
continue_path_decouple = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_x/seed%d/BlueConfig'
continue_path_decouple_nondec = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_x/seed%d/BlueConfig'
change_path_decouple = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_%s/seed%d/BlueConfig'
change_path_decouple_nondec = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_%s/seed%d/BlueConfig'

continue_path_decouple_stim = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_stim/change_x/seed%d/variance%s/BlueConfig'
change_path_decouple_stim = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_stim/change_%s/seed%d/variance%s/BlueConfig'

base_path_ca = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_%s/seed%d/Ca%s/BlueConfig'
continue_path_ca = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_base_seeds%s/seed%d/Ca%s/BlueConfig'
change_path_ca = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_change_%s/seed%d/Ca%s/BlueConfig'

base_path_mvr = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_%s_mvr_scan/nrrp%s/seed%d/BlueConfig'
continue_path_mvr = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_base_seeds_%smvr_scan/nrrp%s/seed%d/BlueConfig'
change_path_mvr = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_change_mvr_scan_%s/nrrp%s/seed%d/BlueConfig'

base_path_stim = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_%s/seed%d/variance%s/BlueConfig'
continue_path_stim = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_base_seeds%s/seed%d/variance%s/BlueConfig'
change_path_stim = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/continue_change_%s/seed%d/variance%s/BlueConfig'

seeds = np.arange(170, 190)
seeds_2 = np.delete(np.arange(150, 170), 17)


base_parameters = ['abcd',
                   'gbcd']

n_base_sims = [20,
               20]

base_parameters_continue = ['', '_g']
n_continue_sims = [20, 20]

parameters=[['a',
            'ab',
            'abcd',
            'abc',
            'abd',
            'abcde',
            'abcdef',
            'b',
            'bcd',
            'c',
            'cd',
            'd',
            'e',
            'g',
            'f',
            'abcdf',
            'y',
            'abcdy',
            'abcdey'],

                ['gv2b',
                 'gv2bc',
                 'gv2bcd',
                 'gv2bcdef',
                 'gv2bcdf',
                 'gv2c',
                 'gv2cd',
                 'gv2d',
                 'gv2e',
                 'gv2g',
                 'gv2f']]

parameters_decouple = ['ab', 'abcd', 'cd', 'e', 'f', 'x']


names=    [['Det. minis, channels, stimulus (a)',
            'Det. channels, stimulus (ab)',
            'Standard model (abcd)',
            'abc',
            'abd',
            'abcde',
            'abcdef',
            'b',
            'Pseudo-deterministic release, stochastic minis (bcd)',
            'c',
            'Pseudo-deterministic release, deterministic minis (cd)',
            'd',
            'e',
            'g',
            'f',
            'abcdf',
            'y',
            'abcdy',
            'abcdey'],
                ['gv2b',
                 'gv2bc',
                 'Deterministic release, stochastic minis (gbcd)',
                 'gv2bcdef',
                 'gv2bcdf',
                 'gv2c',
                 'Deterministic release, deterministic minis (gcd)',
                 'gv2d',
                 'gv2e',
                 'gv2g',
                 'gv2f']]

cas = ['1p1', '1p2', '1p3']

names = dict(zip(parameters[0] + parameters[1], names[0] + names[1]))

linestyles=[['-' for x in parameters[0]], ['--' for x in parameters[1]]]
linestyles = dict(zip(parameters[0] + parameters[1], linestyles[0] + linestyles[1]))
linestyles['abcd'] = ':'

n_sims = [[10 for i in range(len(parameters[0]))], [10 for i in range(len(parameters[0]))]]


def get_soma_time_series(blueconfig, t_start=None, t_end=None, gids=None):
    ### OLD BLUEPY
    # soma = bluepy.Simulation(blueconfig).v2.reports['soma']
    # data = soma.data(t_start=t_start, t_end=t_end, gids=gids)
    # return data, data.axes[1]/1000.0

    ### NEW BLUEPY - dataframe has been transposed
    soma = bluepy.Simulation(blueconfig).v2.report('soma')
    data = soma.get(t_start=t_start, t_end=t_end, gids=gids)
    data = data.T
    return data, data.axes[1]

def get_base_bcs_decouple(params='abcd', n=19):
    return [base_path_decouple % seed for seed in seeds_2[:n]]

def get_continue_bcs_decouple(params='', n=19, decouple=True):
    if decouple:
        return [continue_path_decouple % seed for seed in seeds_2[:n]]
    else:
        return [continue_path_decouple_nondec % seed for seed in seeds_2[:n]]

def get_change_bcs_decouple(params='abcd', n=19, decouple=True):
    if decouple:
        return [change_path_decouple % (params, seed) for seed in seeds_2[:n]]
    else:
        return [change_path_decouple_nondec % (params, seed) for seed in seeds_2[:n]]


def get_continue_bcs_decouple_stim(params='', variance='1p0', n=10):
    return [continue_path_decouple_stim % (seed, variance) for seed in seeds_2[:n]]

def get_change_bcs_decouple_stim(params='d', variance='1p0', n=10):
    if params == 'abcd':
        return [change_path_decouple_stim % (params, seed, variance) for seed in np.arange(3150, 3160)]
    else:
        return [change_path_decouple_stim % (params, seed, variance) for seed in seeds_2[:n]]


def get_base_bcs(params='abcd', n=20):
    return [base_path % (params, seed) for seed in range(seeds[0], seeds[0] + n)]

def get_base_bcs_mvr():
    return [base_path_mvr % seed for seed in seeds]

def get_continue_bcs(params='', n=20):
    return [continue_path % (params, seed) for seed in range(seeds[0], seeds[0] + n)]

def get_change_bcs(params='abcd', n=20):
    return [change_path % (params, seed) for seed in range(seeds[0], seeds[0] + n)]

def get_base_bcs_ca(params='abcd', ca='1p1', n=20):
    return [base_path_ca % (params, seed, ca) for seed in range(seeds[0], seeds[0] + n)]

def get_continue_bcs_ca(params='', ca='1p1', n=20):
    return [continue_path_ca % (params, seed, ca) for seed in range(seeds[0], seeds[0] + n)]

def get_change_bcs_ca(params='abcd', ca='1p1', n=20):
    return [change_path_ca % (params, seed, ca) for seed in range(seeds[0], seeds[0] + n)]

def get_base_bcs_mvr(params='abcd', nrrp='2p0', n=20):
    return [base_path_mvr % (params, nrrp, seed) for seed in range(seeds[0], seeds[0] + n)]

def get_continue_bcs_mvr(params='', nrrp='2p0', n=20):
    return [continue_path_mvr % (params, nrrp, seed) for seed in range(seeds[0], seeds[0] + n)]

def get_change_bcs_mvr(params='abcd', nrrp='2p0', n=20):
    return [change_path_mvr % (params, nrrp, seed) for seed in range(seeds[0], seeds[0] + n)]

def get_base_bcs_stim(params='abcd_stim', variance='1p0', n=20):
    return [base_path_stim % (params, seed, variance) for seed in range(seeds[0], seeds[0] + n)]

def get_continue_bcs_stim(params='_stim', variance='1p0', n=20):
    return [continue_path_stim % (params, seed, variance) for seed in range(seeds[0], seeds[0] + n)]

def get_change_bcs_stim(params='abcd_stim', variance='1p0', n=20):
    return [change_path_stim % (params, seed, variance) for seed in range(seeds[0], seeds[0] + n)]

def get_base_simulation(seed=170):
    return base_path % ('abcd', seed)


def get_continue_simulation(seed=170):
    return continue_path % ('', seed)


def get_change_simulation(seed=170, parameter_change='abcd'):
    return change_path % ('abcd', seed)


def compute_all_shuffled_2():
    pre_compute_corrs(shuffle_level=2)
    compare_cas(shuffle_level=2)


def pre_compute_corrs(shuffle_level=0):
    get_all_abcd_correlations(shuffle_level=shuffle_level)
    get_all_gbcd_correlations_g(shuffle_level=shuffle_level)


def get_all_abcd_correlations(shuffle_level=0):
    for p in parameters[0]:
        print p
        for dt in [10.0, 20.0]:
            get_correlations(parameter_continue='', parameter_change=p, dt=dt, shuffle_level=shuffle_level)


def get_all_gbcd_correlations_g(shuffle_level=0):
    for p in parameters[1]:
        print p
        for dt in [10.0, 20.0]:
            get_correlations(parameter_continue='_gv2', parameter_change=p, dt=dt, shuffle_level=shuffle_level)


def compare_kernel_sizes():
    ks = [0, 1]
    for kernel in ks:
        get_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, shuffle_level=0, kernel=kernel)


def compare_cas(shuffle_level=0):
    for ca in cas:
        get_correlations(parameter_continue='_ca_abcd', parameter_change='ca_abcd', dt=10.0, shuffle_level=shuffle_level, ca=ca)


def compare_stims(shuffle_level=0):
    for variance in variances:
        print "Now computing variance = %s" % variance
        get_correlations(parameter_continue='_stim', parameter_change='abcd_stim', dt=10.0, shuffle_level=shuffle_level,
                         variance=variance)


def compute_decoupled():
    for p in parameters_decouple:
        get_correlations(parameter_continue='', parameter_change=p, dt=10.0, decouple=True)


def compute_mvr():
    for nrrp in [2, 3, 10]:
        get_correlations(parameter_continue='', parameter_change='abcd', dt=10.0, nrrp=nrrp)


def get_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, shuffle_level=0, kernel=0, ca=None,
                     decouple=False, n=20, variance=None, nrrp=1, nondecouple=0):

    folder = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + '/corrs' + parameter_continue + '_' + parameter_change
    if shuffle_level == 0:
        file = folder + '_dt%d' % dt
    elif shuffle_level >= 1:
        file = folder + '_dt%d_shuffle%d' % (dt, shuffle_level)
    if kernel >= 1:
        file += '_kernel%d' % kernel
    if ca is not None:
        file += '_ca%s' % ca
    if variance is not None:
        file += '_variance%s' % variance
    if decouple:
        file += '_decouple'
        if nondecouple > 0:
            file += '%d' % nondecouple

    if nrrp > 1:
        file += '_nrrp%d' % nrrp
    if not n == 20:
        file += '_n%d' % n
    file += '.npz'
    if not os.path.isfile(file):
        corrs, bins = compute_soma_correlations(parameter_continue=parameter_continue,
                                                parameter_change=parameter_change,
                                                shuffle_level=shuffle_level, kernel=kernel, dt=dt, ca=ca,
                                                decouple=decouple, n=n, variance=variance, nrrp=nrrp,
                                                nondecouple=nondecouple)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def compute_soma_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, t_end=3500.0, shuffle_level=0,
                              kernel=0, base_params='abcd', ca=None, decouple=False, n=20, variance=None, nrrp=1,
                              nondecouple=0, t_start=2000.000001):
    """
    ### WARNING t_start=2000.000001 is a recent addition for new bluepy version!


    :param parameter_continue:
    :param parameter_change:
    :return:
    """
    if ca is None and variance is None and nrrp <= 1 and not decouple:
        continue_bcs = get_continue_bcs(params=parameter_continue, n=n)[:n]
        change_bcs = get_change_bcs(params=parameter_change, n=n)[:n]
        base_bcs = get_base_bcs(params=base_params, n=n)[:n] # Won't work for MVR
    elif decouple:
        continue_bcs = get_continue_bcs_decouple(params=parameter_continue, n=n)[:n]
        change_bcs = get_change_bcs_decouple(params=parameter_change, n=n)[:n]
        base_bcs = get_base_bcs_decouple(params=base_params, n=n)[:n]
        if nondecouple > 0:
            continue_bcs = get_continue_bcs_decouple(params=parameter_continue, n=n, decouple=False)[:n]
            if nondecouple > 1:
                change_bcs = get_change_bcs_decouple(params=parameter_change, n=n, decouple=False)[:n]
        elif variance is not None:
            continue_bcs = get_continue_bcs_decouple_stim(params=parameter_continue, n=n, variance=variance)[:n]
            change_bcs = get_change_bcs_decouple_stim(params=parameter_change, n=n, variance=variance)[:n]


    elif variance is None and nrrp <=1:
        continue_bcs = get_continue_bcs_ca(params=parameter_continue, ca=ca)[:n]
        change_bcs = get_change_bcs_ca(params=parameter_change, ca=ca)[:n]
        base_bcs = get_base_bcs_ca(params=base_params, ca=ca)[:n] # Won't work for MVR
    elif nrrp <= 1:
        continue_bcs = get_continue_bcs_stim(params=parameter_continue, variance=variance)[:n]
        change_bcs = get_change_bcs_stim(params=parameter_change, variance=variance)[:n]
        base_bcs = get_base_bcs_stim(params=base_params, variance=variance)[:n] # Won't work for MVR
    else:
        nrrp = '%dp0' % nrrp
        continue_bcs = get_continue_bcs_mvr(params=parameter_continue, nrrp=nrrp)[:n]
        change_bcs = get_change_bcs_mvr(params=parameter_change, nrrp=nrrp)[:n]
        base_bcs = get_base_bcs_mvr(params=base_params, nrrp=nrrp)[:n] # Won't work for MVR


    shuffle = 0
    if shuffle_level >= 1:
        change_bcs = change_bcs[1:] + [change_bcs[0]]
    if shuffle_level == 2:
        shuffle = 1
    if shuffle_level == 3:
        shuffle = 2

    results = []
    for continue_bc, change_bc, base_bc in zip(continue_bcs, change_bcs, base_bcs):

        print continue_bc
        print change_bc
        vm_continue, _ = get_soma_time_series(continue_bc, t_end=t_end, t_start=t_start)
        vm_change, times = get_soma_time_series(change_bc, t_end=t_end, t_start=t_start)
        vm_continue = np.array(vm_continue)
        vm_change = np.array(vm_change)

        if kernel > 0:
            # vm_base, _ = get_soma_time_series(base_bc, t_start=1990)
            # vm_base = np.array(vm_base)
            # offset = vm_base.shape[1]

            # vm_continue = correlations.median_filter(np.hstack([vm_base, vm_continue]), kernel_size=kernel)[:, offset:]
            # vm_change = correlations.median_filter(np.hstack([vm_base, vm_change]), kernel_size=kernel)[:, offset:]

            vm_continue[vm_continue > -40.0] = -40.0
            vm_change[vm_continue > -40.0] = -40.0

        corrs = []
        for i, corr_func in enumerate([correlations.voltage_rmsd_from_data, correlations.voltage_correlation_from_data]):
            corr, bins = corr_func(vm_continue, vm_change, times, dt=dt, shuffle=shuffle)
            corrs.append(corr)
        corrs = np.dstack(corrs)
        print corrs.shape
        results.append(corrs)
    results = np.concatenate([a[..., np.newaxis] for a in results], axis=3)
    return results, bins


def get_firing_rates(parameter_continue='', parameter_change='abcd', dt=5.0, shuffle_level=0, ca=None,
                     decouple=False, n=20, variance=None, nrrp=1):

    folder = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_firing_rates' + '/frs' + parameter_continue + '_' + parameter_change
    if shuffle_level == 0:
        file = folder + '_dt%d' % dt
    elif shuffle_level >= 1:
        file = folder + '_dt%d_shuffle%d' % (dt, shuffle_level)
    if ca is not None:
        file += '_ca%s' % ca
    if variance is not None:
        file += '_variance%s' % variance
    if decouple:
        file += '_decouple'
    if nrrp > 1:
        file += '_nrrp%d' % nrrp
    if not n == 20:
        file += '_n%d' % n
    file += '.npz'
    if not os.path.isfile(file):
        corrs, bins = compute_fr_diffs(parameter_continue=parameter_continue,
                                                parameter_change=parameter_change,
                                                shuffle_level=shuffle_level, dt=dt, ca=ca,
                                                decouple=decouple, n=n, variance=variance, nrrp=nrrp)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']



def compute_fr_diffs(parameter_continue='', parameter_change='abcd', dt=5.0, t_end=3500.0, shuffle_level=0,
                              kernel=0, base_params='abcd', ca=None, decouple=False, n=20, variance=None, nrrp=1):
    """

    :param parameter_continue:
    :param parameter_change:
    :return:
    """
    if ca is None and variance is None and nrrp <= 1 and not decouple:
        continue_bcs = get_continue_bcs(params=parameter_continue, n=n)[:n]
        change_bcs = get_change_bcs(params=parameter_change, n=n)[:n]
        base_bcs = get_base_bcs(params=base_params, n=n)[:n] # Won't work for MVR
    elif decouple:
        continue_bcs = get_continue_bcs_decouple(params=parameter_continue, n=n)[:n]
        change_bcs = get_change_bcs_decouple(params=parameter_change, n=n)[:n]
        base_bcs = get_base_bcs_decouple(params=base_params, n=n)[:n]
    elif variance is None and nrrp <=1:
        continue_bcs = get_continue_bcs_ca(params=parameter_continue, ca=ca)[:n]
        change_bcs = get_change_bcs_ca(params=parameter_change, ca=ca)[:n]
        base_bcs = get_base_bcs_ca(params=base_params, ca=ca)[:n] # Won't work for MVR
    elif nrrp <= 1:
        continue_bcs = get_continue_bcs_stim(params=parameter_continue, variance=variance)[:n]
        change_bcs = get_change_bcs_stim(params=parameter_change, variance=variance)[:n]
        base_bcs = get_base_bcs_stim(params=base_params, variance=variance)[:n] # Won't work for MVR
    else:
        nrrp = '%dp0' % nrrp
        continue_bcs = get_continue_bcs_mvr(params=parameter_continue, nrrp=nrrp)[:n]
        change_bcs = get_change_bcs_mvr(params=parameter_change, nrrp=nrrp)[:n]
        base_bcs = get_base_bcs_mvr(params=base_params, nrrp=nrrp)[:n] # Won't work for MVR


    shuffle = 0
    if shuffle_level >= 1:
        change_bcs = change_bcs[1:] + [change_bcs[0]]
    if shuffle_level == 2:
        shuffle = 1
    if shuffle_level == 3:
        shuffle = 2

    results = []

    bins = np.linspace(0, t_end-1500, (t_end-1500)/dt + 1)

    circuit = bluepy.Simulation(continue_bcs[0]).circuit
    gids_mc2 = np.array(list(circuit.get_target('mc2_Column')))
    print gids_mc2
    for continue_bc, change_bc, base_bc in zip(continue_bcs, change_bcs, base_bcs):

        spikes = bluepy.Simulation(continue_bc).v2.reports['spikes']
        df = spikes.data(t_end=t_end, gids=gids_mc2)
        # gids_spiking = np.array(df.axes[0])
        times = np.array(df)
        hist_values_1, _bins = np.histogram(times, bins=bins)

        spikes = bluepy.Simulation(change_bc).v2.reports['spikes']
        df = spikes.data(t_end=t_end, gids=gids_mc2)
        # gids_spiking = np.array(df.axes[0])
        times = np.array(df)
        hist_values_2, _bins = np.histogram(times, bins=bins)

        corrs = np.sqrt((hist_values_1 - hist_values_2)**2) # this is stupid. replace by np.abs()

        results.append(corrs)
    results = np.vstack(results)
    return results, bins


def compute_all_standard_divs():
    for p in parameters[0]:
        print p
        get_initial_divergence(parameter_continue='', parameter_change=p)
    for p in parameters[1]:
        print p
        get_initial_divergence(parameter_continue='_gv2', parameter_change=p)


def get_initial_divergence(parameter_continue='', parameter_change='abcd' ,ca=None, decouple=False):

    folder = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + '/divs_3' + parameter_continue + '_' + parameter_change
    file = folder
    if ca is not None:
        file += '_ca%s' % ca
    if decouple:
        file += '_decouple'
    file += '.npz'
    if not os.path.isfile(file):
        divs, times = compute_lyapunov(parameter_continue=parameter_continue,
                                                parameter_change=parameter_change, ca=ca,
                                                decouple=decouple)
        np.savez(open(file, 'w'), divs=divs, times=times)
    data = np.load(file)
    return data['divs'], data['times']


def compute_lyapunov(parameter_continue='', parameter_change='abcd', ca=None, decouple=False, t_end=2010):
    """
    :param parameter_continue:
    :param parameter_change:
    :return:
    """
    if ca is None:
        continue_bcs = get_continue_bcs(params=parameter_continue, n=40)
        change_bcs = get_change_bcs(params=parameter_change, n=40)

        if decouple:
            change_bcs = get_change_bcs_decouple(params=parameter_change)
    else:
        continue_bcs = get_continue_bcs_ca(params=parameter_continue, ca=ca)
        change_bcs = get_change_bcs_ca(params=parameter_change, ca=ca)


    results = []
    for i, (continue_bc, change_bc) in enumerate(zip(continue_bcs, change_bcs)):
        vm_continue, _ = get_soma_time_series(continue_bc, t_end=t_end)
        vm_change, times = get_soma_time_series(change_bc, t_end=t_end)
        vm_continue = np.array(vm_continue)
        vm_change = np.array(vm_change)

        times = np.array(times)
        norms = np.mean(np.abs(vm_change - vm_continue), axis=0)
        print norms.shape
        results.append(norms)

    norms = np.vstack(results)
    return norms, times


def plot_all_mean_voltage_traces():

    fig, axs = plt.subplots(2)
    n = 10
    for k in range(2):
        ax = axs[k]

        means = []
        bcs = get_base_bcs(params=base_parameters[k])[:n]
        for bc in bcs:
            print bc
            data, time_range = get_soma_time_series(bc, t_start=1000, t_end=None)
            means.append(data.mean(axis=0))
        means = np.vstack(means).T
        ax.plot(time_range, means, lw=0.5, alpha=0.3)

        means = []
        bcs = get_continue_bcs(params=base_parameters_continue[k])[:n]
        for bc in bcs:
            print bc
            data, time_range = get_soma_time_series(bc, t_end=3000)
            means.append(data.mean(axis=0))

        for l in range(len(parameters[k])):
            bcs = get_change_bcs(params=parameters[k][l])[:n]
            for bc in bcs:
                print bc
                data, time_range = get_soma_time_series(bc, t_end=3000)
                means.append(data.mean(axis=0))
        means = np.vstack(means).T
        ax.plot(time_range, means, lw=0.5, alpha=0.3)

    plt.savefig('combined_time_series.pdf')

def compare_base_seeds():

    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    bcs = [get_base_bcs(params=base_parameters[0])[0],
           get_base_bcs(params=base_parameters[1])[0],
           get_base_bcs_mvr()[0]]
    labels = ['UVR', 'DET-UVR', 'MVR2']

    means = []
    for bc in bcs:
        print bc
        data, time_range = get_soma_time_series(bc, t_start=0, t_end=None)
        means.append(data.mean(axis=0))
    means = np.vstack(means).T

    for k in range(3):
        ax = axs[k, 0]
        ax.plot(time_range, means[:, k], label=labels[k])
        ax.set_title(labels[k])

    bcs = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_abcd_mvr_exc/seed17%d/BlueConfig' %i for i in range(3)]
    labels = ['exc-mosaic', 'exc exc', 'mosaic mosai']

    means = []
    for bc in bcs:
        print bc
        data, time_range = get_soma_time_series(bc, t_start=0, t_end=None)
        means.append(data.mean(axis=0))
    means = np.vstack(means).T

    for k in range(3):
        ax = axs[k, 1]
        ax.plot(time_range, means[:, k], label=labels[k])
        ax.set_title(labels[k])

    plt.tight_layout()
    plt.savefig('compare_time_series.pdf')


def plot_correlations():
    n = 60
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    det = 0
    for index_correlation in range(2):
        means = {}
        errs = {}
        for p in parameters[det]:
            corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=5.0)
            bins -= 2000.0
            errs[p] = corrs.mean(axis=0).std(axis=-1, ddof=1)[:, index_correlation]/np.sqrt(corrs.shape[-1])
            means[p] = corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]
        means_df = pd.DataFrame(means, index=bins[1:])
        errs_df = pd.DataFrame(errs, index=bins[1:])

        print means_df

        ax = axs[index_correlation, 0]
        for p in ['abcd', 'a', 'ab', 'bcd', 'b', 'cd', 'd', 'abd', 'c', 'abc']:
            ax.errorbar(bins[1:(n+1)], np.array(means_df[p])[:n], yerr=np.array(errs_df[p])[:n], label=p, linewidth=0.8)
        ax.legend(loc='upper right')

        ax = axs[index_correlation, 1]
        for p in ['abcd', 'abcde', 'abcdef', 'e', 'f', 'abcdf']:
            ax.errorbar(bins[1:(n+1)], np.array(means_df[p])[:n], yerr=np.array(errs_df[p])[:n], label=p, linewidth=0.8)
        ax.legend(loc='upper right')

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('soma_corrs.pdf')


def compute_all_convergences():
    for dt in [1.0, 5.0, 10.0, 20.0, 50.0]:
        print dt
        convergence_all_sources(dt=dt)

def convergence_all_sources(dt=10.0):
    n = 200
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    funcs = [exponential_rmsd, exponential_corr]

    for shuffle in range(0, 4):
        for index_correlation in range(2):
            func = funcs[index_correlation]
            means = {}
            errs = {}
            popts = {}
            pstds = {}
            for p in ['abcd']:
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=shuffle, n=40)
                bins -= (2000 + dt/2.0)
                bins[0] = 0.0
                errs[p] = np.hstack([np.array([0]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
                means[p] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

                if shuffle == 0:
                    popt, pcov = curve_fit(func, np.repeat(np.repeat(bins[None, 1:], 31346, axis=0)[:, :, None], 40, axis=-1).mean(axis=0).flatten(),
                                           corrs[:, :, index_correlation, :].mean(axis=0).flatten(), p0 = [13.0, 0.08])
                    popts[p] = popt
                    pstds[p] = np.sqrt(np.diag(pcov))

            means_df = pd.DataFrame(means, index=bins[:])
            errs_df = pd.DataFrame(errs, index=bins[:])

            if shuffle > 0:
                ax = axs[index_correlation, 2]
                weights=1.0/corrs[:, :, index_correlation, :].size + np.zeros(corrs[:, :, index_correlation, :].size)
                bins_hist = [np.linspace(0, 10, 101), np.linspace(-1, 1, 101)][index_correlation]
                ax.hist(corrs[:, :, index_correlation, :].flatten(), bins=bins_hist, histtype='step', weights=weights)
                weights=1.0/corrs[:, :, index_correlation, :].mean(axis=-1).size + np.zeros(corrs[:, :, index_correlation, :].mean(axis=-1).size)

                ax = axs[index_correlation, 3]
                bins_hist = [np.linspace(0, 10, 101), np.linspace(-1, 1, 101)][index_correlation]
                ax.hist(corrs[:, :, index_correlation, :].mean(axis=-1).flatten(), bins=bins_hist, histtype='step', weights=weights)
            elif shuffle == 0:
                ax = axs[index_correlation, 2]
                weights=1.0/corrs[:, 0, index_correlation, :].size + np.zeros(corrs[:, 0, index_correlation, :].size)
                bins_hist = [np.linspace(0, 10, 101), np.linspace(-1, 1, 101)][index_correlation]
                ax.hist(corrs[:, 0, index_correlation, :].flatten(), bins=bins_hist, histtype='step', weights=weights)
                weights=1.0/corrs[:, 0, index_correlation, :].mean(axis=-1).size + np.zeros(corrs[:, 0, index_correlation, :].mean(axis=-1).size)

                ax = axs[index_correlation, 3]
                bins_hist = [np.linspace(0, 10, 101), np.linspace(-1, 1, 101)][index_correlation]
                ax.hist(corrs[:, 0, index_correlation, :].mean(axis=-1).flatten(), bins=bins_hist, histtype='step', weights=weights)

            for i, n in enumerate([200/int(dt), 600/int(dt)]):
                ax = axs[index_correlation, i]
                for p in ['abcd']:
                    n_start = 0
                    if shuffle > 0:
                        n_start = 1
                 #   ax.errorbar(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)], yerr=np.array(errs_df[p])[n_start:(n+1)], label='shuffle%d' % shuffle, linewidth=0.8)

                    ax.fill_between(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)] - np.array(errs_df[p])[n_start:(n+1)],
                           np.array(means_df[p])[n_start:(n+1)] + np.array(errs_df[p])[n_start:(n+1)],
                            linewidth=0.8, alpha=0.3)

                    ax.plot(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)],
                             label='shuffle%d' % shuffle, linewidth=0.8)

                    if shuffle == 0:
                        popt = popts[p]
                        print np.sqrt(np.diag(pcov))
                        bins_2 = np.linspace(bins[0], bins[n], 200)
                        ax.plot(bins_2, func(bins_2, *popt), linestyle='-', label='model', linewidth=0.8, color='black')
                        ax.set_title("t = %.2f(%.2f), a = %.2f(%.2f)" % (popts[p][0], pstds[p][0],
                                                                          popts[p][1], pstds[p][1]))
            ax.legend(loc='upper right')

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/convergence_%d.pdf' % int(dt), dpi=300)



def convergence_all_sources_main_figure(dt=10.0):
    n = 200
    fig, axs = plt.subplots(2)

    axs = [axs[0], axs[0].twinx()]
    colors =  ['#e41a1c', '#377eb8']
    for shuffle in range(0, 2):
        for index_correlation in range(2):
            means = {}
            errs = {}
            for p in ['abcd']:
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=shuffle, n=40)
                bins -= (2000 + dt/2.0)
                bins[0] = 0.0
                errs[p] = np.hstack([np.array([0]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
                means[p] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

            means_df = pd.DataFrame(means, index=bins[:])
            errs_df = pd.DataFrame(errs, index=bins[:])

            n = 25
            ax = axs[index_correlation]
            for p in ['abcd']:
                    n_start = 0
                    if shuffle > 0:
                        n_start = 1

                        ax.fill_between(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)] - np.array(errs_df[p])[n_start:(n+1)],
                           np.array(means_df[p])[n_start:(n+1)] + np.array(errs_df[p])[n_start:(n+1)],
                            linewidth=0.8,  color=['#d9d9d9', '#a6bddb'][index_correlation])

                        ax.plot(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)],
                             label='shuffle%d' % shuffle, linewidth=0.8, color=['#737373', '#3690c0'][index_correlation],
                            linestyle=['-', '--'][index_correlation], marker='.', ms=2,)
                    else:
                        ax.errorbar(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)], yerr=np.array(errs_df[p])[n_start:(n+1)], label='shuffle%d' % shuffle, linewidth=0.8,
                                    color=['black', '#034e7b'][index_correlation], marker='.',ms=2,
                                    linestyle=['-', '--'][index_correlation])
                        ax.plot(-1 * bins[n_start:(n+1)], index_correlation + np.zeros(n+1-n_start), linewidth=0.8,
                                color=['black', '#034e7b'][index_correlation], linestyle=['-', '--'][index_correlation]) #, marker='.', markeredgecolor='black', ms=3)

    axs[0].set_ylim([-0.1, 3.2])
    axs[1].set_ylim([-0.0, 1.05])

    for ax in axs:
        ax.set_xlim([-250, 250])
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
    plt.tight_layout()

    plt.savefig('figures/main_divergence_paper.pdf')



def convergence_comparison(normalize=True):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    funcs = [exponential_rmsd, exponential_corr]
    for shuffle in range(0, 1):
        for index_correlation in range(2):
            func = funcs[1]
            means = {}
            errs = {}
            scales = {}
            popts = {}
            pstds = {}
            bins_all = {}
            for dt in [1.0, 5.0, 10.0, 20.0, 50.0]:
                print dt
                p = str(dt)
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=shuffle, n=40)
                corrs_shuffle, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=1, n=40)

                corrs = [-1, 1][index_correlation] * (corrs - corrs_shuffle)
                start_mean = [-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean())
                start_error = mean_confidence_interval([0, 1][index_correlation] - [-1, 1][index_correlation] * corrs_shuffle[:, : , index_correlation, :].flatten())
                scales[p] = start_mean
                bins -= (2000 + dt/2.0)
                bins[0] = 0.0
                errs[p] = np.hstack([np.array([start_error]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
                means[p] = np.hstack([np.array([start_mean]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
                bins_all[p] = bins

                if normalize:
                    means[p] /= scales[p]
                    errs[p] /= scales[p]


                if shuffle == 0 and dt == 10.0:
                    popt, pcov = curve_fit(func, np.repeat(np.repeat(bins[None, 1:], 31346, axis=0)[:, :, None], 40, axis=-1).mean(axis=0).flatten(),
                                           corrs[:, :, index_correlation, :].mean(axis=0).flatten(), p0 = [13.0, 0.08])
                    popts[p] = popt
                    pstds[p] = np.sqrt(np.diag(pcov))


            means_df = means
            errs_df = errs

            ax = axs[index_correlation]
            colors = ['#c7e9b4',
                      '#7fcdbb',
                      '#41b6c4',
                      '#1d91c0',
                      '#225ea8']

            colors_2 = ['#fcbba1',
                        '#fc9272',
                        '#fb6a4a',
                        '#ef3b2c',
                        '#cb181d']

            for i, dt in enumerate([1.0, 5.0, 10.0, 20.0, 50.0]):
                n = 300/int(dt)
                p = str(dt)
                n_start = 0
                if shuffle > 0:
                    n_start = 1
                bins = bins_all[p]

                ax.errorbar(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)], yerr=np.array(errs_df[p])[n_start:(n+1)],
                             label='%d ms' % dt, linewidth=0.4, color=colors[i], linestyle='-', marker='o', markersize=1.0,
                            alpha=0.8)
                if shuffle == 0 and dt == 10.0:
                        popt = popts[p]
                        bins_2 = np.linspace(bins[0], bins[n], 200)
                        label = "t = %.2f(%.2f), a = %.4f(%.4f)" % (popts[p][0], pstds[p][0],
                                                                          popts[p][1], pstds[p][1])
                        ax.plot(bins_2, func(bins_2, *popt), linestyle='-', label=label, linewidth=0.8,
                                color=colors_2[i])

            ax.legend(loc=['lower right', 'upper right'][index_correlation], prop={'size':4})
    axs[1].set_ylabel('r')
    axs[0].set_ylabel('RMSD (mV)')

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/convergence_comparisons.pdf')


def convergence_main_figure(normalize=True):
    fig, axs = plt.subplots(2, 2, figsize=(7, 3))
    func = exponential_norm
    dt = 10.0

    for index_correlation in range(2):
        p = str(dt)
        means = {}
        errs = {}
        scales = {}
        popts = {}
        pstds = {}
        bins_all = {}
        corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=0, n=40)
        corrs_shuffle, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=1, n=40)
        corrs = [-1, 1][index_correlation] * (corrs - corrs_shuffle)
        start_mean = [-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean())
        start_error = mean_confidence_interval([0, 1][index_correlation] - [-1, 1][index_correlation] * corrs_shuffle[:, : , index_correlation, :].flatten())
        scales[p] = start_mean
        bins -= (2000 + dt/2.0)
        bins[0] = 0.0
        errs[p] = np.hstack([np.array([start_error]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
        means[p] = np.hstack([np.array([start_mean]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
        bins_all[p] = bins

        if normalize:
            means[p] /= scales[p]
            errs[p] /= scales[p]
            corrs /= scales[p]

        n = 4
        k = 0
        print bins[None, k:(n+1)]
        print corrs.shape
        bins_3 = np.linspace(bins[0], bins[n], 200)

        popt, pcov = curve_fit(func, np.repeat(np.repeat(bins[None, (k+1):(n+1)], 31346, axis=0)[:, :, None], 40, axis=-1).mean(axis=0).flatten(),
                                           corrs[:, k:n, index_correlation, :].mean(axis=0).flatten(), p0=20)

        # fig, axs = plt.subplots(2)
        # axs[index_correlation].scatter( np.repeat(np.repeat(bins[None, 1:(n+1)], 31346, axis=0)[:, :, None], 40, axis=-1).mean(axis=0).flatten(),
        #                                    corrs[:, :n, index_correlation, :].mean(axis=0).flatten(), rasterized=True)
        # plt.savefig('figures/test_fit.pdf')


        popts[p] = popt
        pstds[p] = np.sqrt(np.diag(pcov))

        means_df = means
        errs_df = errs

        for l in range(2):
            ax = axs[l, index_correlation]
            n = [10, 80][l]
            n_start = 0
            bins = bins_all[p]

            ax.errorbar(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)], yerr=np.array(errs_df[p])[n_start:(n+1)],
                                 label='%d ms' % dt, linewidth=0.8, color='#225ea8', linestyle='-', marker='.', markersize=1.0,
                                 markeredgecolor='black')
            popt = popts[p]
            bins_2 = np.linspace(bins[0], bins[n], 200)
            label = "t = %.2f(%.2f)" % (popts[p], pstds[p])
            ax.plot(bins_3, func(bins_3, *popt), linestyle='-', label=label, linewidth=0.8,
                                    color='#cb181d')
            ax.plot(bins_2, func(bins_2, *popt), linestyle='--', label=label, linewidth=0.8,
                                    color='#cb181d')
            ax.plot(bins_2, np.zeros(bins_2.size), linewidth=0.8, color='orange')

            if l == 1:
                indices = np.array(means_df[p])[n_start:(n+1)] - np.array(errs_df[p])[n_start:(n+1)] > 0
                ax.scatter(bins[n_start:(n+1)][indices], 1.1 + np.zeros(indices.sum()), marker='x',s=1.0, color='black')

            ax.legend(loc='upper right', prop={'size':4})
    axs[0, 1].set_ylabel('r')
    axs[0, 0].set_ylabel('RMSD (mV)')

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/convergence_main_figure.pdf')



def fitting_comparison():
    n = 200
    dt = 5.0
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    for shuffle in range(0, 4):
        for index_correlation in range(2):
            means = {}
            errs = {}
            for p in ['abcd']:
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=shuffle, n=40)
                bins -= (2000 + dt/2.0)
                bins[0] = 0.0
                errs[p] = np.hstack([np.array([0]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
                means[p] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            means_df = pd.DataFrame(means, index=bins[:])
            errs_df = pd.DataFrame(errs, index=bins[:])

            funcs = [exponential_rmsd, exponential_corr]


            for i, n in enumerate([30, 160]):
                func = funcs[index_correlation]
                ax = axs[index_correlation, i]
                for p in ['abcd']:
                    n_start = 0
                    if shuffle > 0:
                        n_start = 1
                 #   ax.errorbar(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)], yerr=np.array(errs_df[p])[n_start:(n+1)], label='shuffle%d' % shuffle, linewidth=0.8)

                    ax.fill_between(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)] - np.array(errs_df[p])[n_start:(n+1)],
                           np.array(means_df[p])[n_start:(n+1)] + np.array(errs_df[p])[n_start:(n+1)],
                            label='shuffle%d' % shuffle, linewidth=0.8, alpha=0.3)

                    ax.plot(bins[n_start:(n+1)], np.array(means_df[p])[n_start:(n+1)],
                             label='shuffle%d' % shuffle, linewidth=0.8)

                    if shuffle == 0:
                        popt, pcov = curve_fit(func, bins[:], np.array(means_df[p])[:], p0 = [13.0, 0.08])
                        print np.sqrt(np.diag(pcov))
                        bins_2 = np.linspace(bins[0], bins[n], 2000)
                        ax.plot(bins_2, func(bins_2, *popt), linestyle='-', label='model', linewidth=0.8, color='black')
                        ax.set_title("t = %.2f(%.2f), a = %.2f(%.2f)" % (popt[0], np.sqrt(np.diag(pcov))[0],
                                                                          popt[1], np.sqrt(np.diag(pcov))[1]))
            ax.legend(loc='upper right')

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/convergence.pdf')


def exponential_norm(t, tau):
            return np.exp(-t/tau)

def exponential_corr(t, tau, a):
            return (1 - a) * np.exp(-t/tau) + a

def exponential_rmsd(t, tau, a):
        return a * (1 - np.exp(-t/tau))

def exponential(t, tau, a, b):
        return a * np.exp(-t/tau) + b


def fig_exponential(x, y, x_2=None, p0=None):
    popt, pcov = curve_fit(exponential, x, y, p0=p0)
    if x_2 is None:
        x_2 = x
    return exponential(x_2, popt), popt, np.sqrt(np.diag(pcov))


def mean_confidence_interval(data, confidence=0.95):
    m, se = data.mean(), scipy.stats.sem(data)
    error = se * scipy.stats.t.ppf((1+confidence)/2., data.size-1)
    return error


def plot_correlations_figure_1_all_data(normalize=True):
    dt = 10.0
    n = 25
    fig, axs = plt.subplots(8, 2, figsize=(8, 16))
    for index_correlation in range(2):
        means = {}
        errs = {}
        p_cont = ['' for x in parameters[0]] + ['_gv2' for x in parameters[1]]
        for p, pc in zip(parameters[0] + parameters[1], p_cont):
            print "loading %s" % p
            corrs, bins = get_correlations(parameter_continue=pc, parameter_change=p, dt=dt)

            # deleting a neuron in faulty simulation report
            if p == 'gv2b':
                corrs = np.delete(corrs, 573, axis=0)

            bins -= 2000.0
            scale = 1.0
            if normalize:
                #corrs_shuffle, _ = get_correlations(parameter_continue=pc, parameter_change=p, dt=dt, shuffle_level=2)
                #if p == 'gv2b':
                #    corrs_shuffle = np.delete(corrs_shuffle, 573, axis=0)
                scale = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])[100:].mean()


            # errs[p] = np.hstack([np.array([0]), corrs.mean(axis=0).std(axis=-1, ddof=1)[:, index_correlation]/np.sqrt(corrs.shape[-1])])
            errs[p] = np.hstack([np.array([0]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means[p] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

            if p == 'gv2b':
                print means[p]

            if index_correlation == 0 and normalize:
                errs[p] = errs[p] / scale
                means[p] = means[p] / scale
            elif normalize:
                errs[p] = errs[p] / (1.0 - scale)
                means[p] = (means[p] - scale) / (1 - scale)


        means_df = pd.DataFrame(means, index=bins[:])
        errs_df = pd.DataFrame(errs, index=bins[:])

        colors = ['black', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

        plot_contents = [['abcd', 'gv2bcd', 'gv2cd'],
                         ['abcd', 'bcd', 'cd'],
                         ['abcd', 'gv2b', 'gv2cd', 'gv2c', 'gv2d'],
                         ['abcd', 'ab', 'cd', 'c', 'd'],
                         ['abcd', 'gv2cd', 'gv2e', 'gv2f'],
                         ['abcd', 'cd', 'e', 'f'],
                         ['abcd', 'abcde', 'abcdf', 'abcdef', 'gv2e', 'gv2f'],
                         ['abcd', 'abcdef', 'gv2bcd', 'gv2bcdef']]

        for j, contents in enumerate(plot_contents):
            ax = axs[j, index_correlation]
            for i, p in enumerate(contents):
                ax.errorbar(bins[:(n+1)], np.array(means_df[p])[:(n+1)], yerr=np.array(errs_df[p])[:(n+1)],
                                linestyle=linestyles[p], label=names[p], linewidth=0.8, color=colors[i])

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('RMSD')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])

    for ax in axs[:, 1].flatten():
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()

    plt.savefig('figures/soma_corrs_fig1_dt%d.pdf' % int(dt))


def plot_correlations_stut_ir(normalize=False):
    """
    Comparison plot of sources of variability

    :param normalize:
    :return:
    """
    dt = 10.0
    index_correlation = 1

    circuit = bluepy.Simulation(get_base_bcs()[0]).circuit
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    etypes = circuit.mvddb.etype_id2name_map().values()


    etypes_c = [etypes[x] for x in [2, 4, 8, 9, 10]]
    print etypes_c
    indices = np.array([etype in etypes_c for etype in cells['etype']])
    print indices.sum()
    print cells['mtype'][indices]
    print cells['etype'][indices]


    corrs_all = {}
    p_cont = ['' for x in parameters[0]] + ['_gv2' for x in parameters[1]]
    for p, pc in zip(parameters[0] + parameters[1], p_cont):
        print "loading %s" % p
        corrs, bins = get_correlations(parameter_continue=pc, parameter_change=p, dt=dt)
        # deleting a neuron in faulty simulation report

        corrs_all[p] = corrs[indices]

    decouple_params = ['abcd', 'ab', 'a', 'c', 'd', 'f', 'g']
    for p in decouple_params:
        print "loading %s" % p
        corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt, decouple=True)
        corrs_all[p + '_dec'] = corrs[indices]
    bins -= (2000.0 + dt/2.0)
    bins[0] = 0


    plot_contents = [['abcd', 'f',    'g'],
                     [p + '_dec' for p in ['abcd', 'f',    'g']],
                     ['abcd', 'ab', 'a', 'c', 'd'],
                     [p + '_dec' for p in ['abcd', 'ab', 'a', 'c', 'd']]]
    colors = ['black','#377eb8','#4daf4a','#984ea3','#ff7f00']
    n = 41
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    for j, contents in enumerate(plot_contents):
        ax = axs.flatten()[j]
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')

        for i, p in enumerate(contents):
            corrs = corrs_all[p]

            means_corr = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])


            if i == 0 or i == 2: #  or j == 3:
                ax.plot(bins[:n], means_corr[:n], label=p, lw=0.8, color=colors[i], linestyle='-')
                ax.fill_between(bins[:n], means_corr[:n] - errs[:n],
                                          means_corr[:n] + errs[:n], alpha=0.3, lw=0.5, color=colors[i],
                                          linewidth=0.0)

            else:
                ax.errorbar(bins[:n], means_corr[:n], yerr=errs[:n], label=p, lw=0.5, color=colors[i])

            #ax.set_ylim([0, 1])

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/deciphering_sources_stutir.pdf')

    # Figure 3 difference visual
    colors = ['black','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    plot_contents = [['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     ['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     ['abcd', 'gv2bcd', 'gv2b', 'gv2cd', 'gv2g', 'gv2f'],
                     ['abcd', 'bcd', 'b', 'cd', 'g', 'f']]
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    for j, contents in enumerate(plot_contents):
        n = [61, 61, 41, 41][j]

        ax = axs.flatten()[j]

        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')

        for i, p in enumerate(contents):
            corrs = corrs_all[p]
            if j == 1:
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, 1].mean()
                print "This needs to be number of simulations:"
                print corrs.mean(axis=0)[100:, 1].mean(axis=0).shape
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, 1].mean(axis=0))
                print "relative scale err = %.3f" % (scale_err/scale)
            means_corr = np.hstack([np.array([1]), corrs.mean(axis=0).mean(axis=-1)[:, 1]])
            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, 1]])

            if j == 1:
                means_corr, errs = get_similarity(scale, scale_err, means_corr, errs, index_correlation=1)

            # print errs[5:7]
            # print errs_2[5:7]
            if i == 0 or j == 3:
                ax.plot(bins[:n], means_corr[:n], label=p, lw=0.8, color=colors[i], linestyle='--')
                ax.fill_between(bins[:n], means_corr[:n] - errs[:n],
                                          means_corr[:n] + errs[:n], alpha=0.3, lw=0.5, color=colors[i],
                                          edgecolor='')

            else:
                ax.errorbar(bins[:n], means_corr[:n], yerr=errs[:n], label=p, lw=0.5, color=colors[i])

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/deciphering_sources_2_stutir.pdf')

    # Figure 4 difference visual
    colors = ['black','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    plot_contents = [['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     [p + '_dec' for p in decouple_params]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 6))

    for j, contents in enumerate(plot_contents):
        for index_correlation in range(2):
            ax = axs[index_correlation, j]
            ax.set_ylabel(['RMSD (mV)', 'r'][index_correlation])
            ax.set_ylim([[0, 5.1], [0, 1.1]][index_correlation])
            for i, p in enumerate(contents):
                corrs = corrs_all[p]
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                ax.bar([i], [scale], yerr=[scale_err], color='#a6bddb')
                ax.set_xticks(np.arange(len(contents)))
                ax.set_xticklabels(contents)
    for j, contents in enumerate(plot_contents):
        for index_correlation in range(2):
            ax = axs[index_correlation + 2, j]
            ax.set_ylabel(['dRMSD', 'dr'][index_correlation])
            ax.set_ylim([0, 1.1])
            for i, p in enumerate(contents):
                corrs = corrs_all[p]
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                step = corrs.mean(axis=0).mean(axis=-1)[1, index_correlation]
                step_err = mean_confidence_interval(corrs.mean(axis=0)[1, index_correlation])

                step_2, step_err = get_similarity(scale, scale_err, step, step_err, index_correlation=index_correlation)

                ax.bar([i], [step_2], yerr=[step_err], color='#a6bddb')
            ax.set_xticks(np.arange(len(contents)))
            ax.set_xticklabels(contents)

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/deciphering_sources_3_stutir.pdf')



def plot_correlations_figure_1_detailed_for_paper(normalize=False):
    """
    Comparison plot of sources of variability

    :param normalize:
    :return:
    """
    dt = 10.0
    index_correlation = 0

    corrs_all = {}
    p_cont = ['' for x in parameters[0]] + ['_gv2' for x in parameters[1]]
    for p, pc in zip(parameters[0] + parameters[1], p_cont):
        print "loading %s" % p
        corrs, bins = get_correlations(parameter_continue=pc, parameter_change=p, dt=dt)
        # deleting a neuron in faulty simulation report
        if p == 'gv2b':
            corrs = np.delete(corrs, 573, axis=0)



        corrs_all[p] = corrs

    decouple_params = ['abcd', 'ab', 'a', 'c', 'd', 'f', 'g']
    for p in decouple_params:
        print "loading %s" % p
        corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt, decouple=True)
        corrs_all[p + '_dec'] = corrs
    bins -= (2000.0 + dt/2.0)
    bins[0] = 0
    plot_contents = [['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     ['gv2bcd', 'gv2b', 'gv2cd', 'gv2g', 'gv2f'],
                     [p + '_dec' for p in decouple_params]]

    # Figure 1 RMSD comparison
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    for j, contents in enumerate(plot_contents):
        ax = axs.flatten()[j]
        ax.set_xlabel('RMSD (mV)')
        ax.set_ylabel('r')

        for p in contents:
            corrs = corrs_all[p]
            means_corr = np.hstack([np.array([1]), corrs.mean(axis=0).mean(axis=-1)[:, 1]])
            means_rmsd = np.hstack([np.array([0]), corrs.mean(axis=0).mean(axis=-1)[:, 0]])
            ax.plot(means_rmsd, means_corr, alpha=0.6, label=p, lw=0.8)

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/rmsd_r_correlation.pdf')

    # Figure 2 difference visual
    plot_contents = [['abcd', 'f',    'g'],
                     [p + '_dec' for p in ['abcd', 'f',    'g']],
                     ['abcd', 'ab', 'a', 'c', 'd'],
                     [p + '_dec' for p in ['abcd', 'ab', 'a', 'c', 'd']]]
    colors = ['black','#377eb8','#4daf4a','#984ea3','#ff7f00']
    n = 41
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    for j, contents in enumerate(plot_contents):
        ax = axs.flatten()[j]
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')

        for i, p in enumerate(contents):
            corrs = corrs_all[p]

            means_corr = np.hstack([np.array([1]), corrs.mean(axis=0).mean(axis=-1)[:, 1]])
            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, 1]])


            if i == 0 or i == 2: #  or j == 3:
                ax.plot(bins[:n], means_corr[:n], label=p, lw=0.8, color=colors[i], linestyle='-')
                ax.fill_between(bins[:n], means_corr[:n] - errs[:n],
                                          means_corr[:n] + errs[:n], alpha=0.3, lw=0.5, color=colors[i],
                                          linewidth=0.0)

            else:
                ax.errorbar(bins[:n], means_corr[:n], yerr=errs[:n], label=p, lw=0.5, color=colors[i])

            ax.set_ylim([0, 1])

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/deciphering_sources.pdf')

    # Figure 3 difference visual
    colors = ['black','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    plot_contents = [['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     ['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     ['abcd', 'gv2bcd', 'gv2b', 'gv2cd', 'gv2g', 'gv2f'],
                     ['abcd', 'bcd', 'b', 'cd', 'g', 'f']]
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    for j, contents in enumerate(plot_contents):
        n = [61, 61, 41, 41][j]

        ax = axs.flatten()[j]

        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')

        for i, p in enumerate(contents):
            corrs = corrs_all[p]
            if j == 1:
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, 1].mean()
                print "This needs to be number of simulations:"
                print corrs.mean(axis=0)[100:, 1].mean(axis=0).shape
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, 1].mean(axis=0))
                print "relative scale err = %.3f" % (scale_err/scale)
            means_corr = np.hstack([np.array([1]), corrs.mean(axis=0).mean(axis=-1)[:, 1]])
            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, 1]])

            if j == 1:
                means_corr, errs = get_similarity(scale, scale_err, means_corr, errs, index_correlation=1)

            # print errs[5:7]
            # print errs_2[5:7]
            if i == 0 or j == 3:
                ax.plot(bins[:n], means_corr[:n], label=p, lw=0.8, color=colors[i], linestyle='--')
                ax.fill_between(bins[:n], means_corr[:n] - errs[:n],
                                          means_corr[:n] + errs[:n], alpha=0.3, lw=0.5, color=colors[i],
                                          edgecolor='')

            else:
                ax.errorbar(bins[:n], means_corr[:n], yerr=errs[:n], label=p, lw=0.5, color=colors[i])

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/deciphering_sources_2.pdf')

    # Figure 4 difference visual
    colors = ['black','#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    plot_contents = [['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     [p + '_dec' for p in decouple_params]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 6))

    for j, contents in enumerate(plot_contents):
        for index_correlation in range(2):
            ax = axs[index_correlation, j]
            ax.set_ylabel(['RMSD (mV)', 'r'][index_correlation])
            ax.set_ylim([[0, 3.1], [0, 1.1]][index_correlation])
            for i, p in enumerate(contents):
                corrs = corrs_all[p]
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                ax.bar([i], [scale], yerr=[scale_err], color='#a6bddb')
                ax.set_xticks(np.arange(len(contents)))
                ax.set_xticklabels(contents)
    for j, contents in enumerate(plot_contents):
        for index_correlation in range(2):
            ax = axs[index_correlation + 2, j]
            ax.set_ylabel(['dRMSD', 'dr'][index_correlation])
            ax.set_ylim([0, 1.1])
            for i, p in enumerate(contents):
                corrs = corrs_all[p]
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                step = corrs.mean(axis=0).mean(axis=-1)[1, index_correlation]
                step_err = mean_confidence_interval(corrs.mean(axis=0)[1, index_correlation])

                step_2, step_err = get_similarity(scale, scale_err, step, step_err, index_correlation=index_correlation)

                ax.bar([i], [step_2], yerr=[step_err], color='#a6bddb')
            ax.set_xticks(np.arange(len(contents)))
            ax.set_xticklabels(contents)

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    plt.savefig('figures/deciphering_sources_3.pdf')


def get_similarity(scale, scale_err, means, mean_errs, index_correlation=0):
    if index_correlation == 1:
        step_2 = (1 - means) / (1 - scale)
        errs = step_2 * np.sqrt((scale_err/(1 - scale))**2 + (mean_errs/(1 - means))**2)
        means = 1 - step_2
    if index_correlation == 0:
        step_2 = means/scale
        errs = step_2 * np.sqrt((scale_err/(scale))**2 + (mean_errs/(means))**2)
        means = 1 - step_2

    return means, errs

def get_similarity_no_error(scale, means, index_correlation=0):
    if index_correlation == 1:
        step_2 = (1 - means) / (1 - scale)
        means = 1 - step_2
    if index_correlation == 0:
        step_2 = means/scale
        means = 1 - step_2
    return means


def plot_evolving_corrs():
    """
    Plots of the distributions of evolving distances for a few time steps.
    :return:
    """
    dt = 10.0
    n = 25
    fig, axs = plt.subplots(2, 2, figsize=(7, 4))
    index_correlation = 0

    p = 'abcd'
    corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt)
    print corrs.shape


    colors = ['#525252', '#969696', '#cccccc', '#f7f7f7', '#67a9cf']
    bins_s = [np.linspace(0, 8, 81), np.linspace(-0.5, 1, 51)]
    xs = [1, 1, 1, 1, 50]
    for index_correlation in range(2):
        bins = bins_s[index_correlation]
        ax = axs[0, index_correlation]

        errs = np.hstack([np.array([0]), corrs.mean(axis=0).std(axis=-1, ddof=1)[:, index_correlation]/np.sqrt(corrs.shape[-1])])
        means = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
        print means.shape
        #ax.vlines(means[1:100], np.zeros(means.shape)[1:], np.zeros(means.shape)[1:100] + 6000, color='black', lw=0.75, alpha=0.2)
        ax.bar([index_correlation], [6200], color='red', edgecolor='black', width=bins_s[index_correlation][1] - bins_s[index_correlation][0])

        #ax.scatter(means[1:100], np.zeros(means.shape)[1:100] + 6000, marker=['o', 'o'][index_correlation],
        #               c=(['red', '#525252', '#969696', '#cccccc'] + ['#f7f7f7' for g in range(4, 150)] + ['#67a9cf'])[1:100],
        #               edgecolor='black', alpha=1, lw=0.75, s=25.0)

        for j, n in enumerate([0, 1, 2, 3, 100]):
            x = xs[j]

            ax.hist(corrs[:, n:(n+x), index_correlation, :].mean(axis=-1).flatten(), bins=bins, histtype='stepfilled',
                    label='%d-%d ms' % (n*10, (n+1)*10), edgecolor="black", color=colors[j], weights=np.ones(corrs.shape[0] * x)/x)


    axs[0, 0].legend(loc='upper right', prop={'size':8}, frameon=False)

    for ax in axs.flatten():
    #     ax.set_xlabel('t (ms)')
        ax.set_ylabel('Neurons')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/evolving_corrs.pdf')


def plot_decoupled_validation():
    """
    Validation of the lastest decoupled simulations.
    """
    dt = 10.0
    n = 25
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    results = [get_correlations(parameter_continue='', parameter_change='abcd', dt=dt),
               get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, decouple=True, nondecouple=2),
               get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, decouple=True, nondecouple=0),
               get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, decouple=True, nondecouple=1),
                              get_correlations(parameter_continue='', parameter_change='x', dt=dt, decouple=True, nondecouple=1)]
    bins = results[0][1]

    labels = ['standard abcd (n=40)', 'standard abcd, new (n=19)', 'full decoupled, abcd (n=19)',
              'half decoupled abcd, both (n=19)', 'half decoupled x, both (n=19)']
    corrs_all = {}
    for label, (corrs, bins) in zip(labels, results):
        corrs_all[label] = corrs

    bins -= (2000.0 + dt/2.0)
    bins[0] = 0

    corrs = corrs_all['half decoupled x, both (n=19)']
    print corrs.shape
    for i in range(19):
        print ((corrs[:, :, 0, i] > 0).sum(axis=1) > 0).sum() / 31346. #/float(corrs[:, :, 0, i].size)
        print ((corrs[:, :, 0, i] > 0.1).sum(axis=1) > 0).sum() / 31346. #/float(corrs[:, :, 0, i].size)

    print corrs[:, :, 0, :].max()
    print corrs[:, :, 0, :].min()
    print np.median(corrs[:, :, 0, :])

    print corrs.size
    for index_correlation in range(2):
        ax = axs[0, index_correlation]
        for j, label in enumerate(labels):
            corrs = corrs_all[label]
            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            ax.errorbar(bins[:(n+1)]+j,means[:(n+1)], yerr=errs[:(n+1)],
                        linestyle='-', label=label, linewidth=0.8)

    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/decoupled_validation_dt%d.pdf' % int(dt), dpi=300)


def plot_decoupled_analysis():
    """
    Plotting the decoupled simulations
    """
    ""
    dt = 10.0
    n = 20

    decouple_params = ['abcd', 'ab', 'a', 'c', 'd', 'e', 'f', 'g']
    corrs_all = {}
    corrs_shuffle, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, decouple=True, shuffle_level=1)
    for p in decouple_params:
        corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt, decouple=True)
        corrs_all[p] = corrs

    bins -= (2000.0 + dt/2.0)
    bins[0] = 0



    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    print corrs.size
    for index_correlation in range(2):
        ax = axs[0, index_correlation]
        for j, label in enumerate(decouple_params):
            corrs = corrs_all[label]
            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            ax.errorbar(bins[:(n+1)]+j,means[:(n+1)], yerr=errs[:(n+1)],
                        linestyle='-', label=label, linewidth=0.8)
        ax = axs[1, index_correlation]
        for j, label in enumerate(decouple_params):
            corrs = [1, -1][index_correlation] *(corrs_shuffle - corrs_all[label])
            corrs /= [1, -1][index_correlation] * (corrs_shuffle[:, :, index_correlation, :].mean() - [0, 1][index_correlation])

            errs = np.hstack([np.array([0]), np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means = np.hstack([np.array([1]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            ax.errorbar(bins[:(n+1)]+j,means[:(n+1)], yerr=errs[:(n+1)],
                        linestyle='-', label=label, linewidth=0.8)
            ax.set_ylim([0, 1.2])


    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/decoupled_analysis_dt%d.pdf' % int(dt), dpi=300)


def plot_cas_figure_2():
    dt = 10.0
    n = 100
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for index_correlation in range(2):
        means = {}
        errs = {}
        cas_2 = cas + ['1p25']
        for ca in cas_2:
            print "loading %s" % ca
            if ca == '1p25':
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=0, n=40)
            else:
                corrs, bins = get_correlations(parameter_continue='_ca_abcd', parameter_change='ca_abcd', dt=dt, shuffle_level=0, ca=ca)
            print "loaded %s" % ca
            bins -= (2000.0 + dt/2)
            bins[0] = 0
            errs[ca] = np.hstack([np.array([0]),
                                     np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means[ca] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

        means_df = pd.DataFrame(means, index=bins[:])
        errs_df = pd.DataFrame(errs, index=bins[:])

        lss =  ['-', '-', '-', '--'] #['--', '--', '--', '-']
        colors = ['#377eb8', '#4daf4a', '#e41a1c', 'black']

        for j in range(2):
            n = [15, 75][j]
            ax = axs[j, index_correlation]
            for i, ca in enumerate(cas_2):
                if ca == '1p3':
                    ax.fill_between(bins[:(n+1)], np.array(means_df[ca])[:(n+1)] - np.array(errs_df[ca])[:(n+1)],
                                    np.array(means_df[ca])[:(n+1)] + np.array(errs_df[ca])[:(n+1)],
                                   label=ca, facecolor=colors[i], alpha=0.3)
                    ax.plot(bins[:(n+1)], np.array(means_df[ca])[:(n+1)],
                                    linestyle=lss[i], label=ca, linewidth=1, color=colors[i], markersize=2.0)

                else:
                    ax.errorbar(bins[:(n+1)], np.array(means_df[ca])[:(n+1)], yerr=np.array(errs_df[ca])[:(n+1)],
                               label=ca, color=colors[i], linewidth=1, linestyle=lss[i])

    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    for ax in axs[:, 1]:
        ax.set_ylim([0, 1])
    plt.tight_layout()

    plt.savefig('figures/calcium_scan_corrs_fig2.pdf')


def plot_mvr():
    dt = 10.0
    n = 15
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for index_correlation in range(2):
        ax = axs[0, index_correlation]

        means = {}
        errs = {}
        cas = ['1', '2', '3', '10', 'det2', 'det1']
        nrrps = [1 , 2, 3, 10, 1, 1]
        labels = ['UVR (1)', 'MVR (2)', 'MVR (3)', 'MVR (10)', 'Det. (mean)', 'Det. (seed)']

        for nrrp, ca in zip(nrrps, cas):
            print "loading %s" % ca
            if ca == 'det1':
                corrs, bins = get_correlations(parameter_continue='', parameter_change='bcd', dt=dt)
            elif ca == 'det2':
                corrs, bins = get_correlations(parameter_continue='_gv2', parameter_change='gv2bcd', dt=dt)
            else:
                n_sims = 20
                if nrrp == 1:
                    n_sims = 40
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, nrrp=nrrp, n=n_sims)
            bins -= (2000 - dt/2)
            errs[ca] = np.hstack([np.array([0]),
                                     np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means[ca] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

        means_df = pd.DataFrame(means, index=bins[:])
        errs_df = pd.DataFrame(errs, index=bins[:])

        for p in means.keys():
            scale = means[p][100:].mean()
            scale_err = np.linalg.norm(errs[p][100:])/means[p][100:].size
            print scale_err/scale

            if index_correlation == 0:
                means_new = means[p] / scale
                # errs[p] /= scale
                errs[p] = means_new * np.sqrt((errs[p]/means[p])**2 + (scale_err/scale)**2)
                means[p] = 1 - means_new
            else:
                errs[p] = errs[p] / (1.0 - scale)
                means[p] = (means[p] - scale) / (1 - scale)
        means_df_norm = pd.DataFrame(means, index=bins[:])
        errs_df_norm = pd.DataFrame(errs, index=bins[:])

        symbols = ['^', 'o', 'o', 'o', 's', 'x']
        lss =  ['-', '-', '-', '-', '--', '-']
        colors = ['red', '#66c2a5', '#fee08b', '#8da0cb', 'black', 'black']
    #    plot_points =
        for i, ca in enumerate(cas):
            ax.fill_between(bins[0:(n+1)] - dt/2, np.array(means_df[ca])[0:(n+1)] - np.array(errs_df[ca])[0:(n+1)],
                            np.array(means_df[ca])[0:(n+1)] + np.array(errs_df[ca])[0:(n+1)],
                            facecolor=colors[i], alpha=0.3)
        for i, ca in enumerate(cas):
            ax.plot(bins[0:(n+1)] - dt/2, np.array(means_df[ca])[0:(n+1)],
                            linestyle=lss[i], label=labels[i], linewidth=0.8, markersize=2,
                            marker=symbols[i], markeredgecolor=colors[i], color=colors[i])

        ax = axs[1, index_correlation]

        for i, ca in enumerate(cas):
            ax.fill_between(bins[0:(n+1)] - dt/2, np.array(means_df_norm[ca])[0:(n+1)] - np.array(errs_df_norm[ca])[0:(n+1)],
                            np.array(means_df_norm[ca])[0:(n+1)] + np.array(errs_df_norm[ca])[0:(n+1)],
                            facecolor=colors[i], alpha=0.3)
        for i, ca in enumerate(cas):
            ax.plot(bins[0:(n+1)] - dt/2, np.array(means_df_norm[ca])[0:(n+1)],
                            linestyle=lss[i], label=labels[i], linewidth=0.8, markersize=2,
                            marker=symbols[i], markeredgecolor=colors[i], color=colors[i])
    axs[1, 1].legend(loc='upper right', prop={'size':5})
    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/mvr_comparison.pdf')


def get_decoupled_rmsd(k=1, normalize=True):
    dt = 10.0

    corrs_all = {}
    decouple_params = ['abcd', 'ab'] #['abcd', 'ab', 'a', 'c']
    for p in decouple_params:
        print "loading %s" % p
        corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt)
        corrs_all[p] = corrs
    for p in decouple_params:
        print "loading %s" % p
        corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt, decouple=True)
        corrs_all[p + '_dec'] = corrs
    values = np.zeros((len(decouple_params), 2, 2))
    values_err = np.zeros((len(decouple_params), 2, 2))
    for index_correlation in range(2):
        for j, p in enumerate(decouple_params):
                name = p
                corrs = corrs_all[name]
                errs = np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]
                means = corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]

                if normalize:
                    scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                    print "This needs to be number of simulations:"
                    print corrs.mean(axis=0)[100:, index_correlation].mean(axis=0).shape
                    scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                    means, errs = get_similarity(scale, scale_err, means, errs, index_correlation=index_correlation)
                values[j, 0, index_correlation] = means[k]
                values_err[j, 0, index_correlation] = errs[k]

                name = p + '_dec'
                corrs = corrs_all[name]

                values[j, 1, index_correlation] = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                values_err[j, 1, index_correlation] = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
    return values, values_err, decouple_params


def plot_stims(normalize=False):
    """
    Plot noisy current injection increase analysis
    :param normalize:
    :return:
    """
    dt = 10.0
    variances = ['0p01', '0p05', '0p1', '0p5', '1p0', '1p5','2p0','10p0']
    variances_plot = ['0p001'] + variances
    variances_float = [float(s.replace('p', '.')) for s in variances_plot]
    parameters_change = ['abcd', 'd']
    abcd_corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, decouple=True)

    mean_abcd = abcd_corrs.mean(axis=0).mean(axis=-1)[100:, :].mean(axis=0)

    all_corrs = {}
    for variance in variances_plot:
        for parameters in parameters_change:
            if variance == '0p001':
                corrs, bins = get_correlations(parameter_continue='', parameter_change=parameters, dt=dt)
            else:
                corrs, bins = get_correlations(parameter_continue='_stim', parameter_change='%s_stim' % parameters, dt=dt, variance=variance)

            all_corrs[variance + '_' + parameters] = corrs
        for parameters in parameters_change:
            if variance == '0p001':
                corrs, bins = get_correlations(parameter_continue='', parameter_change=parameters, dt=dt, decouple=True, n=10)
            else:
                corrs, bins = get_correlations(parameter_continue='', parameter_change=parameters, dt=dt, decouple=True, variance=variance,  n=10)
            all_corrs[variance + '_' + parameters + '_decouple'] = corrs

    bins -= (2000.0 + 10.0/2)
    bins[0] = 0

    fig, axs = plt.subplots(5, 2, figsize=(8, 14))
    colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd', 'black'][::-1]

    values_other, values_other_err, values_other_params = get_decoupled_rmsd(1, normalize=normalize)

    converged_means = []
    converged_errs = []
    converged_means_abcd = []
    converged_errs_abcd = []
    for index_correlation in range(2):
        n = 21
        decay_times_abcd = []
        decay_times_d = []
        decay_times_abcd_err = []
        decay_times_d_err = []
        for j, (parameters, variances_p) in enumerate(zip(['abcd', 'd'], [variances_plot, variances_plot])):
            ax = axs[j, index_correlation]
            for i, variance in enumerate(variances_plot):
                name = variance + '_' + parameters
                corrs = all_corrs[name]
                errs = np.hstack([np.array([0]),
                                         np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
                means = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

                if normalize:
                    scale = corrs.mean(axis=0).mean(axis=-1)[101:, index_correlation].mean()
                    print "This needs to be number of simulations:"
                    print corrs.mean(axis=0)[100:, index_correlation].mean(axis=0).shape
                    scale_err = mean_confidence_interval(corrs.mean(axis=0)[101:, index_correlation].mean(axis=0))
                    means, errs = get_similarity(scale, scale_err, means, errs, index_correlation=index_correlation)

                if j == 0:
                    decay_times_abcd.append(means[2])
                    decay_times_abcd_err.append(errs[2])

                elif j == 1:
                    decay_times_d.append(means[2])
                    decay_times_d_err.append(errs[2])

                ax.errorbar(bins[:n], means[:n], yerr=errs[:n], color=colors[i], label=name)
        n = 150


        for i, variance in enumerate(variances_plot):
            ax = axs[2, index_correlation]
            name = variance + '_' + 'd' + '_decouple'

            corrs = all_corrs[name]
            errs = np.hstack([np.array([0]),
                                         np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means = np.hstack([np.array([0]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            ax.errorbar(bins[:n], means[:n], yerr=errs[:n], color=colors[i])
            if index_correlation == 0:
                converged_means.append(means[101:].mean())
                converged_errs.append(mean_confidence_interval(means[101:]))


        for i, variance in enumerate(variances_plot):
            ax = axs[2, index_correlation]
            name = variance + '_' + 'abcd' + '_decouple'

            corrs = all_corrs[name]
            errs = np.hstack([np.array([0]),
                                         np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means = np.hstack([np.array([0]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            ax.errorbar(bins[:n], means[:n], yerr=errs[:n], color=colors[i])
            if index_correlation == 0:
                converged_means_abcd.append(means[101:].mean())
                converged_errs_abcd.append(mean_confidence_interval(means[101:]))

        ax = axs[3, index_correlation]

        ax.plot(np.arange(len(variances_float)), converged_means, marker='x')
        ax.plot(np.arange(len(variances_float)), converged_means_abcd, marker='x')

        ax.set_xticks(np.arange(len(variances_float)))
        ax.set_xticklabels(variances_float)

        ax = axs[4, index_correlation]
        ax.errorbar(converged_means, decay_times_abcd, xerr=converged_errs, yerr=decay_times_abcd_err, marker='.', label='abcd', color='black')
        ax.errorbar(converged_means, decay_times_d, xerr=converged_errs, yerr=decay_times_d_err, marker='.', label='d', linestyle='--', color='black')


        # ax.errorbar(values_other[:, 1, 0], values_other[:, 0, index_correlation], xerr=values_other_err[:, 1, 0], yerr=values_other_err[:, 0, index_correlation], marker='^',
        #             linestyle=' ', color='black')

        ax.plot([values_other[1, 1, 0], values_other[1, 1, 0]],
                [0, 1], color='black')
        ax.fill_between([values_other[1, 1, 0] - values_other_err[1, 1, 0], values_other[1, 1, 0] + values_other_err[1, 1, 0]],
                [0, 0], [1, 1], color='black', linewidth=0, alpha=0.3)
        ax.plot([0, 5],
                [values_other[1, 0, index_correlation], values_other[1, 0, index_correlation]], color='black')
        ax.fill_between([0, 5],
                np.array([values_other[1, 0, index_correlation], values_other[1, 0, index_correlation]]) - values_other_err[1, 0, index_correlation],
                np.array([values_other[1, 0, index_correlation], values_other[1, 0, index_correlation]]) + values_other_err[1, 0, index_correlation],
                color='black', linewidth=0, alpha=0.3)


    for ax in axs[:, 1].flatten():
        ax.set_ylabel('r')
    for ax in axs[:, 0].flatten():
        ax.set_ylabel('RMSD (mV)')
    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    axs[-2, 0].set_xlabel('Percent variance')
    axs[-2, 1].set_xlabel('Percent variance')
    axs[-1, 0].set_xlabel('RMSD (mV)')
    axs[-1, 1].set_xlabel('r')

    plt.tight_layout()
    plt.savefig('figures/stim_comparison_norm%s.pdf' % normalize)


    fig, ax = plt.subplots()

    d_noise = np.array([0] + converged_means)
    ab_noise = values_other[1, 1, 0]
    abcd_noise = values_other[0, 1, 0]

    print d_noise
    comb_noise_ab = np.sqrt((d_noise**2 + ab_noise**2))
    comb_noise_abcd = np.sqrt((d_noise**2 + abcd_noise**2))



    ax.plot(d_noise, comb_noise_ab)
    ax.plot(d_noise, comb_noise_abcd)
    ax.plot(d_noise[1:], converged_means_abcd, '.--')
    ax.plot([0, d_noise[-1]], [ab_noise, ab_noise])

    plt.savefig('figures/noise_addition.pdf')


def plot_cells_divergence_example():
    dt = 10.0
    n = 20
    fig, axs = plt.subplots(4, 2, figsize=(8, 6))

    circuit = bluepy.Simulation(get_base_bcs()[0]).circuit
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    gids_all = np.array(cells.axes[0])

    gids = [80114, 77991, 84263, 78162] #
    indices = []
    for gid in gids:
        indices.append(np.where(gids_all == gid)[0][0])
    indices = np.array(indices)
    print indices

    mtypes = ['L5_TTPC1', 'L5_TTPC2', 'L6_TPC_L4', 'L5_STPC']
    corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=0, n=40)
    bins -= (2000 - dt/2)

    for i, id in enumerate(indices):
        for j in range(2):
            ax = axs[i, j]
            ax.plot(bins[:n], corrs[id, :n, j, :5], alpha=0.3)
            ax.plot(bins[:n], corrs[id, :n, j, :].mean(axis=-1), alpha=1, color='red')
            ax.set_title(mtypes[i])
            ax.plot(0, j, 'o', color='red', markersize=2.0)



    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/cells_divergence_example.pdf')


def plot_O1():
    dt = 10.0
    n = 200
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for index_correlation in range(2):

        means = {}
        errs = {}
        cas = ['1p25', 'O1']
        labels = ['mc2', 'O1']

        for ca in cas:
            print "loading %s" % ca
            if ca == '1p25':
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, shuffle_level=0, n=40)
            elif ca == 'O1':
                corrs, bins = get_correlations(parameter_continue='_O1', parameter_change='abcd_O1', dt=dt)

            bins -= (2000 + dt/2)
            bins[0] = 0
            errs[ca] = np.hstack([np.array([0]),
                                     np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means[ca] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

        means_df = pd.DataFrame(means, index=bins[:])
        errs_df = pd.DataFrame(errs, index=bins[:])

        lss =  ['--', '-', '--', '-'] #['--', '--', '--', '-']
        colors = ['black', '#8856a7', '#8856a7', '#8856a7']
    #    plot_points =
        ax = axs[0, index_correlation]
        n = 15
        for i, ca in enumerate(cas):
            ax.fill_between(bins[0:(n+1)], np.array(means_df[ca])[0:(n+1)] - np.array(errs_df[ca])[0:(n+1)],
                            np.array(means_df[ca])[0:(n+1)] + np.array(errs_df[ca])[0:(n+1)],
                           label=labels[i], facecolor=colors[i], alpha=0.3)
        means = []
        for i, ca in enumerate(cas):
            if index_correlation == 1:
                means.append(np.array(means_df[ca])[0:(n + 1)])
            ax.plot(bins[0:(n+1)], np.array(means_df[ca])[0:(n+1)],
                            linestyle=lss[i], label=labels[i], linewidth=0.8, color=colors[i], markersize=2.0)
        if index_correlation == 1:
            print means[0] - means[1]
        ax = axs[1, index_correlation]
        n = 75
        for i, ca in enumerate(cas):
            ax.fill_between(bins[0:(n+1)], np.array(means_df[ca])[0:(n+1)] - np.array(errs_df[ca])[0:(n+1)],
                            np.array(means_df[ca])[0:(n+1)] + np.array(errs_df[ca])[0:(n+1)],
                           label=labels[i], facecolor=colors[i], alpha=0.3)
        for i, ca in enumerate(cas):

            ax.plot(bins[0:(n+1)], np.array(means_df[ca])[0:(n+1)],
                            linestyle=lss[i], label=labels[i], linewidth=0.8, color=colors[i], markersize=2.0)

    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/O1_comparison.pdf')


def plot_O1_difference_frs():
    dt = 5.0
    n = 40
    fig, ax = plt.subplots()

    shuffle = 0
    frs_mc2, bins_2 = get_firing_rates(parameter_continue='', parameter_change='abcd', dt=dt, n=40, shuffle_level=shuffle)
    frs_O1, bins_2 = get_firing_rates(parameter_continue='_O1', parameter_change='abcd_O1', dt=dt, shuffle_level=shuffle)



    fr_conversion = 1/31346.0 * 1000.0/dt
    bins_2 += dt/2
    frs_mc2 *= fr_conversion
    frs_O1 *= fr_conversion
    print frs_mc2.shape
    print frs_O1.shape

    #
    # for i in range(40):
    #     ax.plot(bins_2[:n], frs_mc2[i, :n], color='black', linewidth=0.8, alpha=0.3)
    # for i in range(20):
    #     ax.plot(bins_2[:n], frs_O1[i, :n], color='black', linewidth=0.8, alpha=0.3)


    ax.errorbar(bins_2[:n]-0.5, frs_mc2.mean(axis=0)[:n], yerr=np.apply_along_axis(mean_confidence_interval, 0, frs_mc2)[:n], label='mc_2')
    ax.errorbar(bins_2[:n]+0.5, frs_O1.mean(axis=0)[:n], yerr=np.apply_along_axis(mean_confidence_interval, 0, frs_O1)[:n], label='O1')
    ax.set_ylabel('RMSD(FR) Hz')
    ax.set_xlabel('t (ms)')
    ax.legend()
    plt.savefig('figures/frs_diff.pdf')


def plot_O1_difference():
    dt = 10.0
    n = 200

    cas = ['1p25', 'O1']
    labels = ['mc2', 'O1']

    corrs_mc2, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, n=40)
    corrs_O1, bins = get_correlations(parameter_continue='_O1', parameter_change='abcd_O1', dt=dt)


    bins -= (2000 - dt/2.0)
    bins[0] = 0

    circuit = bluepy.Simulation(get_base_bcs()[0]).circuit
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    xs = np.array(cells['x'])
    zs = np.array(cells['z'])
    ds = np.sqrt((xs - xs.mean())**2 + (zs - zs.mean())**2)
    ds_k = np.digitize(ds, [50, 100, 150, 200])

    fig, ax = plt.subplots()
    colors_scatter = ['#eff3ff',
                      '#bdd7e7',
                      '#6baed6',
                      '#3182bd',
                      '#08519c']
    for k in range(5):
        ax.plot(xs[ds_k == k], zs[ds_k == k], linestyle='', marker='.', color=colors_scatter[k], alpha=0.5, rasterized=True,
                markeredgecolor=None)
    ax.set_aspect('equal')
    plt.savefig('figures/scatter_distances.pdf', dpi=300)


    fig, axs = plt.subplots(6, 2, figsize=(7, 10))
    for index_correlation in range(2):
        for j, time_step in enumerate([4, 24, 49]):
            ax = axs[j, index_correlation]
            diffs = (corrs_mc2.mean(axis=-1) - corrs_O1.mean(axis=-1))[:, time_step, index_correlation]
            #ax.hist(diffs)
            ax.scatter(ds, diffs, alpha=0.3, s=1.0, c='black', rasterized=True)

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ds, diffs)
            print slope
            ax.plot(np.linspace(0, 250, 100), np.linspace(0, 250, 100) * 0.0, '--', color='#a6d96a')
            ax.plot(np.linspace(0, 250, 100), np.linspace(0, 250, 100) * slope + intercept, '--', color='red')
            ax.set_title('%.e(%.e), p = %.e' % (slope, std_err, p_value))

    n_series = 10

    for index_correlation in range(2):
        slopes = np.zeros(n_series)
        errs = np.zeros(n_series)
        diffs_series = np.zeros(n_series)
        diffs_series_split = np.zeros((n_series, 5))
        diffs_series_split_err = np.zeros((n_series, 5))

        diffs_series_adjusted = np.zeros(n_series)

        for j, time_step in enumerate(range(n_series)):
            print time_step
            diffs = (corrs_mc2.mean(axis=-1) - corrs_O1.mean(axis=-1))[:, time_step, index_correlation]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ds, diffs)
            slopes[j] = slope
            errs[j] = std_err
            diffs_series[j] = diffs.mean()
            for k in range(5):
                print diffs.shape
                diffs_series_split[j, k] = diffs[ds_k == k].mean()
                diffs_series_split_err[j, k] = mean_confidence_interval(diffs[ds_k == k])

            diffs_series_adjusted[j] = (diffs - (ds * slope + intercept)).mean()
        ax = axs[-3, index_correlation]
        ax.errorbar(np.arange(1, n_series + 1) * dt - dt/2, slopes, yerr=errs, color='red')
        ax = axs[-2, index_correlation]
        ax.plot(np.arange(1, n_series + 1) * dt - dt/2, diffs_series, color='red')
        ax.plot(np.arange(1, n_series + 1) * dt - dt/2, diffs_series_adjusted, color='green')
        print diffs_series_adjusted
        ax = axs[-1, index_correlation]
        ax.errorbar(np.arange(5) * 50, diffs_series_split[1, :], yerr=diffs_series_split_err[1, :])

    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('d (um)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/O1_difference.pdf', dpi=300)


def plot_sample_increase():
    dt = 10.0
    n = 200
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    for index_correlation in range(2):

        means = {}
        errs = {}
        cas = ['20', '40']
        labels = ['n = 20', 'n = 40']

        for ca in cas:
            print "loading %s" % ca
            if ca == '20':
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, n=20)
            elif ca == '40':
                corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, n=40)

            bins -= 2000.5
            errs[ca] = np.hstack([np.array([0]),
                                     np.apply_along_axis(mean_confidence_interval, -1, corrs.mean(axis=0))[:, index_correlation]])
            means[ca] = np.hstack([np.array([index_correlation]), corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])

        means_df = pd.DataFrame(means, index=bins[:])
        errs_df = pd.DataFrame(errs, index=bins[:])

        colors = ['black', '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        symbols = ['v', 's', '^', '.']
        lss =  ['--', '-', '--', '-'] #['--', '--', '--', '-']
        colors = ['black', '#d53e4f', '#66c2a5', '#3288bd']
    #    plot_points =
        ax = axs[0, index_correlation]
        n = 15
        for i, ca in enumerate(cas):
            ax.fill_between(bins[0:(n+1)] - dt/2, np.array(means_df[ca])[0:(n+1)] - np.array(errs_df[ca])[0:(n+1)],
                            np.array(means_df[ca])[0:(n+1)] + np.array(errs_df[ca])[0:(n+1)],
                           label=labels[i], facecolor=colors[i], alpha=0.3)
        for i, ca in enumerate(cas):
            ax.plot(bins[0:(n+1)] - dt/2, np.array(means_df[ca])[0:(n+1)],
                            linestyle=lss[i], label=labels[i], linewidth=0.8, color=colors[i], markersize=2.0)

        ax = axs[1, index_correlation]
        n = 50
        for i, ca in enumerate(cas):
            ax.fill_between(bins[0:(n+1)] - dt/2, np.array(means_df[ca])[0:(n+1)] - np.array(errs_df[ca])[0:(n+1)],
                            np.array(means_df[ca])[0:(n+1)] + np.array(errs_df[ca])[0:(n+1)],
                           label=labels[i], facecolor=colors[i], alpha=0.3)
        for i, ca in enumerate(cas):
            ax.plot(bins[0:(n+1)] - dt/2, np.array(means_df[ca])[0:(n+1)],
                            linestyle=lss[i], label=labels[i], linewidth=0.8, color=colors[i], markersize=2.0)

    for ax in axs.flatten():
        ax.legend(loc='upper right', prop={'size':5})
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('figures/sample_comparison.pdf')

def compare_thresholded():

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    for kernel in range(2):
        det = 0
        for index_correlation in range(2):
            means = {}
            errs = {}
            for p in ['abcd']:
                corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=5.0, kernel=kernel)
                bins -= 2000.0
                errs[p] = np.nanmean(corrs, axis=0).std(axis=-1, ddof=1)[:, index_correlation]/np.sqrt(corrs.shape[-1])
                means[p] = np.nanmean(corrs, axis=0).mean(axis=-1)[:, index_correlation]
            means_df = pd.DataFrame(means, index=bins[1:])
            errs_df = pd.DataFrame(errs, index=bins[1:])

            ax = axs[index_correlation, 0]
            for p in ['abcd']:
                #print np.array(means_df[p])[:100]
                ax.errorbar(bins[1:101], np.array(means_df[p])[:100], yerr=np.array(errs_df[p])[:100], label=p + str(kernel), linewidth=0.8)
            ax.legend(loc='upper right')



    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('r')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()

    plt.savefig('soma_corrs_threshold.pdf')


def plot_median_filter_example_figure():
    """
    plot for paper
    :return:
    """
    kss = [21, 31, 41, 51]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for i, ks in enumerate(kss):
        ax = axs.flatten()[i]

        gids = [75077, 78162, 84263, 80284, 70975, 78728, 76689, 67224]

        t_window = 200

        n_seed = 172
        t_middle = 2000
        shift = 2000
        lims = np.array([-67.5, -52.5])
        lw=0.8

        soma = bluepy.Simulation(get_base_simulation(n_seed)).v2.reports['soma']
        time_range = soma.time_range[soma.time_range >= t_middle - t_window] - shift
        data = soma.data(t_start=t_middle - t_window, gids=gids)

        for j, gid in enumerate(gids):
            ax.plot(time_range, data.loc[gid], color='#2c7fb8', linewidth=lw)
            ax.plot(time_range, medfilt(data.loc[gid], kernel_size=ks), color='black', linewidth=0.5)



        soma = bluepy.Simulation(get_continue_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
        for j, gid in enumerate(gids):
            ax.plot(time_range_cont, data_cont.loc[gid], color='#41b6c4', linewidth=lw)
            ax.plot(time_range_cont, medfilt(data_cont.loc[gid], kernel_size=ks), color='black', linewidth=0.5)

        #ax.set_ylim(lims)
        ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)
        ax.set_title('kernel time = %.1f ms' % (ks * 0.1))
    plt.savefig('median_filter.pdf')


def plot_soma_voltage_example_figure():
    """
    plot for paper
    :return:
    """

    gids = [80114, 77991, 84263, 78162] #
    mtypes = ['L5_TTPC1', 'L5_TTPC2', 'L6_TPC_L4', 'L5_STPC']

    fig, axs = plt.subplots(5, 2, figsize=(8, 7))
    t_window = 1000

    basec= '#1d91c0'
    continuec = '#41b6c4'
    changec = '#7fcdbb'

    for x, n_seed in enumerate([172, 188]):
        t_middle = 2000
        shift = 2000
        lims = np.array([-70, -50])
        lw=0.8

        soma = bluepy.Simulation(get_base_simulation(n_seed)).v2.reports['soma']
        time_range = soma.time_range[soma.time_range >= t_middle - t_window] - shift
        data = soma.data(t_start=t_middle - t_window, gids=gids)
        print data.shape
        data = np.array(data)
        for k in range(4):
            ax = axs[k, x]
            ax.plot(time_range, data[k, :], color=basec, linewidth=lw)


        soma = bluepy.Simulation(get_continue_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
        data_cont = np.array(data_cont)

        for k in range(4):
            ax = axs[k, x]
            ax.set_title(mtypes[k])
            ax.plot(time_range_cont, data_cont[k, :], color=continuec, linewidth=lw)
            ax.set_ylim(lims)
            ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)

        soma = bluepy.Simulation(get_change_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
        data_cont = np.array(data_cont)

        for k in range(4):
            ax = axs[k, x]
            ax.plot(time_range_cont, data_cont[k, :], color=changec, linewidth=lw)
            ax.set_ylim(lims)
            ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)

        ax = axs[-1, x]
        lw = 0.6
        ax.set_ylim([-63, -60])
#        t_window = 250
        soma = bluepy.Simulation(get_base_simulation(n_seed)).v2.reports['soma']
        time_range = soma.time_range[soma.time_range >= t_middle - t_window] - shift
        data = soma.data(t_start=t_middle - t_window)
        ax.plot(time_range, data.mean(axis=0), color=basec, linewidth=lw)
        soma = bluepy.Simulation(get_continue_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window)
        ax.plot(time_range_cont, data_cont.mean(axis=0), color=continuec, linewidth=lw)
        soma = bluepy.Simulation(get_change_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window)
        ax.plot(time_range_cont, data_cont.mean(axis=0), color=changec, linewidth=lw)
        ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('V (mV)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/soma_voltage_example_1.pdf')


def plot_soma_voltage_example_figure_dNAC():
    """
    plot for paper
    :return:
    """
    circuit = bluepy.Simulation(get_base_bcs()[0]).circuit
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2, Cell.ETYPE: 'dNAC'})
    gids_dNAC = np.array(cells.axes[0])
    np.random.seed(1989)
    gids = np.random.choice(gids_dNAC, size=4, replace=False)
    mtypes = ['L5_TTPC1', 'L5_TTPC2', 'L6_TPC_L4', 'L5_STPC']

    fig, axs = plt.subplots(5, 2, figsize=(8, 7))
    t_window = 1000

    basec= '#1d91c0'
    continuec = '#41b6c4'
    changec = '#7fcdbb'

    for x, n_seed in enumerate([172, 188]):
        t_middle = 2000
        shift = 2000
        lims = np.array([-70, -50])
        lw=0.8

        soma = bluepy.Simulation(get_base_simulation(n_seed)).v2.reports['soma']
        time_range = soma.time_range[soma.time_range >= t_middle - t_window] - shift
        data = soma.data(t_start=t_middle - t_window, gids=gids)
        print data.shape
        data = np.array(data)
        for k in range(4):
            ax = axs[k, x]
            ax.plot(time_range, data[k, :], color=basec, linewidth=lw)


        soma = bluepy.Simulation(get_continue_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
        data_cont = np.array(data_cont)

        for k in range(4):
            ax = axs[k, x]
            ax.set_title(mtypes[k])
            ax.plot(time_range_cont, data_cont[k, :], color=continuec, linewidth=lw)
            ax.set_ylim(lims)
            ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)

        soma = bluepy.Simulation(get_change_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
        data_cont = np.array(data_cont)

        for k in range(4):
            ax = axs[k, x]
            ax.plot(time_range_cont, data_cont[k, :], color=changec, linewidth=lw)
            ax.set_ylim(lims)
            ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)

        ax = axs[-1, x]
        lw = 0.6
        ax.set_ylim([-63, -60])
#        t_window = 250
        soma = bluepy.Simulation(get_base_simulation(n_seed)).v2.reports['soma']
        time_range = soma.time_range[soma.time_range >= t_middle - t_window] - shift
        data = soma.data(t_start=t_middle - t_window, gids=gids_dNAC)
        ax.plot(time_range, data.mean(axis=0), color=basec, linewidth=lw)
        soma = bluepy.Simulation(get_continue_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids_dNAC)
        ax.plot(time_range_cont, data_cont.mean(axis=0), color=continuec, linewidth=lw)
        soma = bluepy.Simulation(get_change_simulation(n_seed)).v2.reports['soma']
        time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - shift
        data_cont = soma.data(t_end=t_middle + t_window, gids=gids_dNAC)
        ax.plot(time_range_cont, data_cont.mean(axis=0), color=changec, linewidth=lw)
        ax.plot(np.array([t_middle, t_middle]) - shift, lims, '--', color='red', linewidth=0.5)

    for ax in axs.flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('V (mV)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/soma_voltage_example_dNAC.pdf')


def plot_spike_raster_example_figure():
    """
    plot for paper
    :return:
    """

    params = [['abcd', '', 'abcd'],
              ['gv2bcd', '_gv2', 'gv2bcd']]
    n_seeds = [189, 180]


    t_middle = 2000
    t_window = 250
    fig, axs = plt.subplots(4, 2, figsize=(10, 6))


    for k in range(2):
        n_seed = n_seeds[k]
        n_bc = n_seed - 170

        base_bcs = get_base_bcs(params=params[k][0])
        continue_bcs = get_continue_bcs(params=params[k][1])
        change_bcs = get_change_bcs(params=params[k][2])

        circuit = bluepy.Simulation(base_bcs[0]).circuit
        cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
        ys = np.array(cells['y'])
        gids = np.array(list(bluepy.Simulation(base_bcs[0]).get_circuit_target()))
        sort_idx = np.argsort(ys)
        sort_dict = dict(zip(sort_idx + gids.min(), np.arange(gids.size)))

        gids = np.array(list(bluepy.Simulation(base_bcs[n_bc]).get_circuit_target()))

        ax = axs[0,k]
        spikes = bluepy.Simulation(base_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_start=t_middle - t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df) - t_middle
        ax.vlines(times, gids_spiking, gids_spiking + 200, rasterized=True, lw=0.3)
        ax2 = ax.twinx()
        ax2.hist(times, bins=np.linspace(-t_window, 0, 26), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)
        ax2.set_ylabel('FR (Hz)')
        ax2.set_ylim([0, 3])
        ax2.set_yticks([0, 1, 2, 3])

        spikes = bluepy.Simulation(continue_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_end=t_middle+t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df)
        ax.vlines(times, gids_spiking, gids_spiking + 200, rasterized=True, lw=0.3)
        ax2.hist(times, bins=np.linspace(0, t_window, 26), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)

        ax = axs[1,k]
        spikes = bluepy.Simulation(base_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_start=t_middle - t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df) - t_middle
        ax.vlines(times, gids_spiking, gids_spiking + 200, rasterized=True, lw=0.3)
        ax2 = ax.twinx()
        ax2.hist(times, bins=np.linspace(-t_window, 0, 26), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)
        ax2.set_ylabel('FR (Hz)')
        ax2.set_ylim([0, 3])
        ax2.set_yticks([0, 1, 2, 3])

        spikes = bluepy.Simulation(change_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_end=t_middle+t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df)
        ax.vlines(times, gids_spiking, gids_spiking + 200, rasterized=True, lw=0.3)
        ax2.hist(times, bins=np.linspace(0, t_window, 26), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)

        # gids= [75077, 78162, 84263, 80284, 70975, 78728, 76689, 67224]
        ax = axs[2, k]
        for bc in base_bcs[:5]:
            soma = bluepy.Simulation(bc).v2.reports['soma']
            time_range = soma.time_range[soma.time_range >= t_middle - t_window] - t_middle
            data = soma.data(t_start=t_middle - t_window, gids=gids)
            ax.plot(time_range, data.mean(axis=0), linewidth=1, alpha=0.5, rasterized=True, color='#1f77b4')
        for bc in continue_bcs[:5]:
            soma = bluepy.Simulation(bc).v2.reports['soma']
            time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - t_middle
            data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
            ax.plot(time_range_cont, data_cont.mean(axis=0), linewidth=1, alpha=0.5, rasterized=True, color='#ff7f0e')
        for bc in change_bcs[:5]:
            soma = bluepy.Simulation(bc).v2.reports['soma']
            time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - t_middle
            data_cont = soma.data(t_end=t_middle + t_window, gids=gids)
            ax.plot(time_range_cont, data_cont.mean(axis=0), linewidth=1, alpha=0.5, rasterized=True, color='#ff7f0e')

        ax_s = [axs[3, k].twinx(), axs[3, k]]
        n = 25
        lines = ['-', '--']
        for index_correlation in range(2):
            ax = ax_s[index_correlation]
            corrs, bins = get_correlations(parameter_continue=params[k][1], parameter_change=params[k][2], dt=10.0)
            errs = np.hstack([np.zeros(n), corrs.mean(axis=0).std(axis=-1, ddof=1)[:, index_correlation]])
            means = np.hstack([np.zeros(n) + index_correlation, corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            xs = np.hstack([-(bins[1:(n+1)] - t_middle)[::-1], bins[:(n+1)] - t_middle]) + (bins[1] - bins[0])/2.0
            ys = means[:(n+n+1)]
            print xs.shape
            print ys.shape
            ax.errorbar(xs, ys, yerr=errs[:(n+n+1)],
                        linestyle=lines[index_correlation], linewidth=0.8, color='black', marker='.', markersize='3')
        ax_s[0].set_ylim([0, 3.3])
        ax_s[0].set_yticks([0, 1, 2, 3])

        ax_s[1].set_ylim([0, 1.1])
        ax_s[1].set_yticks([0, 0.5, 1])

        ax_s[0].set_ylabel('RMSD (mV)')
        ax_s[1].set_ylabel('r')



    for ax in axs[:2, :].flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylim([0, 31346])
        ax.set_yticks([0, 10000, 20000, 30000])

        ax.set_ylabel('Neuron')
        ax.set_xlim([-t_window, t_window])

    for ax in axs[2, :].flatten():
        ax.set_xlabel('t (ms)')
        #ax.set_ylim([-63, -60])
        #ax.set_yticks([-63, -62, -61, -60])

        ax.set_ylabel('V (mV)')
        ax.set_xlim([-t_window, t_window])

    for ax in axs[3, :].flatten():
        ax.set_xlabel('t (ms)')

        ax.set_xlim([-t_window, t_window])
    plt.tight_layout()
    plt.savefig('figures/raster_example.pdf', dpi=300)


def plot_spike_raster_example_figure_ca_scan():
    """
    plot for paper
    :return:
    """

    params = [['ca_abcd', '_ca_abcd', 'ca_abcd'],
              ['gv2bcd', '_gv2', 'gv2bcd']]
    n_seeds = [189, 170, 177]


    t_middle = 2000
    t_window = 500
    fig, axs = plt.subplots(4, 3, figsize=(15, 6))

    for k in range(3):
        n_seed = n_seeds[k]
        n_bc = n_seed - 170

        base_bcs = get_base_bcs_ca(params=params[0][0], ca=cas[k])
        continue_bcs = get_continue_bcs_ca(params=params[0][1], ca=cas[k])
        change_bcs = get_change_bcs_ca(params=params[0][2], ca=cas[k])

        circuit = bluepy.Simulation(base_bcs[0]).circuit
        cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
        ys = np.array(cells['y'])
        gids = np.array(list(bluepy.Simulation(base_bcs[0]).get_circuit_target()))
        sort_idx = np.argsort(ys)
        sort_dict = dict(zip(sort_idx + gids.min(), np.arange(gids.size)))

        gids = np.array(list(bluepy.Simulation(base_bcs[n_bc]).get_circuit_target()))

        ax = axs[0,k]
        spikes = bluepy.Simulation(base_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_start=t_middle - t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df) - t_middle
        ax.vlines(times, gids_spiking, gids_spiking + 100, rasterized=True, lw=0.15)
        ax2 = ax.twinx()
        ax2.hist(times, bins=np.linspace(-t_window, 0, 51), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)
        ax2.set_ylabel('FR (Hz)')
        #ax2.set_ylim([0, 3])
        #ax2.set_yticks([0, 1, 2, 3])

        spikes = bluepy.Simulation(continue_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_end=t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df)
        ax.vlines(times, gids_spiking, gids_spiking + 100, rasterized=True, lw=0.15)
        ax2.hist(times, bins=np.linspace(0, t_window, 51), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)

        ax = axs[1,k]
        spikes = bluepy.Simulation(base_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_start=t_middle - t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df) - t_middle
        ax.vlines(times, gids_spiking, gids_spiking + 100, rasterized=True, lw=0.15)
        ax2 = ax.twinx()
        ax2.hist(times, bins=np.linspace(-t_window, 0, 51), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)
        ax2.set_ylabel('FR (Hz)')
        #ax2.set_ylim([0, 3])

        spikes = bluepy.Simulation(change_bcs[n_bc]).v2.reports['spikes']
        df = spikes.data(t_end=t_window)
        gids_spiking = np.array(df.axes[0])
        gids_spiking = np.vectorize(sort_dict.get)(gids_spiking)
        times = np.array(df)
        ax.vlines(times, gids_spiking, gids_spiking + 100, rasterized=True, lw=0.15)
        ax2.hist(times, bins=np.linspace(0, t_window, 51), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)

        ax = axs[2, k]
        for bc in base_bcs:
            soma = bluepy.Simulation(bc).v2.reports['soma']
            time_range = soma.time_range[soma.time_range >= t_middle - t_window] - t_middle
            data = soma.data(t_start=t_middle - t_window)
            ax.plot(time_range, data.mean(axis=0), linewidth=1, alpha=0.5, rasterized=True, color='#1f77b4')
        for bc in continue_bcs:
            soma = bluepy.Simulation(bc).v2.reports['soma']
            time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - t_middle
            data_cont = soma.data(t_end=t_middle + t_window)
            ax.plot(time_range_cont, data_cont.mean(axis=0), linewidth=1, alpha=0.5, rasterized=True, color='#ff7f0e')
        for bc in change_bcs:
            soma = bluepy.Simulation(bc).v2.reports['soma']
            time_range_cont = soma.time_range[soma.time_range < t_middle + t_window] - t_middle
            data_cont = soma.data(t_end=t_middle + t_window)
            ax.plot(time_range_cont, data_cont.mean(axis=0), linewidth=1, alpha=0.5, rasterized=True, color='#ff7f0e')

        ax_s = [axs[3, k].twinx(), axs[3, k]]
        n = 50
        lines = ['-', '--']
        for index_correlation in range(2):
            ax = ax_s[index_correlation]
            corrs, bins = get_correlations(parameter_continue=params[0][1], parameter_change=params[0][2], dt=10.0, ca=cas[k])
            errs = np.hstack([np.zeros(n), corrs.mean(axis=0).std(axis=-1, ddof=1)[:, index_correlation]]) #/np.sqrt(corrs.shape[-1])])
            means = np.hstack([np.zeros(n) + index_correlation, corrs.mean(axis=0).mean(axis=-1)[:, index_correlation]])
            xs = np.hstack([-(bins[1:(n+1)] - t_middle)[::-1], bins[:(n+1)] - t_middle]) + (bins[1] - bins[0])/2.0
            ys = means[:(n+n+1)]
            print xs.shape
            print ys.shape
            ax.errorbar(xs, ys, yerr=errs[:(n+n+1)]/np.sqrt(20),
                        linestyle=lines[index_correlation], linewidth=0.8, color='black')
        ax_s[0].set_ylim([0, 5.3])
        ax_s[0].set_yticks([0, 1, 2, 3, 4, 5])

        ax_s[1].set_ylim([0, 1.1])
        ax_s[1].set_yticks([0, 0.5, 1])

        ax_s[0].set_ylabel('RMSD (mV)')
        ax_s[1].set_ylabel('r')



    for ax in axs[:2, :].flatten():
        ax.set_xlabel('t (ms)')
        ax.set_ylim([0, 31346])
        ax.set_yticks([0, 10000, 20000, 30000])

        ax.set_ylabel('Neuron')
        ax.set_xlim([-t_window, t_window])

    for ax in axs[2, :].flatten():
        ax.set_xlabel('t (ms)')
        #ax.set_ylim([-63, -60])
        #ax.set_yticks([-63, -62, -61, -60])

        ax.set_ylabel('V (mV)')
        ax.set_xlim([-t_window, t_window])

    for ax in axs[3, :].flatten():
        ax.set_xlabel('t (ms)')

        ax.set_xlim([-t_window, t_window])
    plt.tight_layout()
    plt.savefig('figures/raster_example_3_ca_scan.pdf', dpi=300)


def plot_time_bins_fitting():
    fig, ax = plt.subplots(1, figsize=(5, 4))
    norms, times = get_initial_divergence(parameter_continue='', parameter_change='abcd')
    print norms.shape
    times -= 2000
    ax.plot(np.vstack([times for i in range(40)]).T, norms.T, alpha=0.3, color='#a1d76a', lw=0.5)
    ax.plot(times, norms.mean(axis=0), '-', color='red', lw=1.0)
    ax.plot(times, norms.mean(axis=0) - norms.std(axis=0), '--', color='red', lw=1.0)
    ax.plot(times, norms.mean(axis=0) + norms.std(axis=0), '--', color='red', lw=1.0)

    ax.legend(loc='upper right', prop={'size':6})
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('RMSD or abs(diff)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/time_bins_fitting.pdf')


def lorenz_func(t, a, lamb, c):
    return 1 / (1 + c * np.exp(-lamb * t))


def lorenz_func_2(t, c):
    return (0.01)**(np.exp(-c*t))


def simple_func(t, a):
    return 1 - np.exp(-a * t)


def cell_type_analysis(normalize=True, ca='1p25'):
    """
    Mtypes
    Etypes
    layers
    synapse_types
    in_degree
    in_ei_ratio
    """

    circuit = bluepy.Simulation(get_base_bcs()[0]).circuit
    mtypes = circuit.mvddb.mtype_id2name_map().values()
    cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
    etypes = circuit.mvddb.etype_id2name_map().values()
   # print cells
    layers = [0, 1, 2, 3, 4, 5]
    synapse_classes = ['INH', 'EXC']

    print "loading in degree"

    file_path_indegree = 'in_degree.npz'
    if not os.path.isfile(file_path_indegree):
        in_degree = connections(circ=circuit, base_target='mc2_Column').sum(axis=0)
        np.savez(open(file_path_indegree, 'w'), in_degree=in_degree)
    else:
        in_degree = np.load(file_path_indegree)['in_degree']
    #in_degree = np.random.permutation(31346)
    print "loaded"

    file_path_indegree_exc = 'in_degree_exc.npz'
    if not os.path.isfile(file_path_indegree_exc):
        in_degree_exc = connections(circ=circuit, base_target_pre='mc2_Excitatory', base_target_post='mc2_Column').sum(axis=0)
        np.savez(open(file_path_indegree_exc, 'w'), in_degree_exc=in_degree_exc)
    else:
        in_degree_exc = np.load(file_path_indegree_exc)['in_degree_exc']
    #in_degree = np.random.permutation(31346)
    print "loaded"

    file_path_indegree_inh = 'in_degree_inh.npz'
    if not os.path.isfile(file_path_indegree_inh):
        in_degree_inh = connections(circ=circuit, base_target_pre='mc2_Inhibitory', base_target_post='mc2_Column').sum(axis=0)
        np.savez(open(file_path_indegree_inh, 'w'), in_degree_inh=in_degree_inh)
    else:
        in_degree_inh = np.load(file_path_indegree_inh)['in_degree_inh']
    #in_degree = np.random.permutation(31346)
    print "loaded"

    indices = np.array(cells['etype'] == 'dNAC')
    print "Number of dNAC"
    print indices.sum()

    bins_n = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450])
    #bins_n = np.linspace(in_degree.min() + 20, in_degree.max() - 20, 9)
    percentiles_in_degree = np.digitize(in_degree, bins_n)

    index_correlation = 1

    n_plot = 15
    dt = 10.0 #1.0, 10.0, 20.0
    if ca=='1p25':
        corrs_shuffle, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, n=40, shuffle_level=1)
        corrs, bins = get_correlations(parameter_continue='', parameter_change='abcd', dt=dt, n=40, shuffle_level=0)
    else:
        corrs_shuffle, bins = get_correlations(parameter_continue='_ca_abcd', parameter_change='ca_abcd', dt=dt, n=20, shuffle_level=1, ca=ca)
        corrs, bins = get_correlations(parameter_continue='_ca_abcd', parameter_change='ca_abcd', dt=dt, n=20, shuffle_level=0, ca=ca)
    print corrs.shape
    start_mean = [-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean(axis=(-2, -1)))
    base_means = np.copy(start_mean)
    corrs = [-1, 1][index_correlation] * (corrs - corrs_shuffle)

    print "Number of messed up cells:"
    print (start_mean <= 0).sum()

    if normalize == True:
        corrs /=  start_mean[:, None, None, None]
        corrs_shuffle /=  start_mean[:, None, None, None]
        start_mean /= start_mean

    fig, ax = plt.subplots()

    ax.scatter(base_means, corrs.mean(axis=(-1))[:, 1, index_correlation], rasterized=True)

    plt.savefig('figures/scatter_base_vs_speed_ca%s.pdf' % ca, dpi=300)

    bins -= (2000.0 + dt/2)
    bins[0] = 0

    means = {}
    errs = {}

    means_indegree_complete = {}

    for mtype in mtypes:
        indices = np.array(cells['mtype'] == mtype)
        errs[mtype] = np.hstack([np.array([mean_confidence_interval([-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean(axis=(0, -1))))]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs[indices, ...].mean(axis=0))[:, index_correlation]])
        means[mtype] = np.hstack([np.array([start_mean[indices].mean()]), corrs[indices, ...].mean(axis=0).mean(axis=-1)[:, index_correlation]])
    for layer in layers:
        indices = np.array(cells['layer'] == layer)
        errs[layer] = np.hstack([np.array([mean_confidence_interval([-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean(axis=(0, -1))))]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs[indices, ...].mean(axis=0))[:, index_correlation]])
        means[layer] = np.hstack([np.array([start_mean[indices].mean()]), corrs[indices, ...].mean(axis=0).mean(axis=-1)[:, index_correlation]])
    for synapse_class in synapse_classes:
        indices = np.array(cells['synapse_class'] == synapse_class)
        errs[synapse_class] = np.hstack([np.array([mean_confidence_interval([-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean(axis=(0, -1))))]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs[indices, ...].mean(axis=0))[:, index_correlation]])
        means[synapse_class] = np.hstack([np.array([start_mean[indices].mean()]), corrs[indices, ...].mean(axis=0).mean(axis=-1)[:, index_correlation]])
    for etype in etypes:
        indices = np.array(cells['etype'] == etype)
        errs[etype] = np.hstack([np.array([mean_confidence_interval([-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean(axis=(0, -1))))]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs[indices, ...].mean(axis=0))[:, index_correlation]])
        means[etype] = np.hstack([np.array([start_mean[indices].mean()]), corrs[indices, ...].mean(axis=0).mean(axis=-1)[:, index_correlation]])

        if etype == 'dNAC':
            fig, ax = plt.subplots()
            means_neurons = corrs[indices, ...].mean(axis=-1)[:, :, index_correlation]
            print means_neurons.shape
            ax.plot(bins[1:101], means_neurons[0:50, :100].T, linewidth=0.2)
            ax.plot(bins[1:101], means_neurons[0:50, :100].T.mean(axis=1), linewidth=2.2, color='orange')
            ax.plot(bins[1:101], means[etype][1:101], linewidth=2.2, color='red')

            plt.savefig('figures/dNAC_ca%s.pdf' % ca)

    for percentile_id in range(bins_n.size + 1):
        indices = percentiles_in_degree == percentile_id
        errs["p%d" % percentile_id] = np.hstack([np.array([mean_confidence_interval([-1, 1][index_correlation] * ([0, 1][index_correlation] - corrs_shuffle[:, : , index_correlation, :].mean(axis=(0, -1))))]),
                                 np.apply_along_axis(mean_confidence_interval, -1, corrs[indices, ...].mean(axis=0))[:, index_correlation]])
        means["p%d" % percentile_id] = np.hstack([np.array([start_mean[indices].mean()]), corrs[indices, ...].mean(axis=0).mean(axis=-1)[:, index_correlation]])
        means_indegree_complete["p%d" % percentile_id] = corrs[indices, ...].mean(axis=0)[1, index_correlation, :]

    print "CORRS SHAPE"
    print corrs.shape
    all_means_neurons = corrs.mean(axis=-1)[:, 1, index_correlation]
    fig, axs = plt.subplots()
    # ax.scatter(in_degree_exc, in_degree_inh, c=all_means_neurons)

    lowers = [50, 150, 250, 350, 450, 550, 650]
    uppers = [150, 250, 350, 450, 550, 650, 750]

    for i, upper in enumerate(uppers):
        ax = axs
        indices = np.logical_and(in_degree <= upper, in_degree > lowers[i])
        ei_ratio = in_degree_exc[indices] / in_degree_inh[indices].astype(float)
        # ax.scatter(ei_ratio, all_means_neurons[indices], rasterized=True)
        lowers_2 = [5, 10, 15, 20]
        uppers_2 = [10, 15, 20, 25]
        values = []
        errors =[]
        for j, upper_2 in enumerate(uppers_2):
            indices_2 = np.logical_and(ei_ratio <= upper_2, ei_ratio > lowers_2[j])
            values.append(all_means_neurons[indices][indices_2].mean())
            errors.append(all_means_neurons[indices][indices_2].std(ddof=1)/np.sqrt(all_means_neurons[indices][indices_2].size))
        ax.errorbar(lowers_2, values, yerr=errors, marker='.', label=lowers[i])
    axs.legend()
    ax.set_ylabel('s_r_10-20ms')
    ax.set_xlabel('n_E/n_I')
    plt.tight_layout()
    plt.savefig('figures/in_degree_ratio_ei.pdf', dpi=300)


    fig, axs = plt.subplots(4, 2, figsize=(9, 9))
    # ax = axs[0, 0]
    # for mtype in mtypes:
    #     ax.errorbar(bins[:(n_plot+1)], means[mtype][:(n_plot+1)], yerr=errs[mtype][:(n_plot+1)], linewidth=0.8)

    means_indegree = np.array([means["p%d" % percentile_id][:(n_plot+1)][2] for percentile_id, value in enumerate(np.insert(bins_n, 0, 0))])
    errs_indegree = np.array([errs["p%d" % percentile_id][:(n_plot+1)][2] for percentile_id, value in enumerate(np.insert(bins_n, 0, 0))])

    for percentile_id, value in enumerate(np.insert(bins_n, 0, 0)):
        ax = axs[0, 0]
        ax.errorbar(bins[:(n_plot+1)], means["p%d" % percentile_id][:(n_plot+1)], yerr=errs["p%d" % percentile_id][:(n_plot+1)], linewidth=0.8,
                    label=value)
        ax = axs[2, 0]
    ax.errorbar(np.arange(len(np.insert(bins_n, 0, 0))), means_indegree, yerr=errs_indegree, color='black', marker='.')

    # individual sims
    print means_indegree_complete
    list_means_indegree_complete = []
    for percentile_id, value in enumerate(np.insert(bins_n, 0, 0)):
        list_means_indegree_complete.append(means_indegree_complete["p%d" % percentile_id])
    means_indegree_complete = np.vstack(list_means_indegree_complete)
    ax.plot(np.arange(len(np.insert(bins_n, 0, 0))), means_indegree_complete, color='red', alpha=0.3)
    print means_indegree_complete
    print means_indegree_complete.shape

    ax.set_xticks(np.arange(len(np.insert(bins_n, 0, 0))))
    ax.set_xticklabels(np.insert(bins_n, 0, 0) + 50)
    if index_correlation == 1:
        # ax.set_ylim([0.2, 0.55])
        # ax.set_ylim([0.0, 1.0])
        # ax.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
        ax.set_yticks([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 1])

    in_bins = np.arange(len(np.insert(bins_n, 0, 0)))
    dcorr = means_indegree
    dcorr_errs = errs_indegree

    for layer in layers:
        ax = axs[1, 0]
        ax.errorbar(bins[:(n_plot+1)], means[layer][:(n_plot+1)], yerr=errs[layer][:(n_plot+1)], linewidth=0.8, label=layer)
        ax = axs[3, 0]
        ax.bar(layer, means[layer][:(n_plot+1)][2], yerr=errs[layer][:(n_plot+1)][2], color='#cccccc', edgecolor='#636363', width=0.6)
    ax.set_xticks(layers)
    ax.set_xticklabels(['L%d' % l for l in range(1, 7)])

    for i, synapse_class in enumerate(synapse_classes):
        ax = axs[0, 1]
        ax.errorbar(bins[:(n_plot+1)], means[synapse_class][:(n_plot+1)], yerr=errs[synapse_class][:(n_plot+1)], linewidth=0.8, label=synapse_class)
        ax = axs[2, 1]
        ax.bar(i, means[synapse_class][:(n_plot+1)][2], yerr=errs[synapse_class][:(n_plot+1)][2],
               color='#cccccc', edgecolor='#636363', width=0.6)
    ax.set_xticks(np.arange(len(synapse_classes)))
    ax.set_xticklabels(synapse_classes)

    etype_means = np.array([means[etype][:(n_plot+1)][2] for etype in etypes])
    indices = np.argsort(etype_means)
    print indices
    for j, i in enumerate(indices):
        etype = etypes[i]
        ax = axs[1, 1]
        ax.errorbar(bins[:(n_plot+1)], means[etype][:(n_plot+1)], yerr=errs[etype][:(n_plot+1)], label=etype, linewidth=0.8)
        ax = axs[3, 1]
        ax.bar(j, means[etype][:(n_plot+1)][2], yerr=errs[etype][:(n_plot+1)][2],
               color='#cccccc', edgecolor='#636363', width=0.6)
    ax.set_xticks(np.arange(len(etypes)))
    ax.set_xticklabels(np.array(etypes)[indices])

    for ax in axs[:2, :].flatten():
        ax.set_xlabel('t (ms)')
    for ax in axs.flatten():
        ax.legend(loc='upper right', frameon=False, prop={'size':4})
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(['dRMSD', 'dCorrelation'][index_correlation])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('figures/cell_type_1_ca%s.pdf' % ca)


    return in_bins, dcorr, dcorr_errs


def in_degree_ca_divergence():
    cas = ['1p1', '1p2', '1p25', '1p3']
    colors = ['blue', 'green', 'black', 'red']
    results = []
    for i, ca in enumerate(cas):
        in_bins, dcorr, dcorr_errs = cell_type_analysis(normalize=True, ca=ca)
        results.append([in_bins, dcorr, dcorr_errs])
    fig, ax = plt.subplots()
    for i, ca in enumerate(cas):
        in_bins, dcorr, dcorr_errs = results[i]
        ax.errorbar((in_bins + 1) * 50, dcorr, yerr=dcorr_errs, color=colors[i],
                    marker='.')
    ax.set_xticks((in_bins + 1) * 50)
    ax.set_xticklabels((in_bins + 1) * 50)
    ax.set_ylim([0.2, 0.55])
    ax.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
    ax.set_xlabel('n_in')
    ax.set_ylabel('sim_r')
    plt.savefig('figures/in_degree_divergence.pdf')


def plot_correlations_figure_1_detailed_for_paper_no_bars(normalize=False, stut_ir=False):
    """
    Comparison plot of sources of variability

    :param normalize:
    :return:
    """
    dt = 10.0

    if not stut_ir:
        corrs_all = {}
        p_cont = ['' for x in parameters[0]] + ['_gv2' for x in parameters[1]]
        for p, pc in zip(parameters[0] + parameters[1], p_cont):
            print "loading %s" % p
            corrs, bins = get_correlations(parameter_continue=pc, parameter_change=p, dt=dt)
            # deleting a neuron in faulty simulation report
            if p == 'gv2b':
                corrs = np.delete(corrs, 573, axis=0)

            corrs_all[p] = corrs

        decouple_params = ['abcd', 'ab', 'a', 'c', 'd', 'f', 'g']
        for p in decouple_params:
            print "loading %s" % p
            corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt, decouple=True)
            corrs_all[p + '_dec'] = corrs


    if stut_ir:
        circuit = bluepy.Simulation(get_base_bcs()[0]).circuit
        cells = circuit.v2.cells({Cell.HYPERCOLUMN: 2})
        etypes = circuit.mvddb.etype_id2name_map().values()


        etypes_c = [etypes[x] for x in [2, 4, 8, 9, 10]]
        print etypes_c
        indices = np.array([etype in etypes_c for etype in cells['etype']])
        print indices.sum()
        print cells['mtype'][indices]
        print cells['etype'][indices]


        corrs_all = {}
        p_cont = ['' for x in parameters[0]] + ['_gv2' for x in parameters[1]]
        for p, pc in zip(parameters[0] + parameters[1], p_cont):
            print "loading %s" % p
            corrs, bins = get_correlations(parameter_continue=pc, parameter_change=p, dt=dt)
            # deleting a neuron in faulty simulation report

            corrs_all[p] = corrs[indices]

        decouple_params = ['abcd', 'ab', 'a', 'c', 'd', 'f', 'g']
        for p in decouple_params:
            print "loading %s" % p
            corrs, bins = get_correlations(parameter_continue='', parameter_change=p, dt=dt, decouple=True)
            corrs_all[p + '_dec'] = corrs[indices]


    plot_contents = [['abcd', 'bcd', 'abc', 'abd', 'ab', 'a', 'b', 'c', 'd', 'g', 'f'],
                     [p + '_dec' for p in decouple_params]]
    fig, axs = plt.subplots(4, 2, figsize=(10, 6))

    for j, contents in enumerate(plot_contents):
        for index_correlation in range(2):
            ax = axs[index_correlation, j]
            ax.set_ylabel(['RMSD (mV)', 'r'][index_correlation])
            ax.set_ylim([[0, 6.5], [0, 1.1]][index_correlation])
            ax.set_yticks([[0, 1, 2, 3, 4, 5, 6], [0, 0.5, 1]][index_correlation])
            for i, p in enumerate(contents):
                corrs = corrs_all[p]
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                ax.scatter(i + np.zeros(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0).shape),
                           corrs.mean(axis=0)[100:, index_correlation].mean(axis=0),
                           marker='.', color='red')
                ax.bar([i], [scale], yerr=[scale_err], color='#a6bddb')
                ax.set_xticks(np.arange(len(contents)))
                ax.set_xticklabels(contents)
    for j, contents in enumerate(plot_contents):
        for index_correlation in range(2):
            ax = axs[index_correlation + 2, j]
            ax.set_ylabel(['dRMSD', 'dr'][index_correlation])
            ax.set_ylim([0, 1.1])
            for i, p in enumerate(contents):
                corrs = corrs_all[p]
                scale = corrs.mean(axis=0).mean(axis=-1)[100:, index_correlation].mean()
                scale_err = mean_confidence_interval(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0))
                step = corrs.mean(axis=0).mean(axis=-1)[1, index_correlation]
                step_err = mean_confidence_interval(corrs.mean(axis=0)[1, index_correlation])

                step_2, step_err = get_similarity(scale, scale_err, step, step_err, index_correlation=index_correlation)

                ax.bar([i], [step_2], yerr=[step_err], color='#a6bddb')

                step_2_dots = get_similarity_no_error(corrs.mean(axis=0)[100:, index_correlation].mean(axis=0),
                                                      corrs.mean(axis=0)[1, index_correlation],
                                                      index_correlation=index_correlation)
                ax.scatter(i + np.zeros(step_2_dots.shape), step_2_dots, marker='.', color='red', alpha=0.5)

            ax.set_xticks(np.arange(len(contents)))
            ax.set_xticklabels(contents)

    for ax in axs.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
       # ax.set_ylim([0, 1])
        ax.legend(loc='upper right', prop={'size':4})
        ax.set_ylabel('r')

    plt.tight_layout()
    if not stut_ir:
         plt.savefig('figures/deciphering_sources_3b.pdf')
    if stut_ir:
        plt.savefig('figures/deciphering_sources_3b_stut_ir.pdf')

if __name__ == "__main__":
    corrs, bins = get_correlations()
    plot_correlations()
    plot_correlations_g()
    plot_correlations_figure_1()
    plot_soma_voltage_example_figure()
    plot_spike_raster_example_figure_ca_scan()
    plot_correlations_figure_decoupled_2()
    plot_cas_figure_2()
    plot_evolving_corrs()
    plot_correlations_figure_1()
    plot_correlations_figure_1_saturation()
    plot_correlations_figure_1_divergence()
    plot_correlations_between_rmsd_and_corr()
    cell_type_analysis()
    convergence_all_sources()
    plot_O1()
    plot_O1_difference()
    plot_sample_increase()
    plot_time_bins_fitting()
    convergence_comparison()
    plot_cells_divergence_example()
    plot_stims()
    plot_mvr()
    plot_soma_voltage_example_figure_dNAC()
    convergence_all_sources_main_figure()
    convergence_comparison()
    plot_decoupled_analysis()
    plot_decoupled_validation()
    plot_correlations_figure_1_detailed_for_paper()
    plot_stims(normalize=True)
    plot_correlations_stut_ir()
    plot_correlations_figure_1_detailed_for_paper_no_bars(stut_ir=False)
    plot_correlations_figure_1_detailed_for_paper_no_bars(stut_ir=True)
    cell_type_analysis(normalize=True, ca='1p1')
    cell_type_analysis(normalize=True, ca='1p2')
    cell_type_analysis(normalize=True, ca='1p25')
    cell_type_analysis(normalize=True, ca='1p3')
    in_degree_ca_divergence()