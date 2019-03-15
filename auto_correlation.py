from initial_analysis_final import *
import correlations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

def get_auto_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, shuffle_level=0, kernel=0, ca=None,
                     decouple=False, n=20, variance=None, nrrp=1, nondecouple=0):

    folder = '/gpfs/bbp.cscs.ch/project/proj9/nolte/variability/saved_soma_correlations' + '/corrs_auto_' + parameter_continue + '_' + parameter_change
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
        corrs, bins = compute_soma_auto_correlations(parameter_continue=parameter_continue,
                                                parameter_change=parameter_change,
                                                shuffle_level=shuffle_level, kernel=kernel, dt=dt, ca=ca,
                                                decouple=decouple, n=n, variance=variance, nrrp=nrrp,
                                                nondecouple=nondecouple)
        np.savez(open(file, 'w'), corrs=corrs, bins=bins)
    data = np.load(file)
    return data['corrs'], data['bins']


def compute_soma_auto_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, t_end=3000.0, shuffle_level=0,
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
        vm_continue, times = get_soma_time_series(continue_bc, t_end=t_end, t_start=t_start)
        # vm_change, times = get_soma_time_series(change_bc, t_end=t_end, t_start=t_start)
        print "vm shape"
        print vm_continue.shape
        vm_continue = np.array(vm_continue)
        n_roll = int(dt*10)
        print n_roll
        results_sim = []
        for k in range(50):
            vm_change = np.roll(vm_continue, n_roll * k, axis=1)

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
            results_sim.append(corrs)
        results.append(np.concatenate([a[..., np.newaxis] for a in results_sim], axis=3).mean(axis=1))
    results = np.concatenate([a[..., np.newaxis] for a in results], axis=3)
    return results, bins


def plot_auto_corrs():
    corrs, bins = get_auto_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, shuffle_level=0, n=20)
    corrs_div, bins_div = get_correlations(parameter_continue='', parameter_change='abcd', dt=5.0, shuffle_level=0, n=40)
    print corrs_div.shape
    corrs = corrs.astype(float)
    fig, axs = plt.subplots(2)

    times = np.arange(50) * 5.0

    for i in range(2):
        ax = axs[i]
        ax.plot(times, corrs[:, i, :, :].mean(axis=(0, 2)), label='Auto-correlation')
        ax.plot(np.insert(times + 2.5, 0, 0), np.insert(corrs_div[:, :50, i, :].mean(axis=(0, 2)), 0, i),
                label='Divergence', linestyle='--')
    axs[0].set_ylabel('RMSD (mV)')
    axs[1].set_ylabel('r')
    axs[1].set_xlabel('t (ms)')
    axs[0].legend(loc='lower right')


    plt.savefig('figures_auto/auto_corr.pdf')
    return corrs

corrs = plot_auto_corrs()
