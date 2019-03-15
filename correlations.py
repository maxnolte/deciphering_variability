import bluepy
import numpy as np
from scipy.signal import medfilt


def get_soma_time_series(blueconfig, t_start=None, t_end=None, gids=None):
    soma = bluepy.Simulation(blueconfig).v2.reports['soma']
    data = soma.data(t_start=t_start, t_end=t_end, gids=None) #, gids=np.arange(72788, 72800)) #
    return data, data.axes[1]/1000.0


def median_filter(vm, kernel_size=41):
    return np.apply_along_axis(medfilt, 1, vm, kernel_size=kernel_size)


def voltage_correlation_from_data(vm_1, vm_2, times, dt, shuffle=0):
    return fast_coevolution(vm_1, vm_2, times, dt, vcorrcoef, shuffle=shuffle)


def voltage_rmsd_from_data(vm_1, vm_2, times, dt, shuffle=0):
    return fast_coevolution(vm_1, vm_2, times, dt, vrmsd, shuffle=shuffle)


def fast_coevolution(vm_1, vm_2, times, dt, correlation_function, shuffle=0):
    """
    :param vm_1:
    :param vm_2:
    :param times:
    :param dt:
    :return:
    """
    n_gids = vm_1.shape[0]
    n_bins = int(np.floor((times[-1] - times[0] + times[1] - times[0])/dt))
    bins = np.arange(0, n_bins + 1) * dt + times[0]
    n_frames_per_dt = int(times[times < bins[-1]].size/n_bins)

    vm_1 = vm_1[:, :n_bins * n_frames_per_dt]
    vm_2 = vm_2[:, :n_bins * n_frames_per_dt]

    if shuffle == 2:
        vm_1 = vm_1[:, np.random.permutation(n_bins * n_frames_per_dt)]
        vm_2 = vm_2[:, np.random.permutation(n_bins * n_frames_per_dt)]

    # Generate shape (gids x bins x frames_in_bin)
    vm_1 = np.reshape(vm_1, (n_gids, n_bins, n_frames_per_dt))
    vm_2 = np.reshape(vm_2, (n_gids, n_bins, n_frames_per_dt))

    if shuffle == 1:
        vm_1 = vm_1[:, np.random.permutation(vm_1.shape[1]), :]
        vm_2 = vm_2[:, np.random.permutation(vm_2.shape[1]), :]

    vm_1 = np.reshape(vm_1, (n_gids * n_bins, n_frames_per_dt))
    vm_2 = np.reshape(vm_2, (n_gids * n_bins, n_frames_per_dt))

    corr = correlation_function(vm_1, vm_2)

    corr = np.reshape(corr, (n_gids, -1))
    return corr, bins


def vcorrcoef(x, y):
    """
    Compute Pearson's r correlation on 2d array
    """
    xm = np.reshape(np.mean(x, axis=1),(x.shape[0], 1))
    ym = np.reshape(np.mean(y, axis=1),(y.shape[0], 1))
    r_num = np.sum((x - xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((x - xm)**2, axis=1) * np.sum((y - ym)**2, axis=1))
    r = r_num/r_den
    return r


def vrmsd(x, y):
    """
    Compute root mean-square deviation on 2d array
    """
    return np.sqrt(np.sum((x - y)**2, axis=1) / x.shape[1])


def voltage_correlation_from_blueconfig(sim_1, sim_2, t_start=0, t_end=1500, t_window=100, dt_window=10):
    """
    Sliding time window correlation for early preliminary analysis
    """
    gids = bluepy.Simulation(sim_1).reports.soma.gids
    if t_start == 0:
        t_start = 0.00000000001
    v_1 = bluepy.Simulation(sim_1).reports.soma.get_timeslice(t_start, t_end)
    v_2 = bluepy.Simulation(sim_2).reports.soma.get_timeslice(t_start, t_end)

    duration = t_end - t_start
    v_dt = duration/float(v_1.shape[0])

    n_bins = np.floor((duration-t_window+dt_window)/dt_window).astype(int)
    correlations = np.zeros((gids.size, n_bins))
    print correlations.shape

    for i in range(n_bins):
        print i
        index_start = np.floor((i * dt_window)/v_dt).astype(int)
        index_end = np.floor((i * dt_window + t_window)/v_dt).astype(int)
        for j in range(gids.size):
            correlations[j, i] = np.corrcoef(v_1[index_start:index_end, j], v_2[index_start:index_end, j])[0, 1]
    return correlations
