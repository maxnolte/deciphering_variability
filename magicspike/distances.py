import numpy as np
import itertools
import pyspike
import NeuroTools.signals
import warnings

import multiprocessing as mp

"""
    Magic spike train analysis
"""


def spike_train_distance(spike_train_a, spike_train_b, interval, method='spike', time_sorted=True,
                         trimmed=True, **keywords):
    """
    Meta-method to use one of several spike train distances.
    """
    q = 0.05
    sigma = 0.5
    dt = 0.01
    tau = 0.5
    keys = sorted(keywords.keys())
    for kw in keys:
        if kw == 'cost_q':
            q = keywords[kw]
        elif kw == 'sigma_conv':
            sigma = keywords[kw]
        elif kw == 'dt_conv':
            dt = keywords[kw]
        elif kw == 'tau':
            tau = keywords[kw]

    if not trimmed:
        spike_train_a = spike_train_a[np.logical_and(spike_train_a >= interval[0],
                                                     spike_train_a <= interval[1])]
        spike_train_b = spike_train_b[np.logical_and(spike_train_b >= interval[0],
                                                     spike_train_b <= interval[1])]

    if not time_sorted:
        spike_train_a = np.sort(spike_train_a)
        spike_train_b = np.sort(spike_train_b)

    # if spike_train_a.size == 0 or spike_train_b.size == 0:
    #     warnings.warn('Empty spike train used for distance computation!')

    if method == 'spike':
        return distance_spike(spike_train_a, spike_train_b, interval)
    elif method == 'isi':
        return distance_isi(spike_train_a, spike_train_b, interval)
    elif method == 'vpd':
        return distance_vpd(spike_train_a, spike_train_b, q=q)
    elif method == 'schreiber':
        return distance_schreiber(spike_train_a, spike_train_b, interval, sigma=sigma, dt=dt)
    elif method == 'vanrossum':
        return distance_vanrossum(spike_train_a, spike_train_b, interval, tau=tau, dt=dt)


def distance_spike(spike_train_a, spike_train_b, interval):
    """
    SPIKE-distance (Kreutz) using pyspike
    """
    spike_train_1 = pyspike.SpikeTrain(spike_train_a, interval)
    spike_train_2 = pyspike.SpikeTrain(spike_train_b, interval)
    return pyspike.spike_distance(spike_train_1, spike_train_2, interval)


def distance_isi(spike_train_a, spike_train_b, interval):
    """
    ISI-distance (Kreutz) using pyspike
    """
    spike_train_1 = pyspike.SpikeTrain(spike_train_a, interval)
    spike_train_2 = pyspike.SpikeTrain(spike_train_b, interval)
    return pyspike.isi_distance(spike_train_1, spike_train_2, interval)


def distance_vpd(spike_train_a, spike_train_b, q):
    """
    Victor-Purpura distance using NeuroTools
    """
    spike_train_1 = NeuroTools.signals.SpikeTrain(spike_train_a)
    spike_train_2 = NeuroTools.signals.SpikeTrain(spike_train_b)
    return spike_train_1.distance_victorpurpura(spike_train_2, cost=q)


def distance_schreiber(spike_train_a, spike_train_b, interval, sigma=0.5, dt=0.1):
    """
    Reliability measure used in Wang et al., 2013 (Science)
    """
    s_1 = spike_train_convolution(spike_train_a, interval, dt, sigma)
    s_2 = spike_train_convolution(spike_train_b, interval, dt, sigma)
    s_1_norm = np.linalg.norm(s_1)
    s_2_norm = np.linalg.norm(s_2)
    if s_1_norm > 0 and  s_2_norm > 0:
        return np.dot(s_1, s_2)/(s_1_norm*s_2_norm)
    else:
        return 0.0


def distance_vanrossum(spike_train_a, spike_train_b, interval, tau=0.5, dt=0.1):
    """
    Dummy
    """

    s_1 = spike_train_convolution_rossum(spike_train_a, interval, dt, tau)
    s_2 = spike_train_convolution_rossum(spike_train_b, interval, dt, tau)
    return dt/tau * np.dot(s_1-s_2, s_1-s_2)


def spike_train_convolution(spike_times, interval, dt, sigma):
    """
    Needed for Schreiber reliability measure
    """
    N = int(np.floor((interval[1]-interval[0])/dt)+1)
    x = np.linspace(interval[0], interval[1], N)
    s = np.zeros(N)
    for spike in spike_times:
        s = s + gaussian(x, spike, sigma)
    return s


def spike_train_convolution_rossum(spike_times, interval, dt, tau):
    """
    Needed for Schreiber reliability measure
    """
    N = int(np.floor((interval[1]-interval[0])/dt)+1)
    x = np.linspace(interval[0], interval[1], N)
    s = np.zeros(N)
    for spike in spike_times:
        s = s + rossum_exponential(x, spike, tau)
    return s


def not_lazy_gaussian(x, mu, sigma):
    """
    Used for convoluting
    """
    return np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

def gaussian(x, mu, sigma):
    """
    Used for convoluting
    """
    y = np.zeros(x.shape)
    indices = np.logical_and(x >= (mu - 5 * sigma), x <= (mu + 5 * sigma))
    x_near = x[indices] - mu
    y[indices] += np.exp(-x_near*x_near/(2*sigma*sigma))
    return y

def rossum_exponential(x, mu, tau):
    """
    Used for convoluting for van rossum
    """

    y = np.exp(-(x-mu)/tau)
    y[x < mu] = 0  # inefficient
    return y

def compute_pairwise_distances(spike_times, trial_ids, interval, n_trials, n_samples, method='spike', time_sorted=True,
                                            trimmed=True, combi_seed=0, **keywords):
    """
    Compute distances between different trials
    spike_times and trial_ids have to be numpy arrays with the same length.
    """
    combinations = np.array(list(itertools.combinations(range(n_trials), 2)))
    np.random.seed(combi_seed)
    np.random.shuffle(combinations)
    n_pairs = len(combinations)
    if n_samples > n_pairs:
        n_samples = n_pairs
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        distances[i] = spike_train_distance(spike_times[trial_ids == combinations[i, 0]],
                                            spike_times[trial_ids == combinations[i, 1]],
                                            interval, method=method, time_sorted=time_sorted,
                                            trimmed=trimmed, **keywords)
    return distances.mean(), distances.std(ddof=1), distances


def spike_train_distance_mp_wrapper(kwargs):
        return spike_train_distance(**kwargs)


def compute_pairwise_distances_mp(spike_times, trial_ids, interval, n_trials, n_samples, method='spike', time_sorted=True,
                                            trimmed=True, combi_seed=0, **keywords):
    """
    Compute distances between different trials
    spike_times and trial_ids have to be numpy arrays with the same length.
    """
    combinations = np.array(list(itertools.combinations(range(n_trials), 2)))
    np.random.seed(combi_seed)
    np.random.shuffle(combinations)
    n_pairs = len(combinations)
    if n_samples > n_pairs:
        n_samples = n_pairs
    distances = np.zeros(n_samples)
    work_items = []
    for i in range(n_samples):
        d = dict(spike_train_a=spike_times[trial_ids == combinations[i, 0]],
                 spike_train_b=spike_times[trial_ids == combinations[i, 1]],
                 interval=interval, method=method, time_sorted=time_sorted,
                 trimmed=trimmed, **keywords)

        work_items.append(d)
    pool = mp.Pool(processes=mp.cpu_count())
    distances = np.array(pool.map(spike_train_distance_mp_wrapper, work_items))
    return distances.mean(), distances.std(ddof=1), distances


def compute_pairwise_distances_exc_inh_naive(spike_times, trial_ids, interval, n_trials, n_samples,
                                             exc, method='schreiber', time_sorted=True,
                                            trimmed=True, combi_seed=0, **keywords):
    """
    Compute distances between different spike trains of neurons
    """
    combinations = np.array(list(itertools.combinations(range(n_trials), 2)))
    exc_pairs = np.array(list(itertools.combinations(exc, 2)))
    np.random.seed(combi_seed)
    np.random.shuffle(combinations)
    np.random.shuffle(exc_pairs)
    n_pairs = len(combinations)
    exc_or = np.logical_xor(exc_pairs[:, 0], exc_pairs[:, 1])
    if n_samples > n_pairs:
        n_samples = n_pairs
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        sign = 1
        if exc_or[i] > 0:
            sign = -1
        distances[i] = spike_train_distance(spike_times[trial_ids == combinations[i, 0]],
                                            spike_times[trial_ids == combinations[i, 1]],
                                            interval, method=method, time_sorted=time_sorted,
                                            trimmed=trimmed, **keywords) * sign
    return distances.mean(), distances.std(ddof=1), distances
