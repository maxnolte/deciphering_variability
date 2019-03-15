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


def spike_count_difference():
    n_include = 20

    bcs_0 = iaf.get_continue_bcs(params='')[:n_include]
    bcs_1 = iaf.get_change_bcs(params='y')[:n_include]
    bcs_2 = iaf.get_change_bcs(params='e')[:n_include]
    bcs_3 = iaf.get_change_bcs(params='abcd')[:n_include]
    bcs_4 = iaf.get_change_bcs(params='abcdy')[:n_include]
    bcs_list = [bcs_1, bcs_2, bcs_3, bcs_4]
    bins=np.linspace(0, 50, 11)

    differences = np.zeros((4, len(bcs_0), bins.size - 1))
    for i, bcs in enumerate(bcs_list):
        differences[i, :, :] = get_differences(bcs_0, bcs, bins=bins)

    means = differences.mean(axis=1)
    errs = np.apply_along_axis(iaf.mean_confidence_interval, 1, differences)

    fig, axs = plt.subplots(2)
    ax = axs[0]
    xs = bins[:-1] + (bins[1] - bins[0])/2.0
    ax.plot(xs, np.zeros(xs.size), color='red')
    ax.errorbar(xs, means[0], errs[0])
    ax.errorbar(xs + 0.2, means[1], errs[1])

    ax = axs[1]
    xs = bins[:-1] + (bins[1] - bins[0])/2.0
    ax.plot(xs, np.zeros(xs.size), color='red')
    ax.errorbar(xs, means[2], errs[2])
    ax.errorbar(xs + 0.2, means[3], errs[3])

    plt.savefig('figures/perturbation_fr.pdf')


def get_differences(bcs_1, bcs_2, gids=None, bins=np.linspace(0, 100, 21)):

    differences = np.zeros((len(bcs_1), bins.size -1))

    for sign, bcs in zip([-1, 1], [bcs_1, bcs_2]):
        for i, bc in enumerate(bcs):
            print bc
            sim = bluepy.Simulation(bc)
            spikes = sim.v2.reports['spikes'].data(gids=gids)
            spikes = np.array(spikes)
            counts, _ = np.histogram(spikes, bins=bins)
            differences[i, :] += sign * counts
    return differences


if __name__ == "__main__":
   spike_count_difference()