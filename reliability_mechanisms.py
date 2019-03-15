import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bluepy
import numpy as np
import h5py
from scipy.signal import correlate
import scipy.stats as stats
import simplices_graph_theory # Results from https://doi.org/10.3389/fncom.2017.00048
import metrics_factory # Results from https://doi.org/10.3389/fncom.2017.00048
import analysis_exp_25_decoupled
import analysis_poisson


def get_reliabilities_firing_rates_metrics(exc_only=True, mtype=None, inh_only=False):
    """
    Returning reliabilities and firing rates for selected L456 neurons only.
    :param exc_only:
    :param layer:
    :return:
    """
    df_0 = analysis_exp_25_decoupled.get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1, spont=False,
                                           network_removal=False, vpm_ca2p0=False).mean(axis=1)

    df_neurons = analysis_exp_25_decoupled.get_selected_L456_gids()
    df_neurons = df_neurons.loc[np.unique(df_0.index)]
    print df_neurons.shape

    if inh_only:
        df_0 = df_0.loc[df_neurons['synapse_class'] == 'INH']
    elif exc_only:
        df_0 = df_0.loc[df_neurons['synapse_class'] == 'EXC']
        if mtype is not None:
            # To do later if needed
            df_0 = df_0.loc[df_neurons['mtype'] == mtype]

    spike_counts = analysis_poisson.get_spike_counts_evoked(t_start=2000, t_end=7000)
    frs = spike_counts['mean'].loc[df_0.index]/5.0

    # Load corresponding in degree and simplices
    gids_rel = np.array(df_0.index) - 2 * 31346 - 1
    print gids_rel
    nan_indices = np.full(31346, True, dtype=bool)
    nan_indices[gids_rel] = False
    print nan_indices
    simplex_counts_sink, simplex_counts_all = metrics_factory.load_simplices_participation()
    simplex_counts_sink = simplex_counts_sink.T
    simplex_counts_all = simplex_counts_all.T

    print simplex_counts_sink.shape

    metrics = np.vstack([simplex_counts_all[1, :], simplex_counts_all[1, :] - simplex_counts_sink[1], simplex_counts_sink[1],
                         simplex_counts_sink[2:-1]])
    metrics_no_nan = metrics[:, nan_indices == False]
    metric_labels = ['degree', 'out-degree', 'in-degree (1D)',
              '2D',  '3D',  '4D',  '5D']

    return df_0.values, frs, metrics_no_nan, metric_labels


def plot_influence():
    """
    Plotting metric percentiles vs reliability
    :return:
    """
    rs_no_nan, frs, metrics_no_nan, labels = get_reliabilities_firing_rates_metrics()
    #rs_no_nan = frs.values
    #np.random.shuffle(rs_no_nan)
    colors=['dimgray', 'lightgrey', 'black',
              'seagreen',  'mediumseagreen',  'mediumaquamarine',  'mediumturquoise']
    markers = ['s', '<', '>',
               '^', 's','d','*']

    fig, axs = plt.subplots()
    ax = axs
    n_bins = 10
    percentiles = np.linspace(10, 100, n_bins)
    mean_rs = np.zeros((n_bins, metrics_no_nan.shape[0]))
    mean_errs = np.zeros((n_bins, metrics_no_nan.shape[0]))
    mean_percentile_metrics = np.zeros((n_bins, metrics_no_nan.shape[0]))
    mean_percentile_metrics_errs = np.zeros((n_bins, metrics_no_nan.shape[0]))
    bin_maxs = np.zeros((n_bins, metrics_no_nan.shape[0]))
    for i in range(metrics_no_nan.shape[0]):
        values = metrics_no_nan[i, :]
        percentiles_values = np.percentile(values, percentiles)
        bin_maxs[:, i] = percentiles_values
        digits = np.digitize(values, percentiles_values)
        for j in range(n_bins):
            mean_rs[j, i] = rs_no_nan[digits == j].mean()
            mean_errs[j, i] = simplices_graph_theory.mean_confidence_interval(rs_no_nan[digits == j], confidence=0.95)

            mean_percentile_metrics[j, i] = metrics_no_nan[i, digits == j].mean()
            mean_percentile_metrics_errs[j, i] = simplices_graph_theory.mean_confidence_interval(metrics_no_nan[i, digits == j], confidence=0.95)

    values = metrics_no_nan[2, :]
    percentiles_values = np.percentile(values, percentiles)
    digits_1d = np.digitize(values, percentiles_values)


    for i in range(metrics_no_nan.shape[0]):
        if i > 1 and i < 6:
            ax.errorbar(percentiles[:], mean_rs[:, i], yerr=mean_errs[:, i], fmt='-%s' % markers[i], color=colors[i],
                        label=labels[i], markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.5)
    ax.set_xticks(percentiles)
    ax.legend(loc='lower right')
    ax.set_xticklabels(percentiles.astype(int))
    ax.set_xlabel('Metric percentile')
    ax.set_ylabel('Mean reliability')


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('simplices_figures/influence_mean.pdf')

    cmap = matplotlib.cm.inferno
    cmap.set_bad('grey', 1.)

    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    for i, ax in enumerate(axs.flatten()[:7]):
        ax.scatter(metrics_no_nan[i, :], rs_no_nan, rasterized=True, marker='.', alpha=0.4, color='gray')
        ax.errorbar(mean_percentile_metrics[:, i], mean_rs[:, i], yerr=mean_errs[:, i],
                    xerr=mean_percentile_metrics_errs[:, i], lw=1.8, color='black')
        ax.vlines(bin_maxs[:-1, i], 0, 1, color='blue')
        ax.set_xlabel(labels[i])
        ax.set_ylabel('Reliability')
        ax.set_ylim([0, 1])
    plt.savefig('simplices_figures/influence_scatter.pdf')


    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    which_bins = [[0, 3],
                  [4, 7],
                  [7, 10],]
    index_dim = 4
    for j in range(3):
        index_start = which_bins[j][0]
        index_end = which_bins[j][1]
        indices = np.logical_and(digits_1d >= index_start, digits_1d < index_end)
        ax = axs[j, 0]
        ax.scatter(metrics_no_nan[2, indices], rs_no_nan[indices], rasterized=True, marker='.', alpha=0.4, color='gray')
        ax.errorbar(mean_percentile_metrics[index_start:index_end, 2], mean_rs[index_start:index_end, 2], yerr=mean_errs[index_start:index_end, 2],
                        xerr=mean_percentile_metrics_errs[index_start:index_end, 2], lw=1.8, color='black')
        ax.set_xlabel(labels[2])
        ax.set_ylabel('Reliability')
        ax.set_ylim([0, 1])

        ax = axs[j, 2]
        ax.errorbar(np.arange(index_end-index_start), mean_rs[index_start:index_end, 2], yerr=mean_errs[index_start:index_end, 2],
                        lw=1.8, color='black')
        ax = axs[j, 1]
        ax.scatter(metrics_no_nan[index_dim, indices], rs_no_nan[indices], rasterized=True, marker='.', alpha=0.4, color='gray')

        n_bins = 3
        new_values = metrics_no_nan[index_dim, indices]
        percentiles_2 = np.linspace(100/3.0, 100, n_bins)
        percentiles_values_2 = np.percentile(new_values, percentiles_2)
        digits_2 = np.digitize(new_values, percentiles_values_2)
        means_new = np.zeros(n_bins)
        errs_new = np.zeros(n_bins)
        means_new_r = np.zeros(n_bins)
        errs_new_r = np.zeros(n_bins)
        for k in range(n_bins):
            means_new[k] = new_values[digits_2 == k].mean()
            errs_new[k] = simplices_graph_theory.mean_confidence_interval(new_values[digits_2 == k], confidence=0.95)
            means_new_r[k] = rs_no_nan[indices][digits_2 == k].mean()
            errs_new_r[k] = simplices_graph_theory.mean_confidence_interval(rs_no_nan[indices][digits_2 == k], confidence=0.95)
        ax.errorbar(means_new, means_new_r, yerr=errs_new_r,
                        xerr=errs_new, lw=1.8, color='black')
        ax.set_xlabel(labels[4])
        ax.set_ylabel('Reliability')
        ax.set_ylim([0, 1])
        ax = axs[j, 2]
        ax.errorbar(np.arange(index_end-index_start), means_new_r, yerr=errs_new_r,
                        lw=1.8, color='red')
    plt.savefig('simplices_figures/influence_scatter_zoom.pdf')

    fig, ax = plt.subplots()
    x = metrics_no_nan[2, :]
    y = metrics_no_nan[4, :]
    #denominator, xedges, yedges = np.histogram2d(x, y, bins=(np.linspace(0, 400, 21), np.linspace(0, 1200, 21),))
    denominator, xedges, yedges = np.histogram2d(x, y, bins=(np.linspace(0, 800, 26), np.linspace(0, 20000, 26),))

    nominator, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=rs_no_nan)
    mean_2d_rs = (nominator / denominator).T[::-1, :]
    print mean_2d_rs

    masked_array = np.ma.array(mean_2d_rs, mask=np.isnan(mean_2d_rs))

    imgplot = ax.imshow(masked_array, interpolation='nearest', cmap=cmap, vmin=0, vmax=0.8)
    fig.colorbar(imgplot)
    ax.set_xticks(np.arange(xedges.size) - 0.5)
    ax.set_xticklabels(xedges[:])
    ax.set_yticks(np.arange(yedges.size) - 0.5)
    ax.set_yticklabels(yedges[::-1])
    ax.set_xlabel('1D')
    ax.set_ylabel('3D')

    #ax.set_ylim([0, 3000])
    plt.savefig('simplices_figures/influence_scatter_3.pdf')


def plot_in_degree_simplices():
    rs_no_nan, frs, metrics_no_nan, labels = get_reliabilities_firing_rates_metrics()

    colors = ['dimgray', 'lightgrey', 'darkgray',
              'seagreen', 'mediumseagreen', 'mediumaquamarine', 'mediumturquoise']
    markers = ['s', '<','>',
               '^', 's', 'd', '*']

    n_bins = 5
    bin_boundaries = np.linspace(0.2, 1.0, n_bins)
    print bin_boundaries
    digits = np.digitize(rs_no_nan, bin_boundaries)  # Will give n_bins digits
    mean_rs = np.zeros(n_bins)

    mean_metrics = np.zeros((n_bins, metrics_no_nan.shape[0]))

    for i in range(n_bins):
        print i
        print np.sum(digits == i)
        mean_rs[i] = rs_no_nan[digits == i].mean()
        mean_metrics[i, :] = metrics_no_nan[:, digits == i].mean(axis=1)/metrics_no_nan[:, :].mean(axis=1)

    fig, axs = plt.subplots()
    ax = axs

    for i in range(mean_metrics.shape[1]):
        ax.errorbar(mean_rs, mean_metrics[:, i], yerr=0,
                    fmt='-%s' % markers[i], color=colors[i],
                    label=labels[i], markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.5)
    ax.plot([0, 1], [1, 1], '--', color='black')
    ax.legend(loc='lower right')
    ax.set_xlabel('r')
    ax.set_ylabel('overexpression to mean')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig('simplices_figures/simplices_comparison.pdf')


def in_degree_mtype():
    df_0 = analysis_exp_25_decoupled.get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1, spont=False,
                                           network_removal=False, vpm_ca2p0=False).mean(axis=1)
    df_neurons = analysis_exp_25_decoupled.get_selected_L456_gids()
    df_neurons = df_neurons.loc[np.unique(df_0.index)]
    print df_neurons
    mtypes = np.unique(df_neurons[df_neurons['synapse_class'] == 'EXC']['mtype'])

    means = np.zeros((len(mtypes), 2))
    errs = np.zeros((len(mtypes), 2))

    for i, mtype in enumerate(mtypes):
        rs_no_nan, frs, metrics_no_nan, labels = get_reliabilities_firing_rates_metrics(mtype=mtype)
        means[i, 0] = rs_no_nan.mean()
        means[i, 1] = metrics_no_nan[2, :].mean()
        errs[i, 0] = simplices_graph_theory.mean_confidence_interval(rs_no_nan)
        errs[i, 1] = simplices_graph_theory.mean_confidence_interval(metrics_no_nan[2, :])

    fig, ax = plt.subplots()
    ax.errorbar(means[:, 1], means[:, 0], xerr=errs[:, 1], yerr=errs[:, 0], linestyle='', marker='^')
    plt.savefig('simplices_figures/mtype_indegree.pdf')

def cell_type_analysis():
    df_0 = analysis_exp_25_decoupled.get_reliabilities_experiment_25(decouple=False, sigma_conv=5, dt_conv=1, spont=False,
                                           network_removal=False, vpm_ca2p0=False).mean(axis=1)
    df_neurons = analysis_exp_25_decoupled.get_selected_L456_gids()
    df_neurons = df_neurons.loc[np.unique(df_0.index)]
    print df_neurons
    mtypes = np.unique(df_neurons[df_neurons['synapse_class'] == 'EXC']['mtype'])
    print mtypes
    means = np.zeros(len(mtypes))
    errs = np.zeros(len(mtypes))

    fig, ax = plt.subplots()
    for i, mtype in enumerate(mtypes):
        print mtype
        print (df_neurons['mtype'] == mtype).sum()
        values_mtype = df_0.loc[df_neurons[df_neurons['mtype'] == mtype].index].values
        means[i] = values_mtype.mean()
        errs[i] = simplices_graph_theory.mean_confidence_interval(values_mtype, confidence=0.95)
        ax.scatter(i + np.zeros(values_mtype.size) + np.random.uniform(low=-0.3, high=0.3, size=values_mtype.size), values_mtype , marker='.', alpha=0.4)
    ax.bar(np.arange(len(mtypes)), means, yerr=errs,  fill=False, edgecolor='blue')
    ax.set_xticks(np.arange(len(mtypes)))
    ax.set_ylim([0, 1])

    ax.set_xticklabels(mtypes,  rotation='vertical')
    plt.savefig('simplices_figures/cell_types_exc.pdf')

    mtypes = np.unique(df_neurons[df_neurons['synapse_class'] == 'INH']['mtype'])
    print mtypes
    means = np.zeros(len(mtypes))
    errs = np.zeros(len(mtypes))
    fig, ax = plt.subplots()
    for i, mtype in enumerate(mtypes):
        print mtype
        print (df_neurons['mtype'] == mtype).sum()
        values_mtype = df_0.loc[df_neurons[df_neurons['mtype'] == mtype].index].values
        means[i] = values_mtype.mean()
        errs[i] = simplices_graph_theory.mean_confidence_interval(values_mtype, confidence=0.95)
        ax.scatter(i + np.zeros(values_mtype.size) + np.random.uniform(low=-0.3, high=0.3, size=values_mtype.size), values_mtype , marker='.', alpha=0.4)
    ax.bar(np.arange(len(mtypes)), means, yerr=errs, fill=False, edgecolor='blue')
    ax.set_xticks(np.arange(len(mtypes)))

    ax.set_xticklabels(mtypes,  rotation='vertical')
    plt.savefig('simplices_figures/cell_types_inh.pdf')

    etypes = np.unique(df_neurons[df_neurons['synapse_class'] == 'INH']['etype'])
    print etypes
    means = np.zeros(len(etypes))
    errs = np.zeros(len(etypes))
    fig, ax = plt.subplots()
    for i, etype in enumerate(etypes):
        print etype
        print (df_neurons['etype'] == etype).sum()
        values_mtype = df_0.loc[df_neurons[df_neurons['etype'] == etype].index].values
        means[i] = values_mtype.mean()
        errs[i] = simplices_graph_theory.mean_confidence_interval(values_mtype, confidence=0.95)
        ax.scatter(i + np.zeros(values_mtype.size) + np.random.uniform(low=-0.3, high=0.3, size=values_mtype.size), values_mtype , marker='.', alpha=0.4)
    ax.bar(np.arange(len(etypes)), means, yerr=errs, fill=False, edgecolor='blue')
    ax.set_xticks(np.arange(len(etypes)))

    ax.set_xticklabels(etypes,  rotation='vertical')
    plt.savefig('simplices_figures/cell_types_inh_etypes.pdf')


plot_in_degree_simplices()
plot_influence()
cell_type_analysis()
in_degree_mtype()
firing_rate_comparison()