import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import initial_analysis_final as iaf
import bluepy


def analyse_inititial_stim_fluctuations(plot_values = [10, 15, 18, 1, 3]):
    variances_percent = np.array([0.001, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1.5, 2.0,
                                  0.0, 0.00001, 0.0001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
    # variances = ['0p01 y', '0p05 y', '0p1 y', '0p5 y', '1p0 y', '1p5 n','2p0 n','10p0 y']

    labels = variances_percent[plot_values]
    labels[0] = variances_percent[0]

    bcs_stim = iaf.get_change_bcs_decouple(params='dz') # 'ab'
    bcs = iaf.get_change_bcs_decouple(params='x')

    differences = {}
    spike_differences = np.zeros(len(plot_values))
    for j, i in enumerate(plot_values): # [0] + range(10, 19):

        soma_stim, bins = iaf.get_soma_time_series(bcs_stim[i], t_start=2500, t_end=3500) #, gids=[72890])
        soma, bins = iaf.get_soma_time_series(bcs[i], t_start=2500, t_end=3500) #, gids=[72890])

        spikes_1 = np.array(bluepy.Simulation(bcs_stim[i]).v2.reports['spikes'].data()).size
        spikes_2 = np.array(bluepy.Simulation(bcs[i]).v2.reports['spikes'].data()).size
        print spikes_1
        spike_differences[j] = (spikes_1 - spikes_2)/31346.0/2.0

        print soma.shape
        differences[i] = (np.array(soma_stim) - np.array(soma)).T


    fig, axs = plt.subplots(5, figsize=(10, 15))
    ax = axs[0]
    ax.plot(variances_percent[plot_values], spike_differences, marker='x', lw=1.0)
    ax.set_ylabel('dFR (Hz)')
    ax.set_xlabel('variance in percent')

    for j, i in enumerate(plot_values):
        ax = axs[1]
        ax.plot(bins[500:1000], differences[i][500:1000, 15421], lw=0.8, label=variances_percent[i])

        ax = axs[2]
        ax.hist(differences[i].flatten(), bins=np.linspace(-2, 2, 201), label=" %.e" % variances_percent[i],
                histtype='stepfilled', alpha=0.6)

        ax = axs[3]
        ax.scatter(labels[j], [differences[i].flatten().std()])
        ax.set_xlabel('percent variance')
        ax.set_ylabel('soma fluctuations (mV)')
        ax = axs[4]
        ax.scatter([differences[i].flatten().std()], spike_differences[j])
        ax.set_xlabel('soma fluctuations (mV)')
        ax.set_ylabel('dFR (Hz)')

    ax = axs[3]
    print labels[1:4]
    print fluctuation_variance(n=3)
    ax.scatter(labels[1:4], fluctuation_variance(n=3))
    # axs[3].set_xticks(np.arange(len(plot_values)))
    # axs[3].set_xticklabels(variances_percent[plot_values])

    ax.legend()
    axs[1].legend()
    plt.tight_layout()
    plt.savefig('figures/stim_variance.pdf')


def compare_fluctuations():
    parameters = ['abcd', 'ab', 'cd', 'e']
    n=10
    all_stds = np.zeros((len(parameters), n))
    for i, params in enumerate(parameters):
        all_stds[i, :] = fluctuation_variance(params=params, n=n)
    fig, ax = plt.subplots()
    for i, params in enumerate(parameters):
        ax.scatter(i * np.ones(n), all_stds[i, :], alpha=0.4)
    ax.errorbar(np.arange(len(parameters)), all_stds.mean(axis=1), yerr=all_stds.std(axis=1, ddof=1))
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_xticklabels(parameters)
    plt.savefig('figures/decoupled_fluctuations.pdf')


def fluctuation_variance(params='ab', n=3):
    bcs_stim = iaf.get_change_bcs_decouple(params=params) # 'ab'
    bcs = iaf.get_change_bcs_decouple(params='x')
    stds = np.zeros(n)
    for i in range(n): # [0] + range(10, 19):

        soma_stim, bins = iaf.get_soma_time_series(bcs_stim[i], t_start=2050, t_end=3500)
        soma, bins = iaf.get_soma_time_series(bcs[i], t_start=2050, t_end=3500)

        stds[i] = (np.array(soma_stim) - np.array(soma)).flatten().std(ddof=1)
        print stds
    return stds


def get_initial_fluctuations(plot_values = [0, 15, 17, 18, 19, 1, 8, 9, 3]):
    variances_percent = np.array([0.001, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1.5, 2.0,
                                  0.0, 0.00001, 0.0001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
    # variances = ['0p01 y', '0p05 y', '0p1 y', '0p5 y', '1p0 y', '1p5 n','2p0 n','10p0 y']

    labels = variances_percent[plot_values]

    bcs_stim = iaf.get_change_bcs_decouple(params='dz') # 'ab'
    bcs = iaf.get_change_bcs_decouple(params='x')

    differences = np.zeros(len(plot_values))
    for j, i in enumerate(plot_values): # [0] + range(10, 19):
        print "j %d i %d" % (j, i)
        soma_stim, bins = iaf.get_soma_time_series(bcs_stim[i], t_start=2500, t_end=3500) #, gids=[72890])
        soma, bins = iaf.get_soma_time_series(bcs[i], t_start=2500, t_end=3500) #, gids=[72890])

        print soma.shape
        differences[j] = (np.array(soma_stim) - np.array(soma)).std()
    return differences, labels


if __name__ == "__main__":
    print fluctuation_variance()
    analyse_inititial_stim_fluctuations()
    # compare_fluctuations()