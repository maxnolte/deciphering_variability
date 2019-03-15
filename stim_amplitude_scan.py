import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import bluepy


variances = ['0p001', '0p01', '0p05', '0p1', '0p5', '1p0', '1p5', '2p0', '10p0']

bcs = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_abcd_stim/seed170/variance%s/BlueConfig' % s for s in variances[1:]]
bcs = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous/base_seeds_abcd/seed170/BlueConfig'] + bcs

# bcs = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/ei-balance/' \
#        'scan_layer5/Ca%s/BlueConfig' % s for s in cas]

sim = bluepy.Simulation(bcs[0])
gids = np.array(list(sim.get_circuit_target()))
gids_exc = np.random.permutation(np.intersect1d(np.array(list(sim.circuit.get_target('Excitatory'))), gids))
gids_inh = np.random.permutation(np.intersect1d(np.array(list(sim.circuit.get_target('Inhibitory'))), gids))

# bcs = bcs_0
names = ['MVR', 'det_syns']
fig, axs = plt.subplots(len(bcs), 2, figsize=(14, 14))

for i, bc in enumerate(bcs):
    print bc
    sim = bluepy.Simulation(bc)


    ax = axs[i, 0]

    spikes = bluepy.Simulation(bc).v2.reports['spikes']
    df = spikes.data(t_start=1000.0)
    gids_spiking = np.abs(np.array(df.axes[0]) - gids.max())
    times = np.array(df)
    ax.vlines(times, gids_spiking, gids_spiking + 200, rasterized=True, lw=0.3)
    ax2 = ax.twinx()
    ax2.hist(times, bins=np.linspace(1000, 2000, 101), histtype='step', weights=np.zeros(times.size) + (1000.0/10.0)/gids.size)
    ax2.set_ylabel('FR (Hz)')
 #   ax2.set_ylim([0, 3])
 #   ax2.set_yticks([0, 1, 2, 3])

    ax.set_xlabel('t (ms)')
    ax.set_ylabel('Neurons')
    ax.set_title('variance in percent: %s' % variances[i])

    plt.tight_layout()
    plt.savefig('figures/variance_raster.pdf', dpi=300)