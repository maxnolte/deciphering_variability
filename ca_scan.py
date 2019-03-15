import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

import bluepy


cas_g = ['1p26', '1p28', '1p3', '1p32', '1p34', '1p36', '1p38', '1p4', '2p0']
cas_mvr = ['1p1', '1p15', '1p2', '1p21', '1p22', '1p23', '1p24', '1p25']

bcs_mvr = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/ca_scan_mvr/Ca%s/BlueConfig' % s for s in cas_mvr]
bcs_g = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/ca_scan_g/Ca%s/BlueConfig' % s for s in cas_g]

# bcs = ['/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/ei-balance/' \
#        'scan_layer5/Ca%s/BlueConfig' % s for s in cas]

sim = bluepy.Simulation(bcs_mvr[0])
gids = np.array(list(sim.get_circuit_target()))
gids_exc = np.random.permutation(np.intersect1d(np.array(list(sim.circuit.get_target('Excitatory'))), gids))
gids_inh = np.random.permutation(np.intersect1d(np.array(list(sim.circuit.get_target('Inhibitory'))), gids))

# bcs = bcs_0
names = ['MVR', 'det_syns']
for k, (bcs, cas) in enumerate(zip([bcs_mvr, bcs_g], [cas_mvr, cas_g])):
    fig, axs = plt.subplots(len(bcs), 2, figsize=(14, 14))

    for i, bc in enumerate(bcs):
        print bc
        sim = bluepy.Simulation(bc)

        gids = np.array(list(sim.get_circuit_target()))
        od = sim.reports.spike.outdat

        ax = axs[i, 0]
        ax.set_xlim([0, 4000])
        bins = np.linspace(0, 5000, 251)
        fr_norm = (bins[1] - bins[0])/1000.0

        spikes = np.hstack(od.spikes_for_gids(gids_inh))[:100]
        ax.hist(spikes, bins=bins, histtype='step', color='blue', weights=np.ones(spikes.size) * fr_norm)
        spikes = np.hstack(od.spikes_for_gids(gids_exc))[:1000]
        ax.hist(spikes, bins=bins, histtype='step', color='darkred', weights=np.ones(spikes.size) * fr_norm)

        ax.set_xlabel('t (ms)')
        ax.set_ylabel('FR (Hz)')
        ax.set_title('[Ca2+] = %s mM' % cas[i])

        ax = axs[i, 1]
        all_times = []
        all_vmin = []
        all_vmax = []
        colors = []
        for j, gid in enumerate(np.hstack([gids_inh[:100], gids_exc[:1000]])):  #[np.random.permutation(gids.size)][:10000]):
            exc = sim.circuit.mvddb.get_gid(gid).mtype.synapse_class == 'EXC'
            times = od.spikes_for_gid(gid)
            times = times[times > 0.0]
            all_times.append(times)
            all_vmin.append(np.zeros(times.size) + j)
            all_vmax.append(np.zeros(times.size) + j + 10)
            color_options = ['blue', 'red']
            colors += [color_options[exc] for t in times]
        ax.vlines(np.hstack(all_times), np.hstack(all_vmin), np.hstack(all_vmax), colors=colors, rasterized=True, linewidth=0.5)
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('Cells')
        ax.set_title('[Ca2+] = %s mM' % cas[i])

    plt.tight_layout()
    plt.savefig('ca_scan_%s.pdf' % names[k], dpi=300)