import numpy as np
import bluepy
import pandas as pd

n_gids_perturbed = np.zeros(19)
n_additional_spikes = np.zeros(19)
t_additional_spikes = np.zeros(19)
maxs_hist = np.zeros((19, 6))

for j, s in enumerate(np.delete(np.arange(150, 170), 17)):
    config_1 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_x/seed%d/BlueConfig' % s
    config_2 = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_g/seed%d/BlueConfig' % s

    s1 = bluepy.Simulation(config_1)
    s2 = bluepy.Simulation(config_2)

    report1 = s1.v2.spikes.get()
    report2 = s2.v2.spikes.get()


    n_spikes = report1.values.size
    spikes = pd.DataFrame({'time': report1.index, 'gid': report1.values}, index=np.zeros(report1.values.size, dtype=int))
    spikes_pert = pd.DataFrame({'time': report2.index, 'gid': report2.values}, index=np.ones(report2.values.size, dtype=int))

    gids = np.unique(np.hstack([spikes.index, spikes_pert.index]))

    comb = pd.concat([spikes, spikes_pert])

    duplicates = comb.drop_duplicates(keep=False)
    perturbed_gids = np.unique(np.unique(duplicates.gid))
    n_gids_perturbed[j] = perturbed_gids.size

    maxs = []
    extra_spikes = np.zeros(perturbed_gids.size)
    for i, gid in enumerate(perturbed_gids):
        dup_gid = duplicates[duplicates['gid'] == gid]

        if np.unique(dup_gid.index).size == 1:
            extra_spikes[i] == dup_gid.time.min()
        elif dup_gid.loc[0].size == dup_gid.loc[1].size:
            maxs.append(np.abs(np.array(dup_gid.loc[0].time) - np.array(dup_gid.loc[1].time)))
            extra_spikes[i] = -1
        else:
            extra_spikes[i] = np.min([dup_gid.loc[0].time.min(), dup_gid.loc[1].time.min()])


    maxs = np.hstack(maxs)
    maxs_hist[j, :] = np.histogram(maxs, [-0.5, 0.05, 1, 20, 100, 1000, 2000])[0]
    print extra_spikes
    n_additional_spikes[j] = (extra_spikes > 0).sum()
    t_additional_spikes[j] = extra_spikes[(extra_spikes > -0)].min()


print "----- perturbed spike gids ----"
print n_gids_perturbed
print n_gids_perturbed.mean()
print n_gids_perturbed.std(ddof=1)
print "---- gids that had extra or missing spikes ------"
print n_additional_spikes
print n_additional_spikes.mean()
print n_additional_spikes.std(ddof=1)
print "--- t first extra or missing spike ---"
print t_additional_spikes
print np.median(t_additional_spikes)
print t_additional_spikes.min()
print t_additional_spikes.max()
print "--- t shifted spikes ---"
print maxs_hist.mean(axis=0)/maxs_hist.sum(axis=1).mean()
