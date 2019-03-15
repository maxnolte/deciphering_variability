import bluepy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

n = 8



# gidpre seed 156 = 75105
# post l4 pc 74062
def plot_abcd_f_g():
    bc_x = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_x/seed156/BlueConfig'

    a = bluepy.Simulation(bc_x).v2.reports['soma'].data()
    times = a.axes[1]
    gid_min = a.axes[0].min()
    print a.axes[0][11370]
    soma_x = np.array(a)

    fig, axs = plt.subplots(4, 3, figsize=(12, 6))
    for k in range(3):
        changes = ['abcd', 'f', 'g']
        bc_p = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_%s/seed156/BlueConfig' % changes[k]


        soma_p = np.array(bluepy.Simulation(bc_p).v2.reports['soma'].data())

        print soma_x.shape
        print soma_p.shape


        ax = axs[0, k]
        n = 10000
        ax.plot(times[:n], soma_x[:, :n].mean(axis=0))
        ax.plot(times[:n], soma_p[:, :n].mean(axis=0))

        ax = axs[1, k]

        j = 11370 #
        print gid_min
        print j
        ax.plot(times[:n], soma_p[:, :n].mean(axis=0) - soma_x[:, :n].mean(axis=0))


        ax = axs[2, k]
        ax.plot(times[:n], soma_x[j, :n])
        ax.plot(times[:n], soma_p[j, :n])

        ax = axs[3, k]

        ax.plot(times[:n], soma_p[j, :n] - soma_x[j, :n])


        for ax in axs.flatten():
            ax.set_xlabel('t (ms)')
            ax.set_ylabel('r')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.tight_layout()

    plt.savefig('figures/test_decouple_perturbation.pdf')


def plot_a_ab_c():
    bc_x = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_x/seed156/BlueConfig'

    a = bluepy.Simulation(bc_x).v2.reports['soma'].data()
    times = a.axes[1]
    gid_min = a.axes[0].min()
    print a.axes[0][93851 - gid_min]
    soma_x = np.array(a)

#     93851      bIR L6_MC
# 93876      bIR
# 93888    bSTUT
# 93937      bIR
# 93999    dSTUT
# 94001      bIR
# 94032      bIR
# 94037      bIR

    fig, axs = plt.subplots(4, 3, figsize=(12, 6))
    for k in range(3):
        changes = ['ab', 'c', 'd']
        bc_p = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_%s/seed156/BlueConfig' % changes[k]


        soma_p = np.array(bluepy.Simulation(bc_p).v2.reports['soma'].data())

        print soma_x.shape
        print soma_p.shape


        ax = axs[0, k]
        n = 2000
        ax.plot(times[:n], soma_x[:, :n].mean(axis=0))
        ax.plot(times[:n], soma_p[:, :n].mean(axis=0))

        ax = axs[1, k]

        j = 93851 - gid_min #
        print gid_min
        print j
        ax.plot(times[:n], soma_p[:, :n].mean(axis=0) - soma_x[:, :n].mean(axis=0))


        ax = axs[2, k]
        ax.plot(times[:n], soma_x[j, :n])
        ax.plot(times[:n], soma_p[j, :n])

        ax = axs[3, k]

        ax.plot(times[:n], soma_p[j, :n] - soma_x[j, :n])


        for ax in axs.flatten():
            ax.set_xlabel('t (ms)')
            ax.set_ylabel('r')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.tight_layout()

    plt.savefig('figures/test_decouple_perturbation_ab_c_d.pdf')

def plot_ds():

#     93851      bIR L6_MC
# 93876      bIR
# 93888    bSTUT
# 93937      bIR
# 93999    dSTUT
# 94001      bIR
# 94032      bIR
# 94037      bIR
    changes = ['0p01', '0p05', '0p1', '0p5', '1p0', '1p5', '2p0', '10p0']

    fig, axs = plt.subplots(2, 8, figsize=(24, 4))
    for k in range(len(changes)):
        bc_x = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_stim/change_x/seed152/variance%s/BlueConfig' % changes[k]
        bc_p = '/gpfs/bbp.cscs.ch/project/proj9/simulations/nolte/variability/spontaneous_v2/continue_change_decouple_stim/change_d/seed152/variance%s/BlueConfig' % changes[k]
        a = bluepy.Simulation(bc_x).v2.reports['soma'].data()
        times = a.axes[1]
        gid_min = a.axes[0].min()
        print a.axes[0][11370]
        soma_x = np.array(a)
        soma_p = np.array(bluepy.Simulation(bc_p).v2.reports['soma'].data())

        print soma_x.shape
        print soma_p.shape


        j = 11370 #

        ax = axs[0, k]
        ax.plot(times[10000:12001]/10.0, soma_x[j, 10000:12001])
        ax.plot(times[10000:12001]/10.0, soma_p[j, 10000:12001])
        ax.set_ylim([-68, -58])
        ax = axs[1, k]
        ax.plot(times[10000:12001]/10.0, soma_p[j, 10000:12001] - soma_x[j, 10000:12001])


        for ax in axs.flatten():
            ax.set_xlabel('t (ms)')
            ax.set_ylabel('V (mV)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.tight_layout()

    plt.savefig('figures/test_decouple_perturbation_ds.pdf')


plot_ds()