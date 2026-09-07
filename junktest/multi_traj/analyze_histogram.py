"""
Example analysis script -- a TEMPLATE you copy and edit per plot.

The multi-trajectory analogue of the bottom-right panel of plot_histogram.py: a single-panel
figure showing the histogram of ALL bead coordinates -- pooled across every bead, every time step,
AND every trajectory in data.hdf -- with the two analytic harmonic distributions overlaid
(classical and quantum-no-friction Gaussians).

Build data.hdf first with:  python -m workflow.consolidate --config config.py

    python analyze_histogram.py --file=data.hdf --bins=99 --tskip=10.0

    --file   consolidated HDF5 (default: data.hdf)
    --bins   number of histogram bins (default: 99, as in plot_histogram.py)
    --tskip  discard the first tskip a.u. of each trajectory before pooling -- drops the
             equilibration transient so the histogram samples the equilibrium distribution
             (default: 0).
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from workflow import open_run            # h5py.File(path, 'r')


def main():
    ap = argparse.ArgumentParser(description='Pooled bead-position histogram across trajectories.')
    ap.add_argument('--file',  default='data.hdf', help='consolidated HDF5 (default: data.hdf)')
    ap.add_argument('--bins',  type=int,   default=99,  help='histogram bins (default: 99)')
    ap.add_argument('--tskip', type=float, default=0.0, help='discard data before this time, a.u. (default: 0)')
    a = ap.parse_args()

    with open_run(a.file) as f:
        time  = f['time'][:]
        delt  = float(time[1] - time[0])
        nbds  = int(f.attrs['nbds'])
        nnuc  = int(f.attrs['nnuc'])
        ntraj = f['nucR'].shape[0]
        T     = f['nucR'].shape[1]

        # harmonic-distribution parameters (from the stored config). beta here is the REGULAR
        # temperature (not beta/nbds); kforce = force constant = first potparam (kvec).
        beta   = float(f.attrs['beta'])
        gamma  = float(f.attrs['gamma'])
        cfg    = json.loads(f.attrs['config_json'])
        mass   = float(np.atleast_1d(cfg['mass'])[0])
        kforce = float(cfg['kvec'])

        # discard the first tskip a.u. of each trajectory (equilibration transient)
        i_skip = int(round(a.tskip / delt))
        if T - i_skip <= 1:
            raise ValueError(f'tskip={a.tskip} leaves {T - i_skip} rows (T={T}, dt={delt}); too large')

        # pool the reaction coordinate (nuclear DOF 0) of every bead, every time, every trajectory.
        # read one trajectory at a time so the whole run never sits in RAM at once.
        pooled = []
        for i in range(ntraj):
            R = f['nucR'][i][i_skip:]                 # (nt_eff, nbds, nnuc)
            pooled.append(R[:, :, 0].ravel())         # DOF 0 (plot_histogram.py assumes nnuc=1)
            print(f'[analyze] pooled trajectory {i + 1}/{ntraj}', flush=True)
        beads = np.concatenate(pooled)

    # ------------------------------------------------------------------ style (matches plot_histogram)
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 11.5,
        'axes.linewidth': 0.9, 'axes.edgecolor': '#3a3a3a',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'legend.frameon': False, 'legend.fontsize': 10, 'figure.dpi': 120,
    })
    INK, C_HIST = '#222222', '#2a78d6'

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # ---- histogram of ALL pooled bead coordinates
    ax.hist(beads, bins=a.bins, color=C_HIST, edgecolor='white', linewidth=0.6,
            density=True, label='GLE-RPMD')

    # ---- analytic harmonic distributions (hbar = 1). Both Gaussians centered on the sampled mean
    # so they overlay the histogram wherever the well sits. Needs beta, mass m, omega = sqrt(k/m).
    if kforce > 0:
        omega = np.sqrt(kforce / mass)
        x0    = beads.mean()
        xg    = np.linspace(beads.min(), beads.max(), 400)

        # P_classical(x) = sqrt(beta m omega^2 / 2pi) exp[ -(beta m omega^2 / 2) (x-x0)^2 ]
        ac = beta * mass * omega**2
        Pc = np.sqrt(ac / (2 * np.pi)) * np.exp(-0.5 * ac * (xg - x0)**2)

        # P_quantum(x) = sqrt(m omega/pi tanh(beta omega/2)) exp[ -m omega tanh(beta omega/2) (x-x0)^2 ]
        aq = mass * omega * np.tanh(beta * omega / 2)
        Pq = np.sqrt(aq / np.pi) * np.exp(-aq * (xg - x0)**2)

        ax.plot(xg, Pc, color='#e34948', lw=2.0, label='Classical')
        ax.plot(xg, Pq, color='#1baf7a', lw=2.0, label='Quantum (no friction)')
        ax.legend(loc='upper right')

    ax.set_xlabel('bead position  (a.u.)')
    ax.set_ylabel('probability density')
    ax.set_title(f'Bead position histogram (all beads + {ntraj} trajectories pooled, {a.bins} bins)',
                 loc='left')

    # ---- cosmetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color='#000000', alpha=0.07, lw=0.7)
    ax.margins(x=0.01)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)

    fig.suptitle(f'$\\gamma = {gamma}$', x=0.5, y=0.98, ha='center', fontsize=22, fontweight='bold')
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.10)

    print('wrote histogram.png?')
    fig.savefig('histogram.png', dpi=200)
    print('wrote histogram.png')
    #plt.show()


if __name__ == '__main__':
    main()
