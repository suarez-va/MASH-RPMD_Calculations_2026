"""
Animate the histogram of ring-polymer bead positions, pooled over ALL trajectories, versus time, into
an .mp4 -- analogous to EDMD's real_time_De1.15/4_plot_rhot.py.

At each stored time point it pools every bead of every trajectory in data.hdf, histograms them on a
FIXED grid (TOTAL_WIDTH x TOTAL_BINS, centered on CENTER), normalizes to a probability density
(area -> 1 when the window contains all beads, so mass moving donor->acceptor is visible), and
animates the frames with matplotlib's FFMpegWriter.

Each trajectory is weighted by W = 2*|mapSz(t=0)| (the MASH initial-population weight), so a
trajectory started at mapSz=-0.25 contributes half as much as one started at mapSz=-0.5.

    python plot_rho_movie.py [--file data.hdf] [--out rho_movie.mp4]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from workflow import open_run     # h5py.File(path, 'r')

# ------------------------------------------------------------------ fixed histogram grid + movie knobs
TOTAL_WIDTH = 8.0      # a.u.; histogram window is [CENTER - W/2, CENTER + W/2]
TOTAL_BINS  = 100       # number of bins across that window
CENTER      = 0.0      # window center (a.u.)
N_FRAMES    = 300      # target movie frames (evenly subsampled if fewer time points exist -> uses all)
FPS         = 30


def main():
    ap = argparse.ArgumentParser(description='Animated pooled bead-position histogram vs time.')
    ap.add_argument('--file', default='data.hdf')
    ap.add_argument('--out',  default='rho_movie.mp4')
    a = ap.parse_args()

    edges   = np.linspace(CENTER - TOTAL_WIDTH / 2, CENTER + TOTAL_WIDTH / 2, TOTAL_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx      = edges[1] - edges[0]

    # ---- accumulate per-frame pooled histograms (read one trajectory chunk at a time) ----
    with open_run(a.file) as f:
        tgrid = f['time'][:]
        ntraj, T, nbds = f['nucR'].shape[0], f['nucR'].shape[1], f['nucR'].shape[2]
        fidx   = np.unique(np.linspace(0, T - 1, min(N_FRAMES, T)).astype(int))   # time indices -> frames
        has_w = 'mapSz' in f
        if not has_w:
            print('[movie] WARNING: no mapSz in file -> falling back to equal trajectory weights')
        counts = np.zeros((fidx.size, TOTAL_BINS))
        wsum   = 0.0
        for i in range(ntraj):
            Ri = f['nucR'][i][:, :, 0]                 # (T, nbds), DOF 0 -- one lazy chunk read
            # trajectory weight W = 2*|mapSz(t=0)| (MASH initial-population weight); bead-mean of the
            # t=0 spin (uniform across beads for a prepared state). Equal weights if mapSz is absent.
            #Wi = 2.0 * abs(f['mapSz'][i, 0, :].mean()) if has_w else 1.0
            Wi = 2.0 * abs(f['mapSz'][i, 0, :].mean())*np.heaviside(-f['mapSz'][i, 0, :].mean(), 0.5) if has_w else 1.0
            wsum += Wi
            for fi, t in enumerate(fidx):
                counts[fi] += Wi * np.histogram(Ri[t], bins=edges)[0]
            print(f'[movie] pooled trajectory {i + 1}/{ntraj}  (W={Wi:.3f})', flush=True)
    dens    = counts / (wsum * nbds * dx)              # weight-normalized density (area -> 1 in-window)
    tframes = tgrid[fidx]

    # ---- figure (styled after 4_plot_rhot.py) ----
    plt.rcParams.update({
        'figure.dpi': 150, 'axes.linewidth': 1.5,
        'axes.labelsize': 15, 'axes.titlesize': 15,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, dens.max() * 1.08 if dens.max() > 0 else 1.0)
    ax.set_xlabel('bead position  (a.u.)')
    ax.set_ylabel(r'$\rho(x)$  (a.u.)')
    ax.grid(True, color='k', alpha=0.08, lw=0.7)
    title = ax.set_title("")
    bars  = ax.bar(centers, dens[0], width=dx, align='center',
                   color='#2a78d6', edgecolor='white', linewidth=0.3)

    def update(fi):
        for rect, h in zip(bars, dens[fi]):
            rect.set_height(h)
        title.set_text(f"t = {tframes[fi]:.2f} a.u.   ({ntraj} trajectories $\\times$ {nbds} beads)")
        return list(bars) + [title]

    ani = FuncAnimation(fig, update, frames=fidx.size, blit=False)
    ani.save(a.out, writer=FFMpegWriter(fps=FPS))
    print(f'[movie] wrote {a.out}  ({fidx.size} frames, {ntraj} trajectories x {nbds} beads, '
          f'total weight {wsum:.2f}, grid {TOTAL_BINS} bins over [{edges[0]:.2f}, {edges[-1]:.2f}])')


if __name__ == '__main__':
    main()
