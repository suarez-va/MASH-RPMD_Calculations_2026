"""
plot_rho_movie.py -- animate the quantum-jump-weighted BRANCH densities

    rho_+(R,t) = < W_PP^(m)  delta(R - R(t))  h( S_z(t) ) >      upper adiabat, BLUE
    rho_-(R,t) = < W_PP^(m)  delta(R - R(t))  h(-S_z(t) ) >      lower adiabat, RED

into an .mp4.  This is analyze_cpm.py's c_+(t) / c_-(t) with the trajectory average resolved in R
instead of collapsed to a scalar -- same estimator, one level more detail.  The normalization is
chosen so that relation is exact:

    integral rho_+(R,t) dR = c_+(t),   integral rho_-(R,t) dR = c_-(t),
    integral (rho_+ + rho_-) dR = <W_PP^(m)>(t)

which is the script's main self-check.  NOTE this differs from the System-Bath template
(low_friction/plot_rho_movie.py), which divides by the total weight to force area -> 1.  That would
be wrong here: the whole point is that weight MOVES between the two branches, so the areas must be
allowed to change.

--- weighting ---------------------------------------------------------------------------------

The template weights each whole trajectory by the constant 2|mapSz(0)|.  Here every BEAD at every
frame carries its own weight,

    weight of bead k of trajectory i at time t  =  W_PP^(m)(i,t) * h(+/- S_z(i,t,k))

with W_PP^(m) taken at m(t) = #{jumps with t_jump <= t}, i.e. every jump up to AND INCLUDING t --
matching mash_rpmd, which writes the post-jump state at t_jump.  Summing over beads and dividing by
nbds reproduces exactly the bead-mean convention analyze_cpm.py uses, so the two scripts agree for
nbds > 1 as well as for the nbds == 1 runs in this tree.

In MASH sgn(S_z) selects the active adiabat (mash_rpmd.get_PE weights E_bo[:,1] by +1), so rho_+ is
the upper surface and rho_- the lower.  traj.py starts every trajectory on the lower adiabat via
init_map_spin(0), so rho_+ is identically zero until the packet reaches the crossing.

--- window ------------------------------------------------------------------------------------

The histogram grid is FIXED (R_MIN, R_MAX, N_BINS) so frames are comparable and no cross-trajectory
information is needed.  Beads outside the window are dropped; the script tracks how much weight that
costs and reports the worst frame, so clipping is never silent.

    python plot_rho_movie.py [--file data.hdf] [--out rho_movie.mp4]
    python plot_rho_movie.py --rmin -15 --rmax 15 --bins 149 --frames 400
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from workflow import open_run, expand_weights   # h5py.File(path,'r'); per-jump -> dense weights

# ------------------------------------------------------------------ fixed histogram grid + movie knobs
R_MIN    = -12.0      # a.u.; histogram window is [R_MIN, R_MAX]
R_MAX    =  12.0
N_BINS   =  99        # bins across that window
N_FRAMES = 300        # target movie frames (evenly subsampled; uses all if fewer time points exist)
FPS      =  30


def accumulate(path, rmin=R_MIN, rmax=R_MAX, bins=N_BINS, n_frames=N_FRAMES,
               slab=5000, w0_only=False):
    """
    Build rho_+ and rho_- on the fixed grid at the selected frame times.

    Returns a dict with edges/centers/dR, tframes, rho_p/rho_m of shape (n_frames, bins), the
    per-frame total weight in each branch (= c_+(t), c_-(t) before clipping) and the in-window part.
    """
    edges   = np.linspace(rmin, rmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dR      = edges[1] - edges[0]

    with open_run(path) as f:
        tgrid  = f['time'][:]
        n_traj, T, nbds = f['nucR'].shape[0], f['nucR'].shape[1], f['nucR'].shape[2]

        has_w = 'weights' in f
        if not has_w and not w0_only:
            raise KeyError(
                f"{path} has no /weights dataset. Either the run was made without Tjump, or it was "
                f"consolidated with a version of workflow.consolidate that predates the quantum-jump "
                f"weights. Re-run `python -m workflow.consolidate --config <config.py>` to pick them "
                f"up, or pass --w0-only to deliberately analyse the no-jump baseline W^(0).")

        fidx    = np.unique(np.linspace(0, T - 1, min(n_frames, T)).astype(int))
        tframes = tgrid[fidx]
        F       = fidx.size

        cnt_p = np.zeros((F, bins))     # in-window weight sums
        cnt_m = np.zeros((F, bins))
        tot_p = np.zeros(F)             # ALL weight, in or out of window -> c_+(t)*n_traj*nbds
        tot_m = np.zeros(F)
        n_jumps = 0

        for lo in range(0, n_traj, slab):
            hi  = min(lo + slab, n_traj)
            sel = slice(lo, hi)

            R  = f['nucR'][sel, :, :, 0][:, fidx, :]     # (s, F, nbds)
            Sz = f['mapSz'][sel][:, fidx, :]             # (s, F, nbds)

            if w0_only:
                # no-jump baseline W^(0)_PP = 2|Sz(0)|; abs of the BEAD MEAN, matching
                # mash_rpmd.init_jump_weights, which sets Sz0 = np.mean(self.mapSz)
                Sz0  = f['mapSz'][sel][:, 0, :].mean(axis=1)          # (s,)
                W_PP = 2.0 * np.abs(Sz0)[:, None] * np.ones((1, F))   # (s,F)
            else:
                W_PP = expand_weights(f, time=tframes, sel=sel)['W_PP']   # (s,F)
                n_jumps += int(f['n_weights'][sel].sum() - (hi - lo))     # row 0 is t=0, not a jump

            # per-bead branch weights. heaviside(.,0.5) at Sz==0 matches sgn(0)=0, so summing over
            # beads and dividing by nbds equals analyze_cpm.py's bead-mean h.
            wp = W_PP[:, :, None] * np.heaviside( Sz, 0.5)      # (s,F,nbds)
            wm = W_PP[:, :, None] * np.heaviside(-Sz, 0.5)

            tot_p += wp.sum(axis=(0, 2))
            tot_m += wm.sum(axis=(0, 2))
            for fi in range(F):
                cnt_p[fi] += np.histogram(R[:, fi, :].ravel(), bins=edges,
                                          weights=wp[:, fi, :].ravel())[0]
                cnt_m[fi] += np.histogram(R[:, fi, :].ravel(), bins=edges,
                                          weights=wm[:, fi, :].ravel())[0]

    norm  = n_traj * nbds * dR
    rho_p = cnt_p / norm
    rho_m = cnt_m / norm

    # c_+(t), c_-(t): the full averages, and the part that actually landed inside the window
    c_p, c_m = tot_p / (n_traj * nbds), tot_m / (n_traj * nbds)
    in_p, in_m = cnt_p.sum(axis=1) / (n_traj * nbds), cnt_m.sum(axis=1) / (n_traj * nbds)

    return dict(edges=edges, centers=centers, dR=dR, tframes=tframes,
                rho_p=rho_p, rho_m=rho_m, c_p=c_p, c_m=c_m, in_p=in_p, in_m=in_m,
                n_traj=n_traj, nbds=nbds, n_jumps=n_jumps, w0_only=w0_only,
                rmin=rmin, rmax=rmax, bins=bins)


def animate(out, r, fps=FPS):
    centers, dR = r['centers'], r['dR']
    rho_p, rho_m, tframes = r['rho_p'], r['rho_m'], r['tframes']

    plt.rcParams.update({
        'figure.dpi': 150, 'axes.linewidth': 1.5,
        'axes.labelsize': 15, 'axes.titlesize': 13,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(r['edges'][0], r['edges'][-1])
    ymax = max(rho_p.max(), rho_m.max())
    ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.set_xlabel(r'$R$  (a.u.)')
    ax.set_ylabel(r'$\rho_\pm(R,t)$  (a.u.)')
    ax.grid(True, color='k', alpha=0.08, lw=0.7)

    # rho_- drawn first so the (smaller) rho_+ stays visible on top where they overlap
    bars_m = ax.bar(centers, rho_m[0], width=dR, align='center', color='#d62728',
                    alpha=0.65, edgecolor='none', label=r'$\rho_-$  (lower adiabat)')
    bars_p = ax.bar(centers, rho_p[0], width=dR, align='center', color='#2a78d6',
                    alpha=0.65, edgecolor='none', label=r'$\rho_+$  (upper adiabat)')
    ax.legend(loc='upper left')
    title = ax.set_title("")

    def update(fi):
        for rect, h in zip(bars_m, rho_m[fi]):
            rect.set_height(h)
        for rect, h in zip(bars_p, rho_p[fi]):
            rect.set_height(h)
        title.set_text(f"t = {tframes[fi]:.1f} a.u.    "
                       f"$c_+$ = {r['c_p'][fi]:.3f},  $c_-$ = {r['c_m'][fi]:.3f}    "
                       f"({r['n_traj']} trajectories $\\times$ {r['nbds']} beads)")
        return list(bars_m) + list(bars_p) + [title]

    ani = FuncAnimation(fig, update, frames=tframes.size, blit=False)
    ani.save(out, writer=FFMpegWriter(fps=fps))
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Animate the jump-weighted branch densities rho_+(R,t) and rho_-(R,t).')
    ap.add_argument('--file',   default='data.hdf')
    ap.add_argument('--out',    default='rho_movie.mp4')
    ap.add_argument('--rmin',   type=float, default=R_MIN)
    ap.add_argument('--rmax',   type=float, default=R_MAX)
    ap.add_argument('--bins',   type=int,   default=N_BINS)
    ap.add_argument('--frames', type=int,   default=N_FRAMES, help='target movie frames')
    ap.add_argument('--fps',    type=int,   default=FPS)
    ap.add_argument('--slab',   type=int,   default=5000,
                    help='trajectories read per streaming pass (default 5000)')
    ap.add_argument('--w0-only', action='store_true',
                    help='ignore the jump weights and use W^(0)_PP = 2|Sz0|: the no-jump baseline')
    a = ap.parse_args()

    r = accumulate(a.file, rmin=a.rmin, rmax=a.rmax, bins=a.bins, n_frames=a.frames,
                   slab=a.slab, w0_only=a.w0_only)
    animate(a.out, r, fps=a.fps)

    tot_in  = r['in_p'] + r['in_m']
    tot_all = r['c_p']  + r['c_m']
    lost    = np.where(tot_all > 0, 1.0 - tot_in / np.where(tot_all > 0, tot_all, 1.0), 0.0)
    worst   = int(np.argmax(lost))

    print(f"[movie] n_traj={r['n_traj']} nbds={r['nbds']} frames={r['tframes'].size} "
          f"grid {r['bins']} bins over [{r['rmin']:g}, {r['rmax']:g}] (dR={r['dR']:.4f})"
          + ("  [W^(0) ONLY]" if r['w0_only'] else ""))
    if not r['w0_only']:
        print(f"[movie] total jumps across all trajectories: {r['n_jumps']}")
        if r['n_jumps'] == 0:
            print("[movie] NOTE: no jumps fired, so W_PP is frozen at its t=0 value 2|Sz(0)|.")
    print(f"[movie] c_+ : {r['c_p'][0]:.4f} -> {r['c_p'][-1]:.4f}    "
          f"c_- : {r['c_m'][0]:.4f} -> {r['c_m'][-1]:.4f}")
    print(f"[movie] area check  max|int(rho_+ + rho_-)dR - (c_+ + c_-)| over frames = "
          f"{np.abs(tot_in - tot_all).max():.3e}  (nonzero only from window clipping)")
    if lost.max() > 1e-12:
        print(f"[movie] WARNING: the [{r['rmin']:g}, {r['rmax']:g}] window clips weight; worst frame "
              f"t={r['tframes'][worst]:.1f} loses {100*lost[worst]:.2f}% "
              f"({100*lost.mean():.2f}% on average). Widen with --rmin/--rmax to capture it all.")
    print(f"[movie] wrote {a.out}")


if __name__ == '__main__':
    _main()
