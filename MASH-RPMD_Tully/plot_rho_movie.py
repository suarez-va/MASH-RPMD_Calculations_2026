"""
plot_rho_movie.py -- animate the quantum-jump-weighted BRANCH densities

    rho_+(R,t) = < W_PP^(m)  delta(R - R(t))  h( S_z(t) ) >      upper adiabat, BLUE
    rho_-(R,t) = < W_PP^(m)  delta(R - R(t))  h(-S_z(t) ) >      lower adiabat, RED

into an .mp4.  This is analyze_data.py's C_+(t) / C_-(t) with the trajectory average resolved in R
instead of collapsed to a scalar -- same estimator, same weights, one level more detail.  The
normalization is chosen so that relation is exact:

    integral rho_+(R,t) dR = C_+(t),   integral rho_-(R,t) dR = C_-(t),
    integral (rho_+ + rho_-) dR = <W_PP^(m)>(t)

up to the weight that falls outside the histogram window, which is tracked and reported.

--- weighting ---------------------------------------------------------------------------------

Every BEAD at every frame carries its own weight,

    weight of bead k of trajectory i at time t  =  W_PP^(m)(i,t) * h(+/- S_z(i,t,k))

with W_PP^(m) the CONSTANT-between-jumps weight (workflow.expand_weights) at m(t) = #{jumps with
t_jump <= t}, i.e. every jump up to AND INCLUDING t -- matching mash_rpmd, which writes the post-jump
state at t_jump.  h is the Heaviside step with h(0) = 1/2; summing over beads and dividing by nbds
reproduces exactly the bead-mean convention of analyze_data.py, so the two agree for any nbds.

In MASH sgn(S_z) selects the active adiabat, so rho_+ is the upper surface and rho_- the lower.
traj.py starts every trajectory on the lower adiabat via init_map_spin(0), so rho_+ is identically
zero until the packet reaches the crossing.

A run made without Tjump has no /weights table and is analysed, as in analyze_data.py, as the
Tjump -> infinity limit W_PP = c0 2|S_z(0)| (|bead-mean S_z(0)|), with c0 inferred from S_z(0) (all on
one side of the equator -> hemisphere sample, c0 = 1, which is Tully's init_map_spin(0); otherwise
the full sphere, c0 = 2), or given with --c0.  --w0-only ignores the jumps of a jump run the same way
(W_PP^(m) -> W_PP^(0), the stored row 0, which already contains c0).

--- window ------------------------------------------------------------------------------------

The histogram grid is FIXED (--rmin, --rmax, --bins) so frames are comparable.  Bins are
np.histogram's: [e_i, e_i+1), the last one closed.  Beads outside the window are dropped; the script
tracks how much weight that costs and reports the worst frame, so clipping is never silent.

--- input and how it runs (as analyze_data.py) -------------------------------------------------

--file takes a directory of workflow.shards files (shards/shard_*.hdf, read directly -- no merge) or
a merged data.hdf.  The input is cut into UNITS of at most --unit-rows trajectories of one file;
each unit is reduced to its histogram partial sums at the movie frames by one compiled (numba) pass
(_accumulate), on --ncores worker processes, and the parent adds them up.  Each partial is CACHED in
<file>_partials/<version>/ and reused by later runs (a new frame selection or window gets its own
version, so it is never mixed up); a shard still being written is analysed afresh and never cached.
--refresh recomputes everything, --no-cache neither reads nor writes partials.  Without --ncores it
uses the CPUs of the SLURM allocation, else 1.  Rendering the movie is serial (ffmpeg) and takes the
same time whatever the number of trajectories.

    python plot_rho_movie.py                                   # shards/ if present, else data.hdf
    python plot_rho_movie.py --file shards --ncores 20 --unit-rows 10000
    python plot_rho_movie.py --rmin -15 --rmax 15 --bins 149 --frames 400 --out rho_movie.mp4
"""

import os
import json
import argparse
import functools

import numpy as np

from numba import njit

from workflow import plan_units, map_cached, ncores_source, open_readonly   # cached map-reduce

# ------------------------------------------------------------------ fixed histogram grid + movie knobs
R_MIN    = -12.0      # a.u.; histogram window is [R_MIN, R_MAX]
R_MAX    =  12.0
N_BINS   =  99        # bins across that window
N_FRAMES = 300        # target movie frames (evenly subsampled; uses all if fewer time points exist)
FPS      =  30

# Bump whenever the estimator changes: every cached partial is then recomputed.
_VERSION = 'rho_v1'


def frame_indices(T, n_frames):
    """The time rows shown in the movie: n_frames evenly spaced rows (all of them if T is smaller)."""
    return np.unique(np.linspace(0, T - 1, min(n_frames, T)).astype(int))


@njit(cache=True)
def _accumulate(R, Sz, tab, cnt, tframes, edges):
    """
    Histogram partial sums of one unit at the movie frames, in one compiled pass. R and Sz are
    (s, F, nbds) at the frames. Per trajectory k and frame f, m = #{jumps with t_jump <= t_f} and
    W_PP = /weights row m, column W_PP (expand_weights); each bead b adds W_PP h(+-S_z) to the bin of
    its R, with np.histogram's bin assignment (edges[i] <= R < edges[i+1], last bin closed).
    A no-jump run comes in with a synthetic row 0 holding 2|S_z(0)| and cnt = 1 (compute() rescales
    by c0), --w0-only with the real table and cnt = 1. Returns cnt_p, cnt_m (F, bins) in-window
    sums, tot_p, tot_m (F,) sums over every bead, the number of jumps, and whether every S_z(0) is
    <= 0 / >= 0 (bead mean at the first frame, which is t = 0).
    """
    n, F, nb = Sz.shape
    bins = edges.shape[0] - 1
    lo_e = edges[0]; hi_e = edges[bins]
    norm = bins / (hi_e - lo_e)
    cnt_p = np.zeros((F, bins)); cnt_m = np.zeros((F, bins))
    tot_p = np.zeros(F); tot_m = np.zeros(F)
    n_jumps = 0
    le0 = True; ge0 = True
    for k in range(n):
        c = cnt[k]
        n_jumps += c - 1                             # row 0 is the t=0 sample, not a jump
        sz0 = 0.0
        for b in range(nb):
            sz0 += Sz[k, 0, b]
        sz0 /= nb
        le0 = le0 and sz0 <= 0.0
        ge0 = ge0 and sz0 >= 0.0
        m = 0
        for f in range(F):
            while m + 1 < c and tab[k, m + 1, 0] <= tframes[f]:
                m += 1
            W = tab[k, m, 2]
            for b in range(nb):
                z = Sz[k, f, b]
                hp = 1.0 if z > 0.0 else (0.5 if z == 0.0 else 0.0)
                hm = 1.0 if z < 0.0 else (0.5 if z == 0.0 else 0.0)
                wp = W * hp; wm = W * hm
                tot_p[f] += wp; tot_m[f] += wm
                x = R[k, f, b]
                if x < lo_e or x > hi_e:
                    continue
                i = int((x - lo_e) * norm)           # np.histogram: estimate, then fix at the edges
                if i >= bins:
                    i = bins - 1
                if x < edges[i]:
                    i -= 1
                elif i + 1 < bins and x >= edges[i + 1]:
                    i += 1
                cnt_p[f, i] += wp; cnt_m[f, i] += wm
    return cnt_p, cnt_m, tot_p, tot_m, n_jumps, le0, ge0


def _unit_partial(path, lo, hi, jumps, w0_only, time, fidx, edges):
    """MAP step, run in a worker: the partial sums of rows lo..hi of one file."""
    with open_readonly(path) as f:                     # no lock: never blocks a running writer
        if not np.array_equal(f['time'][:], time):
            raise ValueError(f'{path}: time grid differs from the rest of the run')
        if ('weights' in f) != jumps:
            raise ValueError(f'{path}: /weights is {"missing" if jumps else "present"}, unlike the '
                             f'first file -- these files come from different runs')
        R  = np.ascontiguousarray(f['nucR'][lo:hi, :, :, 0][:, fidx], dtype=np.float64)   # (s,F,nbds)
        Sz = np.ascontiguousarray(f['mapSz'][lo:hi][:, fidx], dtype=np.float64)
        if jumps:
            tab = np.ascontiguousarray(f['weights'][lo:hi], dtype=np.float64)
            cnt = np.ascontiguousarray(f['n_weights'][lo:hi], dtype=np.int64)
    if not jumps:
        # Tjump -> infinity: synthetic row 0 with W_PP = 2|S_z(0)| (c0 = 1; compute() rescales)
        tab = np.zeros((hi - lo, 1, 15))
        tab[:, 0, 2] = 2.0 * np.abs(Sz[:, 0].mean(axis=1))
        cnt = np.ones(hi - lo, dtype=np.int64)
    elif w0_only:
        cnt = np.ones(hi - lo, dtype=np.int64)         # never leave segment 0
    cnt_p, cnt_m, tot_p, tot_m, n_jumps, le0, ge0 = _accumulate(R, Sz, tab, cnt, time[fidx], edges)
    return dict(n=np.int64(hi - lo), cnt_p=cnt_p, cnt_m=cnt_m, tot_p=tot_p, tot_m=tot_m,
                n_jumps=np.int64(n_jumps), sz0_le0=np.bool_(le0), sz0_ge0=np.bool_(ge0))


def _version(rmin, rmax, bins, n_frames, w0_only):
    """_VERSION plus everything else the partials depend on, so variants never share a cache."""
    return f"{_VERSION}_{'w0' if w0_only else 'w'}_{rmin:.12g}_{rmax:.12g}_{bins}_{n_frames}"


def default_cache_dir(path, version):
    base = os.path.normpath(path)
    if not os.path.isdir(base):
        base = os.path.splitext(base)[0]
    return os.path.join(base + '_partials', version)


def accumulate(path, rmin=R_MIN, rmax=R_MAX, bins=N_BINS, n_frames=N_FRAMES, slab=None,
               w0_only=False, c0=None, ncores=None, unit_rows=1000, cache=True, refresh=False,
               cache_dir=None):
    """
    REDUCE step: rho_+ and rho_- on the fixed grid at the movie frames, from a shard directory or a
    data.hdf. Returns a dict with edges/centers/dR, tframes, rho_p/rho_m of shape (n_frames, bins),
    the per-frame total weight in each branch (= C_+(t), C_-(t)) and the in-window part. (slab is
    accepted for backward compatibility and ignored: a unit is read in one piece.)
    """
    edges   = np.linspace(rmin, rmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dR      = edges[1] - edges[0]

    units = plan_units(path, unit_rows)
    with open_readonly(units[0].file) as f:
        time  = f['time'][:]
        cfg   = json.loads(f.attrs['config_json'])
        nbds  = int(f.attrs['nbds'])
        jumps = 'weights' in f
    fidx    = frame_indices(time.size, n_frames)
    tframes = time[fidx]

    if ncores is None:
        ncores, why = ncores_source()
    else:
        why = 'given --ncores'
    version = _version(rmin, rmax, bins, n_frames, w0_only)
    cache_dir = cache_dir or default_cache_dir(path, version)
    fn = functools.partial(_unit_partial, jumps=jumps, w0_only=w0_only, time=time, fidx=fidx,
                           edges=edges)
    parts, st = map_cached(fn, units, cache_dir, version, ncores=ncores, refresh=refresh,
                           use_cache=cache, tag='[movie]')
    n_inprog = sum(not u.complete for u in units)
    print(f'[movie] {len(units)} unit(s) of <= {unit_rows} trajectories: {st["cached"]} from cache, '
          f'{st["computed"] + st["in_progress"]} computed on {ncores} worker(s) ({why})'
          + (f'; {n_inprog} unit(s) in shards still being written (never cached)' if n_inprog else '')
          + ('' if cache else '; cache off'), flush=True)
    for note in st['notes']:
        print(f'[movie] WARNING: {note}', flush=True)

    used = [p for p in parts if p is not None]
    if not used:
        raise RuntimeError('no unit could be read: every file is still being written. Rerun later.')
    n_traj = int(sum(int(p['n']) for p in used))
    cnt_p = sum(p['cnt_p'] for p in used); cnt_m = sum(p['cnt_m'] for p in used)
    tot_p = sum(p['tot_p'] for p in used); tot_m = sum(p['tot_m'] for p in used)
    n_jumps = int(sum(int(p['n_jumps']) for p in used))
    sz0_le0 = all(bool(p['sz0_le0']) for p in used); sz0_ge0 = all(bool(p['sz0_ge0']) for p in used)

    if not jumps:
        if c0 is None:
            c0 = 1.0 if (sz0_le0 or sz0_ge0) else 2.0
        cnt_p = cnt_p * c0; cnt_m = cnt_m * c0; tot_p = tot_p * c0; tot_m = tot_m * c0

    norm  = n_traj * nbds * dR
    rho_p = cnt_p / norm
    rho_m = cnt_m / norm

    # C_+(t), C_-(t): the full averages, and the part that actually landed inside the window
    c_p, c_m = tot_p / (n_traj * nbds), tot_m / (n_traj * nbds)
    in_p, in_m = cnt_p.sum(axis=1) / (n_traj * nbds), cnt_m.sum(axis=1) / (n_traj * nbds)

    return dict(edges=edges, centers=centers, dR=dR, tframes=tframes,
                rho_p=rho_p, rho_m=rho_m, c_p=c_p, c_m=c_m, in_p=in_p, in_m=in_m,
                n_traj=n_traj, nbds=nbds, n_jumps=n_jumps, jumps=jumps, c0=c0, w0_only=w0_only,
                Tjump=cfg.get('Tjump'), rmin=rmin, rmax=rmax, bins=bins)


def animate(out, r, fps=FPS):
    # imported here, not at module level: accumulate()'s spawned workers import this module, and
    # none of them plots, so they should not each pay matplotlib's import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

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
        description='Animate the jump-weighted branch densities rho_+(R,t) and rho_-(R,t) from '
                    'RP-MASH shards or a data.hdf.')
    ap.add_argument('--file',   default=None,
                    help='a shard directory or a data.hdf (default: shards/ if it exists, else data.hdf)')
    ap.add_argument('--out',    default='rho_movie.mp4')
    ap.add_argument('--rmin',   type=float, default=R_MIN)
    ap.add_argument('--rmax',   type=float, default=R_MAX)
    ap.add_argument('--bins',   type=int,   default=N_BINS)
    ap.add_argument('--frames', type=int,   default=N_FRAMES, help='target movie frames')
    ap.add_argument('--fps',    type=int,   default=FPS)
    ap.add_argument('--w0-only', action='store_true',
                    help='ignore the jumps: W_PP^(m) -> W_PP^(0) (the no-jump baseline)')
    ap.add_argument('--c0',     type=float, default=None,
                    help='no-jump runs only: initial sampling factor (1 hemisphere, 2 full sphere); '
                         'default: inferred from S_z(0)')
    ap.add_argument('--unit-rows', type=int, default=1000,
                    help='trajectories per unit of work / cached partial (default 1000)')
    ap.add_argument('--slab',   type=int, default=None, help=argparse.SUPPRESS)   # superseded by units
    ap.add_argument('--no-cache', action='store_true',
                    help='neither read nor write the per-unit partials')
    ap.add_argument('--refresh', action='store_true',
                    help='recompute (and re-cache) every unit, ignoring existing partials')
    ap.add_argument('--cache-dir', default=None, help='where the partials live')
    ap.add_argument('--ncores', type=int, default=None,
                    help='worker processes computing units in parallel (default: the CPUs of the '
                         'SLURM allocation, else 1)')
    a = ap.parse_args()
    path = a.file or ('shards' if os.path.isdir('shards') else 'data.hdf')

    try:
        r = accumulate(path, rmin=a.rmin, rmax=a.rmax, bins=a.bins, n_frames=a.frames,
                       w0_only=a.w0_only, c0=a.c0, ncores=a.ncores, unit_rows=a.unit_rows,
                       cache=not a.no_cache, refresh=a.refresh, cache_dir=a.cache_dir)
    except RuntimeError as e:
        raise SystemExit(f'[movie] {e}')
    animate(a.out, r, fps=a.fps)

    tot_in  = r['in_p'] + r['in_m']
    tot_all = r['c_p']  + r['c_m']
    lost    = np.where(tot_all > 0, 1.0 - tot_in / np.where(tot_all > 0, tot_all, 1.0), 0.0)
    worst   = int(np.argmax(lost))

    print(f"[movie] {path}: n_traj={r['n_traj']} nbds={r['nbds']} frames={r['tframes'].size} "
          f"grid {r['bins']} bins over [{r['rmin']:g}, {r['rmax']:g}] (dR={r['dR']:.4f})"
          + ("  [W^(0) ONLY]" if r['w0_only'] else ""))
    if r['jumps']:
        print(f"[movie] Tjump={r['Tjump']}  total jumps across all trajectories: {r['n_jumps']}")
    else:
        print(f"[movie] no /weights table (Tjump={r['Tjump']}): analysed as Tjump -> infinity, "
              f"W_PP = c0 2|S_z(0)|, c0={r['c0']:g}" + (' (from --c0)' if a.c0 is not None else
                                                         ' (inferred from S_z(0))'))
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
