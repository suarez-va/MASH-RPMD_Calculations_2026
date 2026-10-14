"""
analyze_data.py -- jump-weighted populations and coherences C_a(t), and their time-dependent-weight
corrections M_a(t), in one streaming pass.  Output is a single 2x2 figure (analyze_data.png).
Successor to analyze_cpm.py, whose C_+ / C_- it reproduces exactly.

--- the observables --------------------------------------------------------------------------------

    C_+(t) = < W_PP^(m)  h( S_z(t)) >        M_+(t) = < [W_PP^(m)(t) - W_PP^(m)]  h( S_z(t)) >
    C_-(t) = < W_PP^(m)  h(-S_z(t)) >        M_-(t) = < [W_PP^(m)(t) - W_PP^(m)]  h(-S_z(t)) >
    C_x(t) = < W_PC^(m)  S_x(t) >            M_x(t) = < [W_PC^(m)(t) - W_PC^(m)]  S_x(t) >
    C_y(t) = < W_PC^(m)  S_y(t) >            M_y(t) = < [W_PC^(m)(t) - W_PC^(m)]  S_y(t) >

with h the Heaviside step, <...> an average over trajectories, and m = m(t) = #{ jumps with
t_jump <= t } (every jump up to AND INCLUDING t, matching mash_rpmd, which writes the post-jump
state at t_jump).  Figure: top-left C_+, C_-; top-right M_+, M_-; bottom-left C_x, C_y;
bottom-right M_x, M_y.

--- the two kinds of weight ------------------------------------------------------------------------

W_AB^(m)     the weights that are CONSTANT between jumps: workflow.expand_weights.
W_AB^(m)(t)  the TIME-DEPENDENT weights: workflow.expand_weights_t.  These are never written per
             time step.  At jump m, mash_rpmd freezes into /weights the bead-mean spins right
             before (S^(m-1)(t_m)) and right after (S^(m)(t_m)) the resample, and W^(m-1)(t_m) (times
             jump m's sampling factor); the analysis then finishes the chain on the fly (A = P, C):

                 m = 0:  W^(0)(t)    = c0 (2|S_z(t)|, 2, 2, 3)
                 m > 0:  W_AC^(m)(t) = 3 dot_m W_AC^(m-1)(t_m) +            2 fac_m W_AP^(m-1)(t_m)
                         W_AP^(m)(t) = 2 dot_m W_AC^(m-1)(t_m) + 2 |S_z(t)| fac_m W_AP^(m-1)(t_m)

             with dot_m = S_x^(m-1)(t_m) S_x^(m)(t_m) + S_y^(m-1)(t_m) S_y^(m)(t_m) and
             fac_m = 1 + sgn(S_z^(m-1)(t_m) S_z^(m)(t_m)) both fixed at the jump, so t enters only
             through |S_z(t)| (W_PC(t), W_CC(t) are constant within a segment).  Only the M
             functions use them.  Since W^(0)(0) = W^(0), every M_a(0) is 0 -- up to ~1e-10, because
             W^(0)(0) is built from mapSz.dat and W^(0) from weights.dat, each rounded to 8 digits.

Spins: h(+-S_z) is reduced over beads as np.heaviside(Sz, 0.5).mean(axis=beads) (analyze_cpm.py's
convention); S_x, S_y and the S_z(t) inside expand_weights_t are bead means.  For nbds == 1 (what
this tree runs) every bead convention coincides.

Exact identities, checked and reported on the console (h(x) + h(-x) = 1 bead by bead):

    C_+(t) + C_-(t) = < W_PP^(m) >              M_+(t) + M_-(t) = < W_PP^(m)(t) - W_PP^(m) >

--- input: data.hdf OR the shards directly ---------------------------------------------------------

--file takes either a merged data.hdf (python -m workflow.shards / workflow.consolidate) or a
directory of workflow.shards files (shards/shard_*.hdf).  Shards have data.hdf's layout, so they are
read directly -- no merge step.  Trajectories are ranked by index across the whole run (shards hold
disjoint ascending ranges; within a shard rows are ranked by traj_index, since a resumed shard can
commit out of order), and blocks are contiguous in that rank -- the same split analyze_cpm.py makes
on a sorted data.hdf.  So the shards and the merged file give the same result, error bars included.

--- block errors and memory ------------------------------------------------------------------------

The trajectories are split into `nblocks` contiguous blocks (np.array_split) and the SE is the
standard deviation of the block means.  Only running totals and per-block sums are kept, so peak
memory is O(nblocks*T + slab*T) at any n_traj.

A jump run needs full_jump=True and the 15-column /weights table (older 13-column runs used the
superseded definition with dot and sgn taken at t, and are refused).

--- no-jump runs (Tjump=None) ----------------------------------------------------------------------

mash_rpmd writes no weights.dat without Tjump, so there is no /weights table.  Such a run is analysed
as the Tjump -> infinity limit: no jump ever fires, so every t is in segment 0 and

    W^(0)    = c0 (2|S_z(0)|, 2, 2, 3)          W^(0)(t) = c0 (2|S_z(t)|, 2, 2, 3)

with S_z the bead mean, exactly what init_jump_weights / expand_weights_t would have produced.  Hence
M_x = M_y = 0 identically.  The initial sampling factor c0 is not stored in such a run, so it is
inferred from S_z(0): every trajectory on one side of the equator -> a hemisphere sample, c0 = 1
(Tully's init_map_spin(0)); otherwise the full sphere, c0 = 2.  For a full-sphere run the chance of
all n_traj landing on one side is 2^(1-n_traj).  --c0 overrides the inference.

Usage (from the run dir, PYTHONPATH=<RP_MASH root>):
    python analyze_data.py                          # reads shards/ if present, else data.hdf
    python analyze_data.py --file data.hdf --nblocks 20 --tmax 3000
"""

import os
import glob
import json
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from workflow import expand_weights, expand_weights_t   # per-jump table -> dense W^(m), W^(m)(t)


_NAMES = ('Cp', 'Cm', 'Cx', 'Cy', 'Mp', 'Mm', 'Mx', 'My')


def block_sem_from_means(block_means, n_blocks):
    """SE over trajectories from ALREADY-FORMED block means (same estimator as analyze_cpm.py)."""
    if n_blocks < 2:
        return np.zeros(block_means.shape[1:])
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def _sources(path):
    """
    [(file path, traj_index)] in trajectory order: one entry for a data.hdf, one per shard for a
    shard directory (sorted by smallest index; shards hold disjoint ascending ranges).
    """
    if os.path.isdir(path):
        paths = glob.glob(os.path.join(path, 'shard_*.hdf'))
        if not paths:
            raise FileNotFoundError(f'no shard_*.hdf in {path}')
    else:
        paths = [path]
    out = []
    for p in paths:
        with h5py.File(p, 'r') as f:
            ti = f['traj_index'][:]
        if ti.size:
            out.append((p, ti))
    if not out:
        raise FileNotFoundError(f'no committed trajectories in {path}')
    return sorted(out, key=lambda s: s[1].min())


def _infer_c0(sources):
    """Initial sampling factor of a no-jump run: 1 if every S_z(0) is on one side, else 2."""
    sz0 = []
    for p, _ in sources:
        with h5py.File(p, 'r') as f:
            sz0.append(f['mapSz'][:, 0, :].mean(axis=-1))
    sz0 = np.concatenate(sz0)
    return 1.0 if (np.all(sz0 <= 0.0) or np.all(sz0 >= 0.0)) else 2.0


def _w0(Sz, c0):
    """No-jump weights, Tjump -> infinity: (W_PP^(0), W_PC^(0)) and (W_PP^(0)(t), W_PC^(0)(t))."""
    W  = {'W_PP': c0 * 2.0 * np.abs(Sz[:, :1]), 'W_PC': np.full(Sz.shape, c0 * 2.0)}
    Wt = {'W_PP': c0 * 2.0 * np.abs(Sz),        'W_PC': W['W_PC']}
    W['W_PP'] = np.broadcast_to(W['W_PP'], Sz.shape)
    return W, Wt


def compute(path, nblocks=None, slab=5000, c0=None):
    sources = _sources(path)
    n_traj = sum(ti.size for _, ti in sources)

    with h5py.File(sources[0][0], 'r') as f:
        time = f['time'][:]
        cfg  = json.loads(f.attrs['config_json'])
        nbds = int(f.attrs['nbds'])
        jumps = 'weights' in f
        if jumps and f['weights'].shape[2] < 15:
            raise KeyError(f"{sources[0][0]} has a {f['weights'].shape[2]}-column /weights table, "
                           f"expected 15. The run predates the corrected time-dependent weights; "
                           f"re-run it.")
    if jumps and not cfg.get('full_jump', True):
        raise ValueError('the time-dependent weights are only defined for full_jump=True runs')
    if not jumps and c0 is None:
        c0 = _infer_c0(sources)

    # contiguous blocks over the trajectory RANK, identical to np.array_split(..., axis=0)
    nb = n_traj if nblocks is None else max(1, min(int(nblocks), n_traj))
    block_of = np.concatenate([np.full(len(b), i, dtype=int)
                               for i, b in enumerate(np.array_split(np.arange(n_traj), nb))])
    block_sizes = np.bincount(block_of, minlength=nb).astype(float)

    T = time.size
    tot = {k: np.zeros(T) for k in _NAMES}
    blk = {k: np.zeros((nb, T)) for k in _NAMES}
    tot_w = np.zeros(T); tot_dw = np.zeros(T)      # <W_PP>, <W_PP(t) - W_PP>: identity checks
    n_jumps = 0

    offset = 0
    for p, ti in sources:
        rank = offset + np.argsort(np.argsort(ti, kind='stable'))    # global rank of each row
        with h5py.File(p, 'r') as f:
            if not np.array_equal(f['time'][:], time):
                raise ValueError(f'{p}: time grid differs from {sources[0][0]}')
            if ('weights' in f) != jumps:
                raise ValueError(f'{p}: /weights is {"missing" if jumps else "present"}, unlike '
                                 f'{sources[0][0]} -- these files come from different runs')
            for lo in range(0, ti.size, slab):
                hi  = min(lo + slab, ti.size)
                sel = slice(lo, hi)

                Sz = f['mapSz'][sel]                                  # (s, T, nbds)
                Sx = f['mapSx'][sel].mean(axis=2)                     # (s, T) bead means
                Sy = f['mapSy'][sel].mean(axis=2)
                hp = np.heaviside( Sz, 0.5).mean(axis=2)              # (s, T) upper adiabat
                hm = np.heaviside(-Sz, 0.5).mean(axis=2)              # (s, T) lower adiabat

                if jumps:
                    W  = expand_weights(f, time=time, sel=sel)                      # W^(m)
                    Wt = expand_weights_t(f, sel=sel, S=(Sx, Sy, Sz.mean(axis=2)))  # W^(m)(t)
                    n_jumps += int(f['n_weights'][sel].sum() - (hi - lo))  # row 0 is t=0, not a jump
                else:
                    W, Wt = _w0(Sz.mean(axis=2), c0)                                # Tjump -> inf
                dPP = Wt['W_PP'] - W['W_PP']
                dPC = Wt['W_PC'] - W['W_PC']

                vals = {'Cp': W['W_PP'] * hp, 'Cm': W['W_PP'] * hm,
                        'Cx': W['W_PC'] * Sx, 'Cy': W['W_PC'] * Sy,
                        'Mp': dPP * hp,       'Mm': dPP * hm,
                        'Mx': dPC * Sx,       'My': dPC * Sy}
                b = block_of[rank[lo:hi]]
                for k, v in vals.items():
                    tot[k] += v.sum(axis=0)
                    np.add.at(blk[k], b, v)
                tot_w  += W['W_PP'].sum(axis=0)
                tot_dw += dPP.sum(axis=0)
        offset += ti.size

    r = {k: tot[k] / n_traj for k in _NAMES}
    r.update({k + '_SE': block_sem_from_means(blk[k] / block_sizes[:, None], nb) for k in _NAMES})
    W_mean, dW_mean = tot_w / n_traj, tot_dw / n_traj
    r.update(time=time, n_traj=n_traj, nbds=nbds, nblocks=nb, n_jumps=n_jumps,
             n_files=len(sources), Tjump=cfg.get('Tjump'), full_jump=cfg.get('full_jump'),
             jumps=jumps, c0=c0,
             W_mean=W_mean,
             identity_C=float(np.max(np.abs(r['Cp'] + r['Cm'] - W_mean))),
             identity_M=float(np.max(np.abs(r['Mp'] + r['Mm'] - dW_mean))))
    return r


def _panel(ax, t, r, keys, labels, colors, ylabel):
    ax.axhline(0.0, color='0.7', lw=0.8, ls='--')
    for k, lab, c in zip(keys, labels, colors):
        ax.plot(t, r[k], color=c, lw=1.6, label=lab)
        ax.fill_between(t, r[k] - r[k + '_SE'], r[k] + r[k + '_SE'], color=c, alpha=0.3, lw=0)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', frameon=False, fontsize=9)


def _plot(out, r, tmax=None):
    t = r['time']
    if tmax is not None:                         # truncate for plotting only
        msk = t <= tmax
        t = t[msk]
        r = {k: (v[msk] if isinstance(v, np.ndarray) and v.shape == msk.shape else v)
             for k, v in r.items()}

    fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.0))
    _panel(axs[0, 0], t, r, ('Cp', 'Cm'),
           (r'$C_+(t)=\langle W_{PP}^{(m)}\,h(S_z(t))\rangle$',
            r'$C_-(t)=\langle W_{PP}^{(m)}\,h(-S_z(t))\rangle$'), ('C3', 'C0'), r'$C_\pm(t)$')
    _panel(axs[0, 1], t, r, ('Mp', 'Mm'),
           (r'$M_+(t)=\langle [W_{PP}^{(m)}(t)-W_{PP}^{(m)}]\,h(S_z(t))\rangle$',
            r'$M_-(t)=\langle [W_{PP}^{(m)}(t)-W_{PP}^{(m)}]\,h(-S_z(t))\rangle$'), ('C3', 'C0'),
           r'$M_\pm(t)$')
    _panel(axs[1, 0], t, r, ('Cx', 'Cy'),
           (r'$C_x(t)=\langle W_{PC}^{(m)}\,S_x(t)\rangle$',
            r'$C_y(t)=\langle W_{PC}^{(m)}\,S_y(t)\rangle$'), ('C2', 'C1'), r'$C_{x,y}(t)$')
    _panel(axs[1, 1], t, r, ('Mx', 'My'),
           (r'$M_x(t)=\langle [W_{PC}^{(m)}(t)-W_{PC}^{(m)}]\,S_x(t)\rangle$',
            r'$M_y(t)=\langle [W_{PC}^{(m)}(t)-W_{PC}^{(m)}]\,S_y(t)\rangle$'), ('C2', 'C1'),
           r'$M_{x,y}(t)$')

    fig.suptitle(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}, "
                 f"Tjump={r['Tjump']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Jump-weighted C_a(t) and time-dependent-weight corrections M_a(t) from RP-MASH '
                    'shards or a data.hdf.')
    ap.add_argument('--file',    default=None,
                    help='a shard directory or a data.hdf (default: shards/ if it exists, else data.hdf)')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--slab',    type=int, default=5000,
                    help='trajectories read per streaming pass (default 5000; lower it if RAM is tight)')
    ap.add_argument('--out',     default='analyze_data.png', help='output png')
    ap.add_argument('--c0',      type=float, default=None,
                    help='no-jump runs only: initial sampling factor (1 hemisphere, 2 full sphere); '
                         'default: inferred from S_z(0)')
    a = ap.parse_args()
    path = a.file or ('shards' if os.path.isdir('shards') else 'data.hdf')

    r = compute(path, nblocks=a.nblocks, slab=a.slab, c0=a.c0)
    _plot(a.out, r, tmax=a.tmax)

    print(f"[cm] {path}: n_traj={r['n_traj']} from {r['n_files']} file(s), nbds={r['nbds']}, "
          f"nblocks={r['nblocks']}")
    if r['jumps']:
        print(f"[cm] Tjump={r['Tjump']} full_jump={r['full_jump']}  "
              f"total jumps across all trajectories: {r['n_jumps']}")
    else:
        print(f"[cm] no /weights table (Tjump={r['Tjump']}): analysed as Tjump -> infinity, "
              f"W^(0) only, c0={r['c0']:g}" + (' (from --c0)' if a.c0 is not None else
                                                ' (inferred from S_z(0))'))
    for k, name in (('Cp', 'C_+'), ('Cm', 'C_-'), ('Cx', 'C_x'), ('Cy', 'C_y'),
                    ('Mp', 'M_+'), ('Mm', 'M_-'), ('Mx', 'M_x'), ('My', 'M_y')):
        print(f"[cm] {name}(0)={r[k][0]: .6e} +/- {r[k + '_SE'][0]:.3e}   "
              f"{name}(T)={r[k][-1]: .6e} +/- {r[k + '_SE'][-1]:.3e}")
    print(f"[cm] identities: max|C_+ + C_- - <W_PP>| = {r['identity_C']:.3e}   "
          f"max|M_+ + M_- - <W_PP(t) - W_PP>| = {r['identity_M']:.3e}")
    if max(r['identity_C'], r['identity_M']) > 1e-10 * max(1.0, np.abs(r['W_mean']).max()):
        print('[cm] WARNING: an identity is violated beyond round-off. Something is wrong.')
    print(f"[cm] wrote {a.out}")


if __name__ == '__main__':
    _main()
