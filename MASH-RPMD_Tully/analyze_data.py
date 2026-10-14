"""
analyze_cm.py -- jump-weighted populations and coherences C_a(t), and their time-dependent-weight
corrections M_a(t), in one streaming pass.  Output is a single 2x2 figure (analyze_cm.png).
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

--- input: the shards directly (or a data.hdf) ------------------------------------------------------

--file takes a directory of workflow.shards files (shards/shard_*.hdf) or a merged data.hdf
(python -m workflow.shards / workflow.consolidate).  Shards have data.hdf's layout and are read
directly, so a sharded run needs NO merge step: the shards are the dataset.

--- how it runs: compiled, parallel, cached ---------------------------------------------------------

The input is cut into UNITS of at most --unit-rows trajectories of one file (default 1000 = one
production shard; a data.hdf or a bigger shard is sliced).  Each unit is reduced to its partial sums
-- the 8 observables, <W_PP>, <W_PP(t) - W_PP>, the jump count -- by one compiled (numba) pass over
trajectories x times (_accumulate), on --ncores worker processes, and the parent adds the partials.

Each partial is CACHED in <file>_partials/<version>/ (e.g. shards_partials/cm_v1/shard_0042_0_1000.npz)
and reused by every later run, so a unit is reduced once, ever:

  * run it WHILE production is going: each run only computes the shards finished since the last one,
    and the final run after production is almost entirely reads of small cached partials;
  * a shard still being written (fewer committed rows than its index range) is analysed afresh every
    run and never cached.  It is opened without an HDF5 lock, so the analysis never blocks the
    production writer; if it is caught mid-append it is skipped with a warning.  Numbers that include
    in-progress shards are provisional -- rerun once production is done;
  * a cached partial is reused only if the source file's size, mtime and committed rows, the row
    range and the analysis version (_VERSION) all match; --refresh recomputes everything, --no-cache
    neither reads nor writes partials.  Bump _VERSION whenever an observable's definition changes.

Run it as ONE multi-core job, not on a login node.  Without --ncores it uses the CPUs of the SLURM
allocation it runs in: $SLURM_CPUS_PER_TASK if set (-c / --cpus-per-task), otherwise -- inside any
SLURM job, e.g. -n N, --exclusive, or an interactive desktop -- the number of CPUs the job confines it
to; outside SLURM, 1.  The first line of output says how many units came from the cache, how many
were computed, and on how many workers.  All cores must be on ONE node.

    #SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24
    #SBATCH --mem=24G                       # a unit is ~ unit_rows x T x nbds x 8 B x 3 arrays
    #SBATCH --time=01:00:00                 # plus the account / env lines of config['slurm']
    python analyze_data.py --file shards --nblocks 100

--- block errors and memory ------------------------------------------------------------------------

The SE is the standard deviation of `nblocks` block means, the blocks being contiguous groups of
whole units (np.array_split over the units, in trajectory order; block mean = block sum / block
count).  With full equal shards and nblocks dividing the number of units (production: 2000 shards,
100 blocks) this is exactly the split into contiguous blocks of trajectories that analyze_cpm.py
makes; otherwise block edges sit on unit edges instead -- an equally valid batch-means SE, since the
trajectories are independent.  The means do not depend on the blocking at all.  If nblocks exceeds
the number of units, there is one block per unit.  Memory: O(nblocks*T) in the parent plus one unit
per worker.

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
    python analyze_data.py --file shards --ncores 16
    python analyze_data.py --file shards --refresh  # after changing an observable (or bump _VERSION)
    python analyze_data.py --file data.hdf --nblocks 20 --tmax 3000
"""

import os
import json
import argparse
import functools

import numpy as np
import h5py

from numba import njit

from workflow import plan_units, map_cached, ncores_source, open_readonly   # cached map-reduce


_NAMES = ('Cp', 'Cm', 'Cx', 'Cy', 'Mp', 'Mm', 'Mx', 'My')


def block_sem_from_means(block_means, n_blocks):
    """SE over trajectories from ALREADY-FORMED block means (same estimator as analyze_cpm.py)."""
    if n_blocks < 2:
        return np.zeros(block_means.shape[1:])
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


# Bump whenever the observables or their weights change: every cached partial is then recomputed.
_VERSION = 'cm_v1'


@njit(cache=True)
def _accumulate(Sz, Sx, Sy, tab, cnt, time, jumps):
    """
    The partial sums of one unit of trajectories, in one compiled pass over trajectories x times
    (no dense weight arrays). Per trajectory k and time t, with m = #{jumps with t_jump <= t}
    (searchsorted-right over the jump times, as expand_weights / expand_weights_t):

        W_PP, W_PC   = /weights row m, columns W_PP, W_PC                         (expand_weights)
        m = 0:  Wt_PP = W_CP^(0) |S_z(t)|,  Wt_PC = W_PC^(0)                      (expand_weights_t)
        m > 0:  dot = Sx_post Sx_prev + Sy_post Sy_prev,   fac = 1 + sgn(Sz_n Sz_prev)
                Wt_PP = dot W_PC_t + |S_z(t)| fac W_PP_t,  Wt_PC = 3/2 dot W_PC_t + fac W_PP_t

    with the same operation order as workflow.consolidate. Without jumps (jumps=False) the weights
    are the Tjump -> infinity ones with c0 = 1: W_PP = 2|S_z(0)|, W_PC = 2, Wt_PP = 2|S_z(t)|,
    Wt_PC = 2 (compute() rescales by c0). h(+-S_z) is the bead mean of heaviside(+-S_z, 0.5);
    S_x, S_y and the |S_z(t)| of the weights are bead means.
    Returns tot (8, T) in _NAMES order, tot_w = sum W_PP, tot_dw = sum (Wt_PP - W_PP), the number of
    jumps, and whether every S_z(0) is <= 0 / >= 0.
    """
    n, T, nb = Sz.shape
    tot = np.zeros((8, T))
    tot_w = np.zeros(T)
    tot_dw = np.zeros(T)
    n_jumps = 0
    le0 = True
    ge0 = True
    for k in range(n):
        c = cnt[k]
        if jumps:
            n_jumps += c - 1                         # row 0 is the t=0 sample, not a jump
        m = 0
        W0 = 0.0
        for t in range(T):
            hp = 0.0; hm = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
            for b in range(nb):
                z = Sz[k, t, b]
                hp += 1.0 if z > 0.0 else (0.5 if z == 0.0 else 0.0)
                hm += 1.0 if z < 0.0 else (0.5 if z == 0.0 else 0.0)
                sx += Sx[k, t, b]; sy += Sy[k, t, b]; sz += z
            hp /= nb; hm /= nb; sx /= nb; sy /= nb; sz /= nb
            asz = np.abs(sz)
            if t == 0:
                le0 = le0 and sz <= 0.0
                ge0 = ge0 and sz >= 0.0
            if jumps:
                while m + 1 < c and tab[k, m + 1, 0] <= time[t]:
                    m += 1
                W_PP = tab[k, m, 2]
                W_PC = tab[k, m, 4]
                if m == 0:
                    Wt_PP = tab[k, 0, 3] * asz
                    Wt_PC = tab[k, 0, 4]
                else:
                    dot = tab[k, m, 13] * tab[k, m, 6] + tab[k, m, 14] * tab[k, m, 7]
                    fac = 1.0 + np.sign(tab[k, m, 1] * tab[k, m, 8])
                    Wt_PP = dot * tab[k, m, 11] + asz * fac * tab[k, m, 9]
                    Wt_PC = 1.5 * dot * tab[k, m, 11] + fac * tab[k, m, 9]
            else:
                if t == 0:
                    W0 = 2.0 * asz
                W_PP = W0
                W_PC = 2.0
                Wt_PP = 2.0 * asz
                Wt_PC = 2.0
            dPP = Wt_PP - W_PP
            dPC = Wt_PC - W_PC
            tot[0, t] += W_PP * hp
            tot[1, t] += W_PP * hm
            tot[2, t] += W_PC * sx
            tot[3, t] += W_PC * sy
            tot[4, t] += dPP * hp
            tot[5, t] += dPP * hm
            tot[6, t] += dPC * sx
            tot[7, t] += dPC * sy
            tot_w[t] += W_PP
            tot_dw[t] += dPP
    return tot, tot_w, tot_dw, n_jumps, le0, ge0


def _unit_partial(path, lo, hi, jumps, time):
    """
    MAP step, run in a worker: the partial sums of rows lo..hi of one file (a shard or a data.hdf).
    workflow.map_cached caches the result per unit, so a unit is reduced once.
    """
    with open_readonly(path) as f:                     # no lock: never blocks a running writer
        if not np.array_equal(f['time'][:], time):
            raise ValueError(f'{path}: time grid differs from the rest of the run')
        if ('weights' in f) != jumps:
            raise ValueError(f'{path}: /weights is {"missing" if jumps else "present"}, unlike the '
                             f'first file -- these files come from different runs')
        Sz = np.ascontiguousarray(f['mapSz'][lo:hi], dtype=np.float64)       # (s, T, nbds)
        Sx = np.ascontiguousarray(f['mapSx'][lo:hi], dtype=np.float64)
        Sy = np.ascontiguousarray(f['mapSy'][lo:hi], dtype=np.float64)
        if jumps:
            tab = np.ascontiguousarray(f['weights'][lo:hi], dtype=np.float64)  # (s, R, 15)
            cnt = np.ascontiguousarray(f['n_weights'][lo:hi], dtype=np.int64)
        else:
            tab = np.zeros((hi - lo, 1, 15)); cnt = np.ones(hi - lo, dtype=np.int64)
    tot, tot_w, tot_dw, n_jumps, le0, ge0 = _accumulate(Sz, Sx, Sy, tab, cnt, time, jumps)
    return dict(n=np.int64(hi - lo), tot=tot, tot_w=tot_w, tot_dw=tot_dw,
                n_jumps=np.int64(n_jumps), sz0_le0=np.bool_(le0), sz0_ge0=np.bool_(ge0))


def default_cache_dir(path):
    """<shards>_partials/<_VERSION> next to a shard directory, <data>_partials/... for a data.hdf."""
    base = os.path.normpath(path)
    if not os.path.isdir(base):
        base = os.path.splitext(base)[0]
    return os.path.join(base + '_partials', _VERSION)


def compute(path, nblocks=None, slab=None, c0=None, ncores=None, unit_rows=1000,
            cache=True, refresh=False, cache_dir=None):
    """
    REDUCE step. `path` (a shard directory or a data.hdf) is split into units of at most unit_rows
    rows of one file (workflow.plan_units); every unit is reduced to its partial sums by
    _unit_partial on `ncores` workers (default: the CPUs of the SLURM allocation, else 1), or read
    back from the cache of an earlier run (workflow.map_cached), and the partials are added up.
    The error-bar blocks are nblocks contiguous groups of whole units. (slab is accepted for
    backward compatibility and ignored: a unit is read in one piece.)
    """
    units = plan_units(path, unit_rows)
    with open_readonly(units[0].file) as f:
        time = f['time'][:]
        cfg  = json.loads(f.attrs['config_json'])
        nbds = int(f.attrs['nbds'])
        jumps = 'weights' in f
        if jumps and f['weights'].shape[2] < 15:
            raise KeyError(f"{units[0].file} has a {f['weights'].shape[2]}-column /weights table, "
                           f"expected 15. The run predates the corrected time-dependent weights; "
                           f"re-run it.")
    if jumps and not cfg.get('full_jump', True):
        raise ValueError('the time-dependent weights are only defined for full_jump=True runs')

    # parallel setup, reported so a silent fall-back to one core cannot go unnoticed
    if ncores is None:
        ncores, why = ncores_source()
    else:
        why = 'given --ncores'
    cache_dir = cache_dir or default_cache_dir(path)
    parts, st = map_cached(functools.partial(_unit_partial, jumps=jumps, time=time), units,
                           cache_dir, _VERSION, ncores=ncores, refresh=refresh, use_cache=cache)
    n_inprog = sum(not u.complete for u in units)
    print(f'[cm] {len(units)} unit(s) of <= {unit_rows} trajectories: {st["cached"]} from cache, '
          f'{st["computed"] + st["in_progress"]} computed on {ncores} worker(s) ({why})'
          + (f'; {n_inprog} unit(s) in shards still being written (never cached)' if n_inprog else '')
          + ('' if cache else '; cache off'), flush=True)
    for note in st['notes']:
        print(f'[cm] WARNING: {note}', flush=True)
    if ncores == 1 and why == 'not inside a SLURM job' and st['computed'] + st['in_progress'] > 1:
        print('[cm] NOTE: computed on ONE core. Pass --ncores N to use N cores.', flush=True)

    used = [p for p in parts if p is not None]
    n_units = len(used)
    if n_units == 0:
        raise RuntimeError('no unit could be read: every file is still being written. Rerun later.')
    n_traj = int(sum(int(p['n']) for p in used))
    T = time.size

    # error bars: nblocks contiguous groups of whole units, block mean = block sum / block count
    nb = n_units if nblocks is None else max(1, min(int(nblocks), n_units))
    if nblocks is not None and int(nblocks) > n_units:
        print(f'[cm] NOTE: only {n_units} units, so {nb} error-bar blocks instead of {nblocks}',
              flush=True)
    tot = np.zeros((len(_NAMES), T))
    blk = np.zeros((len(_NAMES), nb, T))
    block_sizes = np.zeros(nb)
    tot_w = np.zeros(T); tot_dw = np.zeros(T)
    n_jumps = 0
    sz0_le0 = sz0_ge0 = True
    for b, grp in enumerate(np.array_split(np.arange(n_units), nb)):
        for i in grp:
            p = used[i]
            tot += p['tot']
            blk[:, b] += p['tot']
            block_sizes[b] += int(p['n'])
            tot_w += p['tot_w']; tot_dw += p['tot_dw']
            n_jumps += int(p['n_jumps'])
            sz0_le0 &= bool(p['sz0_le0']); sz0_ge0 &= bool(p['sz0_ge0'])

    if not jumps:
        # c0 not stored without Tjump: all S_z(0) on one side -> hemisphere sample (c0 = 1), else 2.
        # For a full-sphere run the chance of all n_traj landing on one side is 2^(1-n_traj).
        if c0 is None:
            c0 = 1.0 if (sz0_le0 or sz0_ge0) else 2.0
        tot *= c0; blk *= c0; tot_w *= c0; tot_dw *= c0

    r = {k: tot[i] / n_traj for i, k in enumerate(_NAMES)}
    r.update({k + '_SE': block_sem_from_means(blk[i] / block_sizes[:, None], nb)
              for i, k in enumerate(_NAMES)})
    W_mean, dW_mean = tot_w / n_traj, tot_dw / n_traj
    r.update(time=time, n_traj=n_traj, nbds=nbds, nblocks=nb, n_jumps=n_jumps,
             n_files=len({u.file for u in units}), n_units=n_units, cache_stats=st,
             Tjump=cfg.get('Tjump'), full_jump=cfg.get('full_jump'), jumps=jumps, c0=c0,
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
    # imported here, not at module level: compute()'s spawned workers import this module, and none
    # of them plots, so they should not each pay matplotlib's import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
    ap.add_argument('--unit-rows', type=int, default=1000,
                    help='trajectories per unit of work / cached partial (default 1000 = one shard of '
                         'the production config); the error-bar blocks are groups of whole units')
    ap.add_argument('--slab',    type=int, default=None, help=argparse.SUPPRESS)   # superseded by units
    ap.add_argument('--no-cache', action='store_true',
                    help='neither read nor write the per-unit partials (default: cache them in '
                         '<file>_partials/' + _VERSION + '/ and reuse them on the next run)')
    ap.add_argument('--refresh', action='store_true',
                    help='recompute (and re-cache) every unit, ignoring existing partials')
    ap.add_argument('--cache-dir', default=None, help='where the partials live (default: see --no-cache)')
    ap.add_argument('--out',     default='analyze_data.png', help='output png')
    ap.add_argument('--ncores',  type=int, default=None,
                    help='worker processes computing units in parallel (default: the CPUs of the '
                         'SLURM allocation, else 1)')
    ap.add_argument('--c0',      type=float, default=None,
                    help='no-jump runs only: initial sampling factor (1 hemisphere, 2 full sphere); '
                         'default: inferred from S_z(0)')
    a = ap.parse_args()
    path = a.file or ('shards' if os.path.isdir('shards') else 'data.hdf')

    try:
        r = compute(path, nblocks=a.nblocks, c0=a.c0, ncores=a.ncores, unit_rows=a.unit_rows,
                    cache=not a.no_cache, refresh=a.refresh, cache_dir=a.cache_dir)
    except RuntimeError as e:
        raise SystemExit(f'[cm] {e}')
    _plot(a.out, r, tmax=a.tmax)

    print(f"[cm] {path}: n_traj={r['n_traj']} from {r['n_files']} file(s) / {r['n_units']} unit(s), "
          f"nbds={r['nbds']}, nblocks={r['nblocks']}")
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
