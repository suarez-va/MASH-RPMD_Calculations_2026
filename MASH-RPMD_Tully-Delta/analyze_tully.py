"""
analyze_tully.py -- channel-resolved scattering populations C_c(t) for the Tully single avoided
crossing, and their time-dependent-weight corrections M_c(t), in one streaming pass.  Output is a
single two-panel figure (analyze_tully.png).  Built exactly like analyze_data.py (same input, weights,
compiled kernel, cached parallel map-reduce, block errors); only the observables differ.

--- the observables --------------------------------------------------------------------------------

    C_Rg(t) = < W_PP^(m)  h(-R(t))  h(-S_z(t)) >    M_Rg(t) = < [W_PP^(m)(t) - W_PP^(m)]  h(-R(t))  h(-S_z(t)) >
    C_Tg(t) = < W_PP^(m)  h( R(t))  h(-S_z(t)) >    M_Tg(t) = < [W_PP^(m)(t) - W_PP^(m)]  h( R(t))  h(-S_z(t)) >
    C_Te(t) = < W_PP^(m)  h( R(t))  h( S_z(t)) >    M_Te(t) = < [W_PP^(m)(t) - W_PP^(m)]  h( R(t))  h( S_z(t)) >

Reflected (R < 0) on the ground adiabat, transmitted (R > 0) on the ground adiabat, transmitted on
the excited adiabat.  h is the Heaviside step with h(0) = 1/2, <...> an average over trajectories,
and m = m(t) = #{ jumps with t_jump <= t } (every jump up to AND INCLUDING t).  Figure: left C_Rg,
C_Tg, C_Te; right M_Rg, M_Tg, M_Te.

The weights are analyze_data.py's: W_PP^(m) constant between jumps (workflow.expand_weights) and
W_PP^(m)(t) time-dependent (workflow.expand_weights_t), finished on the fly from the 15-column
/weights table -- see analyze_data.py for the chain.  Every M_c(0) is 0 up to ~1e-10 (rounding of
weights.dat vs mapSz.dat).

Beads: the channel factor is taken bead by bead and then averaged, (1/nbds) sum_b h(+-R_b) h(+-S_z,b),
as in plot_rho_movie.py (each bead deposits W h(+-S_z) at its own R), so e.g. C_Tg(t) is the integral
of the movie's rho_-(R,t) over R > 0.  The |S_z(t)| inside W_PP^(m)(t) is the bead mean, as in
analyze_data.py.  For nbds == 1 (what this tree runs) every bead convention coincides.

Exact identities, checked and reported on the console (h(R) + h(-R) = 1 bead by bead), with C_-, M_-
analyze_data.py's:

    C_Rg(t) + C_Tg(t) = C_-(t)                  M_Rg(t) + M_Tg(t) = M_-(t)

--- input, parallelism, cache, error bars, no-jump runs ---------------------------------------------

All exactly as in analyze_data.py: --file takes a shard directory or a data.hdf; the input is cut
into units of at most --unit-rows trajectories, each reduced by one compiled pass (_accumulate) on
--ncores workers (default: the CPUs of the SLURM allocation, else 1) and cached in
<file>_partials/tully_v1/; the SE is over --nblocks contiguous groups of whole units; a run without
/weights (an OLD run made without Tjump) is the Tjump -> infinity limit with c0 inferred from
S_z(0) (--c0 overrides).  Current runs always write /weights, so a jump_index=[] run has one row per
trajectory, with c0 stored, and goes through the ordinary path.

Usage (from the run dir, PYTHONPATH=<RP_MASH root>):
    python analyze_tully.py                          # reads shards/ if present, else data.hdf
    python analyze_tully.py --file shards --ncores 16
    python analyze_tully.py --file shards --refresh  # after changing an observable (or bump _VERSION)
    python analyze_tully.py --file data.hdf --nblocks 20 --tmax 3000
"""

import os
import json
import argparse
import functools

import numpy as np

from numba import njit

from workflow import plan_units, map_cached, ncores_source, open_readonly   # cached map-reduce


# the six plotted observables, then C_-, M_- for the identity checks
_NAMES = ('CRg', 'CTg', 'CTe', 'MRg', 'MTg', 'MTe', 'Cm', 'Mm')


def block_sem_from_means(block_means, n_blocks):
    """SE over trajectories from ALREADY-FORMED block means (same estimator as analyze_cpm.py)."""
    if n_blocks < 2:
        return np.zeros(block_means.shape[1:])
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def jump_label(cfg):
    """'jump_index=[...]' for current runs, 'Tjump=...' for old ones: titles and console lines."""
    if 'jump_index' in cfg:
        ji = list(cfg['jump_index'])
        if len(ji) > 6:
            return f'jump_index=[{ji[0]}, {ji[1]}, ..., {ji[-1]}] ({len(ji)} jumps)'
        return f'jump_index={ji}'
    return f"Tjump={cfg.get('Tjump')}"


# Bump whenever the observables or their weights change: every cached partial is then recomputed.
_VERSION = 'tully_v1'


@njit(cache=True)
def _accumulate(R, Sz, tab, cnt, time, jumps):
    """
    The partial sums of one unit of trajectories, in one compiled pass over trajectories x times.
    R and Sz are (s, T, nbds). Per trajectory k and time t, with m = #{jumps with t_jump <= t}:

        W_PP  = /weights row m, column W_PP                                       (expand_weights)
        m = 0:  Wt_PP = W_CP^(0) |S_z(t)|                                         (expand_weights_t)
        m > 0:  Wt_PP = dot W_PC_t + |S_z(t)| fac W_PP_t,
                dot = Sx_post Sx_prev + Sy_post Sy_prev,   fac = 1 + sgn(Sz_n Sz_prev)

    exactly as analyze_data._accumulate. Without jumps (jumps=False) W_PP = 2|S_z(0)|,
    Wt_PP = 2|S_z(t)| (c0 = 1; compute() rescales by c0). The channel factors are bead means of
    h(+-R_b) h(+-S_z,b); |S_z(t)| in the weights is the bead mean.
    Returns tot (8, T) in _NAMES order, the number of jumps, and whether every S_z(0) is <= 0 / >= 0.
    """
    n, T, nb = Sz.shape
    tot = np.zeros((8, T))
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
            rg = 0.0; tg = 0.0; te = 0.0; hm = 0.0; sz = 0.0
            for b in range(nb):
                z = Sz[k, t, b]
                x = R[k, t, b]
                hzp = 1.0 if z > 0.0 else (0.5 if z == 0.0 else 0.0)
                hzm = 1.0 if z < 0.0 else (0.5 if z == 0.0 else 0.0)
                hrp = 1.0 if x > 0.0 else (0.5 if x == 0.0 else 0.0)
                hrm = 1.0 if x < 0.0 else (0.5 if x == 0.0 else 0.0)
                rg += hrm * hzm
                tg += hrp * hzm
                te += hrp * hzp
                hm += hzm
                sz += z
            rg /= nb; tg /= nb; te /= nb; hm /= nb; sz /= nb
            asz = np.abs(sz)
            if t == 0:
                le0 = le0 and sz <= 0.0
                ge0 = ge0 and sz >= 0.0
            if jumps:
                while m + 1 < c and tab[k, m + 1, 0] <= time[t]:
                    m += 1
                W_PP = tab[k, m, 2]
                if m == 0:
                    Wt_PP = tab[k, 0, 3] * asz
                else:
                    dot = tab[k, m, 13] * tab[k, m, 6] + tab[k, m, 14] * tab[k, m, 7]
                    fac = 1.0 + np.sign(tab[k, m, 1] * tab[k, m, 8])
                    Wt_PP = dot * tab[k, m, 11] + asz * fac * tab[k, m, 9]
            else:
                if t == 0:
                    W0 = 2.0 * asz
                W_PP = W0
                Wt_PP = 2.0 * asz
            dPP = Wt_PP - W_PP
            tot[0, t] += W_PP * rg
            tot[1, t] += W_PP * tg
            tot[2, t] += W_PP * te
            tot[3, t] += dPP * rg
            tot[4, t] += dPP * tg
            tot[5, t] += dPP * te
            tot[6, t] += W_PP * hm
            tot[7, t] += dPP * hm
    return tot, n_jumps, le0, ge0


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
        R  = np.ascontiguousarray(f['nucR'][lo:hi, :, :, 0], dtype=np.float64)  # (s, T, nbds)
        Sz = np.ascontiguousarray(f['mapSz'][lo:hi], dtype=np.float64)          # (s, T, nbds)
        if jumps:
            tab = np.ascontiguousarray(f['weights'][lo:hi], dtype=np.float64)  # (s, R, 15)
            cnt = np.ascontiguousarray(f['n_weights'][lo:hi], dtype=np.int64)
        else:
            tab = np.zeros((hi - lo, 1, 15)); cnt = np.ones(hi - lo, dtype=np.int64)
    tot, n_jumps, le0, ge0 = _accumulate(R, Sz, tab, cnt, time, jumps)
    return dict(n=np.int64(hi - lo), tot=tot, n_jumps=np.int64(n_jumps),
                sz0_le0=np.bool_(le0), sz0_ge0=np.bool_(ge0))


def default_cache_dir(path):
    """<shards>_partials/<_VERSION> next to a shard directory, <data>_partials/... for a data.hdf."""
    base = os.path.normpath(path)
    if not os.path.isdir(base):
        base = os.path.splitext(base)[0]
    return os.path.join(base + '_partials', _VERSION)


def compute(path, nblocks=None, slab=None, c0=None, ncores=None, unit_rows=1000,
            cache=True, refresh=False, cache_dir=None):
    """
    REDUCE step, as analyze_data.compute: `path` is split into units of at most unit_rows rows of
    one file, every unit is reduced by _unit_partial on `ncores` workers or read back from the cache,
    and the partials are added up. The error-bar blocks are nblocks contiguous groups of whole
    units. (slab is accepted for backward compatibility and ignored.)
    """
    units = plan_units(path, unit_rows)
    with open_readonly(units[0].file) as f:
        time = f['time'][:]
        cfg  = json.loads(f.attrs['config_json'])
        nbds = int(f.attrs['nbds'])
        jumps = 'weights' in f
        if f['nucR'].shape[2] != 1:
            raise ValueError(f"{units[0].file}: nnuc={f['nucR'].shape[2]}, but the Tully channels "
                             f"need a single nuclear coordinate")
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
                           cache_dir, _VERSION, ncores=ncores, refresh=refresh, use_cache=cache,
                           tag='[tully]')
    n_inprog = sum(not u.complete for u in units)
    print(f'[tully] {len(units)} unit(s) of <= {unit_rows} trajectories: {st["cached"]} from cache, '
          f'{st["computed"] + st["in_progress"]} computed on {ncores} worker(s) ({why})'
          + (f'; {n_inprog} unit(s) in shards still being written (never cached)' if n_inprog else '')
          + ('' if cache else '; cache off'), flush=True)
    for note in st['notes']:
        print(f'[tully] WARNING: {note}', flush=True)
    if ncores == 1 and why == 'not inside a SLURM job' and st['computed'] + st['in_progress'] > 1:
        print('[tully] NOTE: computed on ONE core. Pass --ncores N to use N cores.', flush=True)

    used = [p for p in parts if p is not None]
    n_units = len(used)
    if n_units == 0:
        raise RuntimeError('no unit could be read: every file is still being written. Rerun later.')
    n_traj = int(sum(int(p['n']) for p in used))
    T = time.size

    # error bars: nblocks contiguous groups of whole units, block mean = block sum / block count
    nb = n_units if nblocks is None else max(1, min(int(nblocks), n_units))
    if nblocks is not None and int(nblocks) > n_units:
        print(f'[tully] NOTE: only {n_units} units, so {nb} error-bar blocks instead of {nblocks}',
              flush=True)
    tot = np.zeros((len(_NAMES), T))
    blk = np.zeros((len(_NAMES), nb, T))
    block_sizes = np.zeros(nb)
    n_jumps = 0
    sz0_le0 = sz0_ge0 = True
    for b, grp in enumerate(np.array_split(np.arange(n_units), nb)):
        for i in grp:
            p = used[i]
            tot += p['tot']
            blk[:, b] += p['tot']
            block_sizes[b] += int(p['n'])
            n_jumps += int(p['n_jumps'])
            sz0_le0 &= bool(p['sz0_le0']); sz0_ge0 &= bool(p['sz0_ge0'])

    if not jumps:
        # c0 not stored without Tjump: all S_z(0) on one side -> hemisphere sample (c0 = 1), else 2.
        if c0 is None:
            c0 = 1.0 if (sz0_le0 or sz0_ge0) else 2.0
        tot *= c0; blk *= c0

    r = {k: tot[i] / n_traj for i, k in enumerate(_NAMES)}
    r.update({k + '_SE': block_sem_from_means(blk[i] / block_sizes[:, None], nb)
              for i, k in enumerate(_NAMES)})
    r.update(time=time, n_traj=n_traj, nbds=nbds, nblocks=nb, n_jumps=n_jumps,
             n_files=len({u.file for u in units}), n_units=n_units, cache_stats=st,
             Tjump=cfg.get('Tjump'), jump_label=jump_label(cfg), full_jump=cfg.get('full_jump'),
             jumps=jumps, c0=c0,
             identity_C=float(np.max(np.abs(r['CRg'] + r['CTg'] - r['Cm']))),
             identity_M=float(np.max(np.abs(r['MRg'] + r['MTg'] - r['Mm']))))
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

    colors = ('C0', 'C2', 'C3')
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.0))
    _panel(axs[0], t, r, ('CRg', 'CTg', 'CTe'),
           (r'$C_{Rg}(t)=\langle W_{PP}^{(m)}\,h(-R(t))\,h(-S_z(t))\rangle$',
            r'$C_{Tg}(t)=\langle W_{PP}^{(m)}\,h(R(t))\,h(-S_z(t))\rangle$',
            r'$C_{Te}(t)=\langle W_{PP}^{(m)}\,h(R(t))\,h(S_z(t))\rangle$'), colors, r'$C(t)$')
    _panel(axs[1], t, r, ('MRg', 'MTg', 'MTe'),
           (r'$M_{Rg}(t)=\langle [W_{PP}^{(m)}(t)-W_{PP}^{(m)}]\,h(-R(t))\,h(-S_z(t))\rangle$',
            r'$M_{Tg}(t)=\langle [W_{PP}^{(m)}(t)-W_{PP}^{(m)}]\,h(R(t))\,h(-S_z(t))\rangle$',
            r'$M_{Te}(t)=\langle [W_{PP}^{(m)}(t)-W_{PP}^{(m)}]\,h(R(t))\,h(S_z(t))\rangle$'),
           colors, r'$M(t)$')

    fig.suptitle(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}, "
                 f"{r['jump_label']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Channel-resolved Tully populations C_Rg, C_Tg, C_Te and their time-dependent-'
                    'weight corrections M from RP-MASH shards or a data.hdf.')
    ap.add_argument('--file',    default=None,
                    help='a shard directory or a data.hdf (default: shards/ if it exists, else data.hdf)')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--unit-rows', type=int, default=1000,
                    help='trajectories per unit of work / cached partial (default 1000); the '
                         'error-bar blocks are groups of whole units')
    ap.add_argument('--slab',    type=int, default=None, help=argparse.SUPPRESS)   # superseded by units
    ap.add_argument('--no-cache', action='store_true',
                    help='neither read nor write the per-unit partials (default: cache them in '
                         '<file>_partials/' + _VERSION + '/ and reuse them on the next run)')
    ap.add_argument('--refresh', action='store_true',
                    help='recompute (and re-cache) every unit, ignoring existing partials')
    ap.add_argument('--cache-dir', default=None, help='where the partials live (default: see --no-cache)')
    ap.add_argument('--out',     default='analyze_tully.png', help='output png')
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
        raise SystemExit(f'[tully] {e}')
    _plot(a.out, r, tmax=a.tmax)

    print(f"[tully] {path}: n_traj={r['n_traj']} from {r['n_files']} file(s) / {r['n_units']} "
          f"unit(s), nbds={r['nbds']}, nblocks={r['nblocks']}")
    if r['jumps']:
        print(f"[tully] {r['jump_label']} full_jump={r['full_jump']}  "
              f"total jumps across all trajectories: {r['n_jumps']}")
    else:
        print(f"[tully] no /weights table (old run, {r['jump_label']}): Tjump -> infinity, "
              f"W^(0) only, c0={r['c0']:g}" + (' (from --c0)' if a.c0 is not None else
                                                ' (inferred from S_z(0))'))
    for k, name in (('CRg', 'C_Rg'), ('CTg', 'C_Tg'), ('CTe', 'C_Te'), ('Cm', 'C_-'),
                    ('MRg', 'M_Rg'), ('MTg', 'M_Tg'), ('MTe', 'M_Te'), ('Mm', 'M_-')):
        print(f"[tully] {name:4s}(0)={r[k][0]: .6e} +/- {r[k + '_SE'][0]:.3e}   "
              f"{name:4s}(T)={r[k][-1]: .6e} +/- {r[k + '_SE'][-1]:.3e}")
    print(f"[tully] identities: max|C_Rg + C_Tg - C_-| = {r['identity_C']:.3e}   "
          f"max|M_Rg + M_Tg - M_-| = {r['identity_M']:.3e}")
    if max(r['identity_C'], r['identity_M']) > 1e-10 * max(1.0, np.abs(r['Cm']).max()):
        print('[tully] WARNING: an identity is violated beyond round-off. Something is wrong.')
    print(f"[tully] wrote {a.out}")


if __name__ == '__main__':
    _main()
