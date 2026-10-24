"""
analyze_decoherence_rate.py -- analyze_p2.py with a DECOHERENCE CUT on the coherence factor: the
quantum-jump-weighted population P_2(t), its time-dependent-weight correction M_2(t), the rate
integrand K_12(t) with a BLOCK ANALYSIS of the rate, and the rate correction M_12(t), in one
compiled, parallel, cached pass over the shards (or a data.hdf).  Output is a single 2x2 figure
(analyze_decoherence_rate.png): top-left P_2, top-right M_2, bottom-left K_12 with its line fits,
bottom-right M_12.  Everything is exactly analyze_p2.py's except Ct.

--- the observables --------------------------------------------------------------------------------

 P_2(t)  =  1/4 < P0 W_PP^(m) Pt >  -  1/4 < C0 W_CP^(m) Pt >  +  1/4 < P0 W_PC^(m) Ct >  -  1/4 < C0 W_CC^(m) Ct >
 P_2'(t) =  the same with the time-dependent weights W_AB^(m)(t)
 M_2(t)  =  P_2'(t) - P_2(t)

    P0 = 1 + K(0) sgn(Sz(0))    C0 = D(0) Sx(0)
    Pt = 1 - K(t) sgn(Sz(t))    Ct = h(Rc - Rbar(t)) D(t) Sx(t)

    Rc = sqrt(2/(lbd*kvec)) (ngap/beta - epsil/2)

h is the Heaviside step (h(0) = 1/2) and Rbar the bead-average (centroid) nucR.  Rc is where
kappa(R) = sqrt(lbd*kvec/2) R + epsil/2 reaches ngap/beta, i.e. where the adiabatic gap 2*kappa is
2*ngap/beta: once a trajectory's centroid has crossed past Rc its coherence is taken to have
decohered, and its Ct terms contribute 0.  ngap is --ngap (default 4); beta and epsil are the same
values P2eq uses (the config's, or --beta / --epsil).  --ngap very large recovers analyze_p2.py.

with t = 0 the first row of `time` (a prepared initial condition, not a sliding time origin) and
<...> an average over trajectories.  The electronic-mixing functions are

    kappa(R) = sqrt(lbd*kvec/2)*R + epsil/2,   Delta = delta,   norm = sqrt(kappa^2 + Delta^2)
    K(R) = kappa/norm,    D(R) = Delta/norm

(Delta is a constant in ET_with_bath.)  NOTE the relation to analyze_rate.py's a_b_functions, which
returns a = -delta/norm and b = kappa/norm: K == b but D == -a.

The rate integrand and its correction:

    P2eq      = 1 / (1 + exp(-beta*epsil))
    K_12(t)   = -P2eq ln| 1 - P_2(t) /P2eq |        K_12'(t) likewise from P_2'(t)
    M_12(t)   = K_12'(t) - K_12(t) = -P2eq ln| (P2eq - P_2'(t)) / (P2eq - P_2(t)) |

The rate is the slope of a least-squares line fit of K_12 over [fitmin, fitmax].

The prefactor is 1/4 (not the 1/2 of the earlier version of this script) because mash_rpmd now folds
the initial sampling factor into the weights: init_jump_weights stores c0 (2|Sz(0)|, 2, 2, 3), with
c0 = 2 for the full-sphere init_map_spin(None) these runs use.  1/4 x c0 = 2 is the old 1/2.

--- the weights ------------------------------------------------------------------------------------

m = m(t) = #{ jumps with t_jump <= t } (every jump up to AND INCLUDING t, matching mash_rpmd, which
writes the post-jump state at t_jump).

W_AB^(m)     constant between jumps: the /weights row of segment m (as workflow.expand_weights).
W_AB^(m)(t)  time-dependent (as workflow.expand_weights_t), finished here from what mash_rpmd froze
             at jump m (A = P, C):

                 m = 0:  W^(0)(t)    = c0 (2|Sz(t)|, 2, 2, 3)
                 m > 0:  W_AP^(m)(t) =     dot_m W_AC^(m-1)(t_m) + |Sz(t)| fac_m W_AP^(m-1)(t_m)
                         W_AC^(m)(t) = 3/2 dot_m W_AC^(m-1)(t_m) +          fac_m W_AP^(m-1)(t_m)

             dot_m = Sx^(m-1)(t_m) Sx^(m)(t_m) + Sy^(m-1)(t_m) Sy^(m)(t_m),
             fac_m = 1 + sgn(Sz^(m-1)(t_m) Sz^(m)(t_m)), both fixed at the jump, so t enters only
             through |Sz(t)|.  A full_jump=False run (hemisphere jumps) uses the same rule with
             dot_m = 0, just as mash_rpmd does for its constant weights:
                         W_AP^(m)(t) = |Sz(t)| fac_m W_AP^(m-1)(t_m),  W_AC^(m)(t) = fac_m W_AP^(m-1)(t_m)
             A full_jump=False run made before mash_rpmd logged S^(m-1)(t_m) and W^(m-1)(t_m) for
             it has NaN there: fine without jumps (or with --w0-only), an error with them.  Since W^(0)(0) = W^(0), M_2(0) = 0 -- up to ~1e-10, because W^(0)(0)
             is built from mapSz.dat and W^(0) from weights.dat, each rounded to 8 digits.

Beads: K, D at the centroid Rbar; Sx the bead mean; sgn(Sz) the bead mean of sign(Sz); the |Sz(t)|
inside the weights the |bead mean|.  For nbds == 1 every convention coincides.

--w0-only ignores every jump (W^(m) -> W^(0), W^(m)(t) -> W^(0)(t)): the no-jump baseline curve.

Current runs always write /weights (a jump_index=[] run has one row per trajectory, with c0
stored).  An OLD run made without Tjump has no /weights table; it is analysed as the Tjump -> infinity limit with
W^(0) = c0 (2|Sz(0)|, 2, 2, 3).  c0 is not stored there, so it is inferred from Sz(0): every
trajectory on one side of the equator -> hemisphere sample, c0 = 1; otherwise the full sphere, c0 = 2
(chance of a wrong call 2^(1-n_traj)).  --c0 overrides the inference.

--- input: the shards directly (or a data.hdf) ------------------------------------------------------

--file takes a directory of workflow.shards files (shards/shard_*.hdf) or a merged data.hdf.  Shards
have data.hdf's layout and are read directly, so a sharded run needs NO merge step.

--- how it runs: compiled, parallel, cached ---------------------------------------------------------

The input is cut into UNITS of at most --unit-rows trajectories of one file (default 1000 = one
production shard; a data.hdf or a bigger shard is sliced).  Each unit is reduced to its partial sums
-- the four terms of P_2 and of P_2', and the jump count -- by one compiled (numba) pass over
trajectories x times (_accumulate), on --ncores worker processes, and the parent adds the partials.

Each partial is CACHED in <file>_partials/<version>/ and reused by every later run, so a unit is
reduced once, ever:

  * run it WHILE production is going: each run only computes the shards finished since the last one;
  * a shard still being written is analysed afresh every run and never cached.  It is opened without
    an HDF5 lock, so the analysis never blocks the production writer; if it is caught mid-append it
    is skipped with a warning.  Numbers that include in-progress shards are provisional;
  * a cached partial is reused only if the source file's size, mtime and committed rows, the row
    range and the version all match.  The version holds _VERSION, the potential parameters
    (kvec, epsil, lbd, delta), Rc (so --ngap, --beta and --epsil each get their own partials) and
    --w0-only.  --refresh recomputes everything, --no-cache neither
    reads nor writes partials.  Bump _VERSION whenever an observable's definition changes.

Run it as ONE multi-core job, not on a login node.  Without --ncores it uses the CPUs of the SLURM
allocation it runs in ($SLURM_CPUS_PER_TASK, else the CPUs the job is confined to); outside SLURM, 1.
The first line of output says how many units came from the cache, how many were computed, and on how
many workers.  All cores must be on ONE node.

    #SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24
    #SBATCH --mem=24G
    #SBATCH --time=01:00:00                 # plus the account / env lines of config['slurm']
    python ../../../../analyze_decoherence_rate.py --file shards --nblocks 100

--- block analysis ---------------------------------------------------------------------------------

The blocks are `nblocks` contiguous groups of whole units (np.array_split over the units, in
trajectory order; block mean = block sum / block count).  With full equal shards and nblocks
dividing the number of units this is exactly a split into contiguous blocks of trajectories;
otherwise block edges sit on unit edges -- an equally valid batch-means SE.  If nblocks exceeds the
number of units, there is one block per unit.

  P_2, M_2  SE = std of the block means / sqrt(nblocks)  (both are linear in the data)
  K_12      SE = SE(P_2) / |u|,  u = 1 - P_2/P2eq       (error propagated through the logarithm)
  M_12      SE = std over blocks of  (P_2'_b - P_2')/u' - (P_2_b - P_2)/u,  / sqrt(nblocks)
            (the same linearization, applied per block so the strong P_2 / P_2' correlation is kept)

Each block gets its own K_12 and line fit, giving one slope per block:

    m_bar = mean(slopes),   sigma_m = std(slopes, ddof=1) / sqrt(nblocks)

CAVEAT: K_12 is only meaningful while u > 0.  A block built from few trajectories can have
P_2 > P2eq, driving u through zero and K_12 through the log singularity; those blocks produce
meaningless slopes.  The script warns when it happens -- if you see that warning, use fewer, larger
blocks.

Usage (from a scan-point dir, e.g. Low-Friction/Epsilon-Scan/MASH-Classical/epsil_3.0,
PYTHONPATH=<RP_MASH root>):
    python ../../../../analyze_decoherence_rate.py                # shards/ if present, else data.hdf
    python ../../../../analyze_decoherence_rate.py --file shards --ncores 16 --ngap 4
    python ../../../../analyze_decoherence_rate.py --ngap 2 --out decoh_ngap2.png
    python ../../../../analyze_decoherence_rate.py --refresh      # after changing an observable
"""

import os
import json
import argparse
import functools

import numpy as np
from scipy.stats import linregress

from numba import njit

from workflow import plan_units, map_cached, ncores_source, open_readonly   # cached map-reduce


# Bump whenever the observables or their weights change: every cached partial is then recomputed.
_VERSION = 'dr_v1'

_TERMS = ('+P0 W_PP Pt', '-C0 W_CP Pt', '+P0 W_PC Ct', '-C0 W_CC Ct')


def jump_label(cfg):
    """'jump_index=[...]' for current runs, 'Tjump=...' for old ones: titles and console lines."""
    if 'jump_index' in cfg:
        ji = list(cfg['jump_index'])
        if len(ji) > 6:
            return f'jump_index=[{ji[0]}, {ji[1]}, ..., {ji[-1]}] ({len(ji)} jumps)'
        return f'jump_index={ji}'
    return f"Tjump={cfg.get('Tjump')}"


def block_sem_from_means(block_means, n_blocks):
    """SE over trajectories from ALREADY-FORMED block means (same estimator as analyze_rate.py)."""
    if n_blocks < 2:
        return np.zeros(block_means.shape[1:])
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


@njit(cache=True)
def _accumulate(R, Sz, Sx, tab, cnt, time, prm, full):
    """
    The partial sums of one unit of trajectories, in one compiled pass over trajectories x times
    (no dense weight arrays). Per trajectory k and time t, with m = #{jumps with t_jump <= t}
    (searchsorted-right over the jump times, as expand_weights / expand_weights_t):

        W   = /weights row m, columns W_PP, W_CP, W_PC, W_CC                       (expand_weights)
        m = 0:  Wt = (W_CP^(0) |Sz(t)|, W_CP^(0), W_PC^(0), W_CC^(0))              (expand_weights_t)
        m > 0:  dot = Sx_post Sx_prev + Sy_post Sy_prev,   fac = 1 + sgn(Sz_n Sz_prev)
                Wt_AP = dot W_AC_t + |Sz(t)| fac W_AP_t,   Wt_AC = 3/2 dot W_AC_t + fac W_AP_t
                (full = False, i.e. a full_jump=False run: dot = 0)

    in the operation order of workflow.consolidate._full_jump_recursion. A no-jump run comes in
    with a synthetic row 0 and cnt = 1, --w0-only with the real table and cnt = 1: either way m
    stays 0. prm = (kvec, epsil, lbd, delta, Rc); Ct carries h(Rc - Rbar), h(0) = 1/2. Returns
    tot (8, T) -- the four terms of P_2 (constant weights), then of P_2' (time-dependent weights),
    in _TERMS order, each with its 1/4 -- past (T,), the number of trajectories with Rbar > Rc at
    each time, the number of jumps, and whether every Sz(0) is <= 0 / >= 0.
    """
    n, T, nb = Sz.shape
    kvec = prm[0]; epsil = prm[1]; lbd = prm[2]; delta = prm[3]; Rc = prm[4]
    slope = np.sqrt(lbd * kvec / 2.0)
    tot = np.zeros((8, T))
    past = np.zeros(T)
    n_jumps = 0
    le0 = True
    ge0 = True
    for k in range(n):
        c = cnt[k]
        n_jumps += c - 1                             # row 0 is the t=0 sample, not a jump
        m = 0
        P0 = 0.0; C0 = 0.0
        for t in range(T):
            rb = 0.0; sx = 0.0; sz = 0.0; sg = 0.0
            for b in range(nb):
                z = Sz[k, t, b]
                rb += R[k, t, b]; sx += Sx[k, t, b]; sz += z; sg += np.sign(z)
            rb /= nb; sx /= nb; sz /= nb; sg /= nb
            asz = np.abs(sz)
            kappa = slope * rb + epsil / 2.0
            norm = np.sqrt(kappa**2 + delta**2)
            K = kappa / norm
            D = delta / norm
            if t == 0:
                P0 = 1.0 + K * sg
                C0 = D * sx
                le0 = le0 and sz <= 0.0
                ge0 = ge0 and sz >= 0.0
            Pt = 1.0 - K * sg
            hR = 1.0 if rb < Rc else (0.5 if rb == Rc else 0.0)   # h(Rc - Rbar): decoherence cut
            if rb > Rc:
                past[t] += 1.0
            Ct = hR * D * sx

            while m + 1 < c and tab[k, m + 1, 0] <= time[t]:
                m += 1
            W_PP = tab[k, m, 2]; W_CP = tab[k, m, 3]; W_PC = tab[k, m, 4]; W_CC = tab[k, m, 5]
            if m == 0:
                Wt_PP = tab[k, 0, 3] * asz
                Wt_CP = tab[k, 0, 3]
                Wt_PC = tab[k, 0, 4]
                Wt_CC = tab[k, 0, 5]
            else:
                dot = (tab[k, m, 13] * tab[k, m, 6] + tab[k, m, 14] * tab[k, m, 7]) if full else 0.0
                fac = 1.0 + np.sign(tab[k, m, 1] * tab[k, m, 8])
                Wt_PP = dot * tab[k, m, 11] + asz * fac * tab[k, m, 9]
                Wt_CP = dot * tab[k, m, 12] + asz * fac * tab[k, m, 10]
                Wt_PC = 1.5 * dot * tab[k, m, 11] + fac * tab[k, m, 9]
                Wt_CC = 1.5 * dot * tab[k, m, 12] + fac * tab[k, m, 10]

            tot[0, t] += 0.25 * P0 * W_PP * Pt
            tot[1, t] -= 0.25 * C0 * W_CP * Pt
            tot[2, t] += 0.25 * P0 * W_PC * Ct
            tot[3, t] -= 0.25 * C0 * W_CC * Ct
            tot[4, t] += 0.25 * P0 * Wt_PP * Pt
            tot[5, t] -= 0.25 * C0 * Wt_CP * Pt
            tot[6, t] += 0.25 * P0 * Wt_PC * Ct
            tot[7, t] -= 0.25 * C0 * Wt_CC * Ct
    return tot, past, n_jumps, le0, ge0


def _unit_partial(path, lo, hi, jumps, w0_only, time, prm, full=True):
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
        R  = np.ascontiguousarray(f['nucR'][lo:hi, :, :, 0], dtype=np.float64)   # (s, T, nbds)
        Sz = np.ascontiguousarray(f['mapSz'][lo:hi], dtype=np.float64)
        Sx = np.ascontiguousarray(f['mapSx'][lo:hi], dtype=np.float64)
        if jumps:
            tab = np.ascontiguousarray(f['weights'][lo:hi], dtype=np.float64)    # (s, R, 15)
            cnt = np.ascontiguousarray(f['n_weights'][lo:hi], dtype=np.int64)
    if not jumps:
        # Tjump -> infinity: a synthetic row 0 holding W^(0) with c0 = 1 (compute() rescales by c0)
        tab = np.zeros((hi - lo, 1, 15))
        tab[:, 0, 2] = 2.0 * np.abs(Sz[:, 0].mean(axis=1))
        tab[:, 0, 3] = 2.0; tab[:, 0, 4] = 2.0; tab[:, 0, 5] = 3.0
        cnt = np.ones(hi - lo, dtype=np.int64)
    elif w0_only:
        cnt = np.ones(hi - lo, dtype=np.int64)         # never leave segment 0
    elif not full and any(np.isnan(tab[k, 1:cnt[k], 6:13]).any() for k in range(hi - lo)):
        raise ValueError(f'{path}: /weights has NaN in S^(m-1)(t_m) / W^(m-1)(t_m) of a jump row. '
                         f'This full_jump=False run was made before RP_MASH wrote its time-dependent '
                         f'weights; re-run it (or pass --w0-only).')
    tot, past, n_jumps, le0, ge0 = _accumulate(R, Sz, Sx, tab, cnt, time, prm, full)
    return dict(n=np.int64(hi - lo), tot=tot, past=past, n_jumps=np.int64(n_jumps),
                sz0_le0=np.bool_(le0), sz0_ge0=np.bool_(ge0))


def _version(prm, w0_only, full=True):
    """_VERSION plus everything else the partials depend on, so variants never share a cache."""
    tag = '_'.join(f'{x:.12g}' for x in prm)
    return f"{_VERSION}_{'w0' if w0_only else 'w'}_{tag}" + ('' if full else '_fj0')


def default_cache_dir(path, version):
    """<shards>_partials/<version> next to a shard directory, <data>_partials/... for a data.hdf."""
    base = os.path.normpath(path)
    if not os.path.isdir(base):
        base = os.path.splitext(base)[0]
    return os.path.join(base + '_partials', version)


def k12_from_p2(P2, P2eq, P2_SE=None):
    """K12 = -P2eq*ln|1 - P2/P2eq| and, if given an error, its propagated SE = sigma_P2/|u|."""
    u = 1.0 - P2 / P2eq
    with np.errstate(divide='ignore', invalid='ignore'):
        K12 = -P2eq * np.log(np.abs(u))
        SE  = None if P2_SE is None else P2_SE / np.abs(u)
    if SE is not None:
        SE = np.where(np.isfinite(SE), SE, 0.0)     # |u|=0 -> divide by zero -> error 0
    return K12, SE, u


def line_fit(time, K12, fitmin, fitmax):
    """Least-squares m*t+b of K12 over [fitmin, fitmax], finite points only. NaNs if too few."""
    mask = (time >= fitmin) & (time <= fitmax) & np.isfinite(K12)
    if int(mask.sum()) < 2:
        return float('nan'), float('nan'), float('nan'), int(mask.sum())
    fit = linregress(time[mask], K12[mask])
    return float(fit.slope), float(fit.intercept), float(fit.rvalue), int(mask.sum())


def compute(path, nblocks=None, fitmin=5.0, fitmax=15.0, beta_ovr=None, epsil_ovr=None,
            slab=None, w0_only=False, c0=None, ncores=None, unit_rows=1000,
            cache=True, refresh=False, cache_dir=None, ngap=4.0):
    """
    REDUCE step. `path` (a shard directory or a data.hdf) is split into units of at most unit_rows
    rows of one file (workflow.plan_units); every unit is reduced to its partial sums by
    _unit_partial on `ncores` workers (default: the CPUs of the SLURM allocation, else 1), or read
    back from the cache of an earlier run (workflow.map_cached), and the partials are added up.
    The blocks are nblocks contiguous groups of whole units. (slab is accepted for backward
    compatibility and ignored: a unit is read in one piece.)
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
    full = bool(cfg.get('full_jump', True))      # False: hemisphere jumps, dot = 0 in W^(m)(t)

    kvec  = float(cfg['kvec']); epsil = float(cfg['epsil'])
    lbd   = float(cfg['lbd']);  delta = float(cfg['delta'])
    beta  = float(cfg['beta'])
    if beta_ovr  is not None: beta  = float(beta_ovr)
    if epsil_ovr is not None: epsil = float(epsil_ovr)
    # decoherence cut: kappa(Rc) = ngap/beta, with the same beta / epsil as P2eq
    Rc = float(np.sqrt(2.0 / (lbd * kvec)) * (float(ngap) / beta - epsil / 2.0))
    prm = np.array([kvec, epsil, lbd, delta, Rc], dtype=np.float64)

    # parallel setup, reported so a silent fall-back to one core cannot go unnoticed
    if ncores is None:
        ncores, why = ncores_source()
    else:
        why = 'given --ncores'
    version = _version(prm, w0_only, full)
    cache_dir = cache_dir or default_cache_dir(path, version)
    fn = functools.partial(_unit_partial, jumps=jumps, w0_only=w0_only, time=time, prm=prm,
                           full=full)
    parts, st = map_cached(fn, units, cache_dir, version, ncores=ncores, refresh=refresh,
                           use_cache=cache, tag='[dr]')
    n_inprog = sum(not u.complete for u in units)
    print(f'[dr] {len(units)} unit(s) of <= {unit_rows} trajectories: {st["cached"]} from cache, '
          f'{st["computed"] + st["in_progress"]} computed on {ncores} worker(s) ({why})'
          + (f'; {n_inprog} unit(s) in shards still being written (never cached)' if n_inprog else '')
          + ('' if cache else '; cache off'), flush=True)
    for note in st['notes']:
        print(f'[dr] WARNING: {note}', flush=True)
    if ncores == 1 and why == 'not inside a SLURM job' and st['computed'] + st['in_progress'] > 1:
        print('[dr] NOTE: computed on ONE core. Pass --ncores N to use N cores.', flush=True)

    used = [p for p in parts if p is not None]
    n_units = len(used)
    if n_units == 0:
        raise RuntimeError('no unit could be read: every file is still being written. Rerun later.')
    n_traj = int(sum(int(p['n']) for p in used))
    T = time.size

    # blocks: nblocks contiguous groups of whole units, block mean = block sum / block count
    nb = n_units if nblocks is None else max(1, min(int(nblocks), n_units))
    if nblocks is not None and int(nblocks) > n_units:
        print(f'[dr] NOTE: only {n_units} units, so {nb} blocks instead of {nblocks}', flush=True)
    tot = np.zeros((8, T))
    past = np.zeros(T)
    blk = np.zeros((nb, 8, T))
    block_sizes = np.zeros(nb)
    n_jumps = 0
    sz0_le0 = sz0_ge0 = True
    for b, grp in enumerate(np.array_split(np.arange(n_units), nb)):
        for i in grp:
            p = used[i]
            tot += p['tot']
            past += p['past']
            blk[b] += p['tot']
            block_sizes[b] += int(p['n'])
            n_jumps += int(p['n_jumps'])
            sz0_le0 &= bool(p['sz0_le0']); sz0_ge0 &= bool(p['sz0_ge0'])

    if not jumps:
        # c0 not stored without Tjump: all Sz(0) on one side -> hemisphere sample (c0 = 1), else 2
        if c0 is None:
            c0 = 1.0 if (sz0_le0 or sz0_ge0) else 2.0
        tot *= c0; blk *= c0

    terms   = tot[:4] / n_traj                         # the four terms of P_2
    terms_t = tot[4:] / n_traj                         # ... and of P_2'
    P2, P2t = terms.sum(axis=0), terms_t.sum(axis=0)
    M2 = P2t - P2
    blk_P2  = blk[:, :4].sum(axis=1) / block_sizes[:, None]       # (nb, T) block means
    blk_P2t = blk[:, 4:].sum(axis=1) / block_sizes[:, None]
    P2_SE = block_sem_from_means(blk_P2, nb)
    M2_SE = block_sem_from_means(blk_P2t - blk_P2, nb)

    P2eq = 1.0 / (1.0 + np.exp(-beta * epsil))

    # ---- K12, K12' and M12 = K12' - K12 ----
    K12, K12_SE, u = k12_from_p2(P2, P2eq, P2_SE)
    K12t, _, ut = k12_from_p2(P2t, P2eq)
    M12 = K12t - K12
    # delta-method SE of M12, per block so the P_2 / P_2' correlation is kept: dK = dP_2 / u
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = (blk_P2t - P2t) / ut - (blk_P2 - P2) / u
    M12_SE = block_sem_from_means(dev, nb)
    M12_SE = np.where(np.isfinite(M12_SE), M12_SE, 0.0)

    # ---- full-average K12 line fit ----
    m_full, b_full, r_full, n_full = line_fit(time, K12, fitmin, fitmax)

    # ---- per-block K12 and per-block line fits ----
    win = (time >= fitmin) & (time <= fitmax)
    K12_blks, blk_m, blk_b = [], [], []
    n_nonphys, n_skipped = 0, 0
    for i in range(nb):
        K12_b, _, u_b = k12_from_p2(blk_P2[i], P2eq)
        K12_blks.append(K12_b)
        if np.any(u_b[win] <= 0.0):
            n_nonphys += 1
        m_b, b_b, _, _ = line_fit(time, K12_b, fitmin, fitmax)
        blk_m.append(m_b); blk_b.append(b_b)
        if not np.isfinite(m_b):
            n_skipped += 1

    blk_m = np.asarray(blk_m); blk_b = np.asarray(blk_b)
    good  = np.isfinite(blk_m)
    slopes, intercepts = blk_m[good], blk_b[good]
    if slopes.size >= 2:
        m_bar   = float(slopes.mean())
        b_bar   = float(intercepts.mean())
        sigma_m = float(slopes.std(ddof=1) / np.sqrt(slopes.size))
    elif slopes.size == 1:
        m_bar, b_bar, sigma_m = float(slopes[0]), float(intercepts[0]), float('nan')
    else:
        m_bar = b_bar = sigma_m = float('nan')

    return dict(time=time, P2=P2, P2_SE=P2_SE, P2t=P2t, M2=M2, M2_SE=M2_SE,
                terms=terms, terms_t=terms_t,
                K12=K12, K12_SE=K12_SE, K12t=K12t, K12_blks=K12_blks, M12=M12, M12_SE=M12_SE,
                m_full=m_full, b_full=b_full, r_full=r_full, n_full=n_full,
                slopes=slopes, intercepts=intercepts, blk_m=blk_m, blk_b=blk_b,
                m_bar=m_bar, b_bar=b_bar, sigma_m=sigma_m,
                n_nonphys=n_nonphys, n_skipped=n_skipped,
                P2eq=P2eq, beta=beta, n_traj=n_traj, nbds=nbds, nblocks=nb,
                n_files=len({u.file for u in units}), n_units=n_units, cache_stats=st,
                n_jumps=n_jumps, jumps=jumps, c0=c0, full_jump=full, Tjump=cfg.get('Tjump'),
                jump_label=jump_label(cfg), ngap=float(ngap), Rc=Rc, frac_past=past / n_traj,
                fitmin=fitmin, fitmax=fitmax, w0_only=w0_only,
                params=dict(kvec=kvec, epsil=epsil, lbd=lbd, delta=delta))


def _fmt_sci(x, sig=4):
    """Format x as mathtext, e.g. -2.442\\times10^{-3}; plain decimal for moderate magnitudes."""
    if not np.isfinite(x):
        return 'nan'
    if x == 0.0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    if -3 < e < 4:
        return f'{x:.{sig}g}'
    return rf'{x / 10.0**e:.{sig}g}\times10^{{{e}}}'


def _block_colors(plt, nb):
    """Distinct color per block: tab10 up to 10 blocks, else a viridis ramp."""
    if nb <= 10:
        return [plt.cm.tab10(i % 10) for i in range(nb)]
    return [plt.cm.viridis(x) for x in np.linspace(0.0, 0.9, nb)]


def _band(ax, t, y, se, color, label):
    ax.axhline(0.0, color='0.7', lw=0.8, ls='--')
    ax.plot(t, y, color=color, lw=1.6, label=label)
    ax.fill_between(t, y - se, y + se, color=color, alpha=0.3, lw=0)
    ax.set_xlabel('time (a.u.)')
    ax.legend(loc='best', frameon=False, fontsize=9)


def _plot(out, r, tmax=None):
    # imported here, not at module level: compute()'s spawned workers import this module, and none
    # of them plots, so they should not each pay matplotlib's import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = r['time']
    K12_blks = r['K12_blks']
    if tmax is not None:                         # truncate for plotting only (fits already done)
        msk = t <= tmax
        t = t[msk]
        K12_blks = [k[msk] for k in K12_blks]
        r = {k: (v[msk] if isinstance(v, np.ndarray) and v.shape == msk.shape else v)
             for k, v in r.items()}

    fmin, fmax = r['fitmin'], r['fitmax']
    tf = np.array([fmin, fmax])

    fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.5), sharex=True)

    # ---------------- top left: P_2 ----------------
    _band(ax[0, 0], t, r['P2'], r['P2_SE'], 'C0', r'$P_2(t)$')
    ax[0, 0].set_ylabel(r'$P_2(t)$')

    # ---------------- top right: M_2 ----------------
    _band(ax[0, 1], t, r['M2'], r['M2_SE'], 'C3', r"$M_2(t)=P_2'(t)-P_2(t)$")
    ax[0, 1].set_ylabel(r'$M_2(t)$')

    # ---------------- bottom left: K12, full + per-block, with the line fits ----------------
    ak = ax[1, 0]
    K12, K12_SE = r['K12'], r['K12_SE']
    ak.axvspan(fmin, fmax, color='0.92', zorder=0)
    ak.axhline(0.0, color='0.7', lw=0.8, ls='--')
    colors = _block_colors(plt, len(K12_blks))
    for i, K_b in enumerate(K12_blks):
        ak.plot(t, K_b, color=colors[i], lw=0.9, alpha=0.55)
        if np.isfinite(r['blk_m'][i]):
            ak.plot(tf, r['blk_m'][i] * tf + r['blk_b'][i], color=colors[i], lw=1.6)
    ak.plot(t, K12, color='C0', lw=1.6, zorder=4,
            label=r'$K_{12}(t)=-P_2^{\mathrm{eq}}\ln|1-P_2/P_2^{\mathrm{eq}}|$')
    ak.fill_between(t, K12 - K12_SE, K12 + K12_SE, color='C0', alpha=0.3, lw=0, zorder=3)
    if np.isfinite(r['m_full']):
        ak.plot(tf, r['m_full'] * tf + r['b_full'], color='k', lw=2.0, ls='--', zorder=6,
                label=(fr"fit [{fmin:g},{fmax:g}]:  m={r['m_full']:.4g}, "
                       fr"b={r['b_full']:.4g}, r={r['r_full']:.4f}"))
    if np.isfinite(r['m_bar']):
        ak.plot(tf, r['m_bar'] * tf + r['b_bar'], color='k', lw=2.5, zorder=5)
        ak.text(0.03, 0.03, rf"$m = {_fmt_sci(r['m_bar'])} \pm {_fmt_sci(r['sigma_m'])}$",
                transform=ak.transAxes, fontsize=10, va='bottom', ha='left')
    ak.set_xlabel('time (a.u.)'); ak.set_ylabel(r'$K_{12}(t)$')
    ak.legend(loc='best', frameon=False, fontsize=8)

    # clip the view: blocks crossing the log singularity spike arbitrarily high
    finite = [k[np.isfinite(k)] for k in K12_blks if np.isfinite(k).any()]
    finite = np.concatenate(finite) if finite else np.array([0.0])
    lo, hi = np.percentile(finite, [1.0, 99.0])
    band = np.concatenate([K12 - K12_SE, K12 + K12_SE]); band = band[np.isfinite(band)]
    if band.size:
        lo, hi = min(lo, band.min()), max(hi, band.max())
    ends = [v for i in range(len(K12_blks)) if np.isfinite(r['blk_m'][i])
            for v in (r['blk_m'][i] * tf + r['blk_b'][i])]
    if np.isfinite(r['m_bar']):
        ends += list(r['m_bar'] * tf + r['b_bar'])
    if np.isfinite(r['m_full']):
        ends += list(r['m_full'] * tf + r['b_full'])
    if ends:
        lo, hi = min(lo, min(ends)), max(hi, max(ends))
    pad = 0.12 * (hi - lo) if hi > lo else 1.0
    ak.set_ylim(lo - pad, hi + pad)

    # ---------------- bottom right: M12 ----------------
    _band(ax[1, 1], t, r['M12'], r['M12_SE'], 'C3',
          r"$M_{12}(t)=K_{12}'(t)-K_{12}(t)$")
    ax[1, 1].set_ylabel(r'$M_{12}(t)$')

    p = r['params']
    fig.suptitle(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}, "
                 f"{r['jump_label']}   ngap={r['ngap']:g}, Rc={r['Rc']:.4g}\n"
                 f"P2eq={r['P2eq']:.6f} (beta={r['beta']:g}, epsil={p['epsil']:g})   "
                 f"fit window [{fmin:g}, {fmax:g}]"
                 + ("   [W^(0) ONLY -- no-jump baseline]" if r['w0_only'] else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='analyze_p2 with the decoherence cut Ct = h(Rc - Rbar) D Sx: jump-weighted '
                    'P2(t), M2(t), K12(t) with a block analysis of the rate, and M12(t), from '
                    'RP-MASH shards or a data.hdf.')
    ap.add_argument('--file',    default=None,
                    help='a shard directory or a data.hdf (default: shards/ if it exists, else data.hdf)')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--fitmin',  type=float, default=5.0, help='fit window start (default 5)')
    ap.add_argument('--fitmax',  type=float, default=15.0, help='fit window end (default 15)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--beta',    type=float, default=None, help='override full inverse temperature')
    ap.add_argument('--epsil',   type=float, default=None, help='override driving force epsil')
    ap.add_argument('--ngap',    type=float, default=4.0,
                    help='decoherence cut: Rc = sqrt(2/(lbd*kvec)) (ngap/beta - epsil/2) (default 4)')
    ap.add_argument('--w0-only', action='store_true',
                    help='ignore the jumps: W^(m) -> W^(0), W^(m)(t) -> W^(0)(t) (no-jump baseline)')
    ap.add_argument('--c0',      type=float, default=None,
                    help='no-jump runs only: initial sampling factor (1 hemisphere, 2 full sphere); '
                         'default: inferred from Sz(0)')
    ap.add_argument('--unit-rows', type=int, default=1000,
                    help='trajectories per unit of work / cached partial (default 1000 = one shard of '
                         'the production config); the blocks are groups of whole units')
    ap.add_argument('--slab',    type=int, default=None, help=argparse.SUPPRESS)   # superseded by units
    ap.add_argument('--no-cache', action='store_true',
                    help='neither read nor write the per-unit partials (default: cache them in '
                         '<file>_partials/<version>/ and reuse them on the next run)')
    ap.add_argument('--refresh', action='store_true',
                    help='recompute (and re-cache) every unit, ignoring existing partials')
    ap.add_argument('--cache-dir', default=None, help='where the partials live (default: see --no-cache)')
    ap.add_argument('--ncores',  type=int, default=None,
                    help='worker processes computing units in parallel (default: the CPUs of the '
                         'SLURM allocation, else 1)')
    ap.add_argument('--out',     default='analyze_decoherence_rate.png', help='output png')
    a = ap.parse_args()
    path = a.file or ('shards' if os.path.isdir('shards') else 'data.hdf')

    try:
        r = compute(path, nblocks=a.nblocks, fitmin=a.fitmin, fitmax=a.fitmax,
                    beta_ovr=a.beta, epsil_ovr=a.epsil, w0_only=a.w0_only, c0=a.c0,
                    ncores=a.ncores, unit_rows=a.unit_rows, cache=not a.no_cache,
                    refresh=a.refresh, cache_dir=a.cache_dir, ngap=a.ngap)
    except RuntimeError as e:
        raise SystemExit(f'[dr] {e}')
    _plot(a.out, r, tmax=a.tmax)

    p = r['params']
    print(f"[dr] {path}: n_traj={r['n_traj']} from {r['n_files']} file(s) / {r['n_units']} unit(s), "
          f"nbds={r['nbds']}, nblocks={r['nblocks']}" + ("  [W^(0) ONLY]" if r['w0_only'] else ""))
    if r['jumps']:
        print(f"[dr] {r['jump_label']} full_jump={r['full_jump']}  total jumps across all trajectories: {r['n_jumps']}")
    else:
        print(f"[dr] no /weights table (old run, {r['jump_label']}): Tjump -> infinity, "
              f"W^(0) only, c0={r['c0']:g}" + (' (from --c0)' if a.c0 is not None else
                                                ' (inferred from Sz(0))'))
    print(f"[dr] params: kvec={p['kvec']} epsil={p['epsil']} lbd={p['lbd']} delta={p['delta']:.6e}")
    print(f"[dr] beta={r['beta']:g}  P2eq={r['P2eq']:.6f}")
    print(f"[dr] decoherence cut: ngap={r['ngap']:g}  Rc={r['Rc']:.6g}   fraction of trajectories "
          f"with Rbar > Rc (Ct cut off): t=0 {r['frac_past'][0]:.4f}, t=T {r['frac_past'][-1]:.4f}")
    print(f"[dr] P2(0)={r['P2'][0]:.6e} +/- {r['P2_SE'][0]:.3e}   "
          f"P2 range [{r['P2'].min():.6e}, {r['P2'].max():.6e}]")
    print(f"[dr] M2(0)={r['M2'][0]:.3e} (0 up to rounding)   "
          f"M2 range [{r['M2'].min():.6e}, {r['M2'].max():.6e}]")
    print(f"[dr] P2 term means at t=0: " + ', '.join(f'{v:.4e}' for v in r['terms'][:, 0]))
    print(f"[dr] FULL fit of K12 over t=[{r['fitmin']:g},{r['fitmax']:g}] "
          f"({r['n_full']} pts):  m={r['m_full']:.6g}  b={r['b_full']:.6g}  r={r['r_full']:.6f}")
    print(f"[dr] BLOCK analysis ({r['slopes'].size} usable blocks):  "
          f"m = {r['m_bar']:.6g} +/- {r['sigma_m']:.6g}")
    print(f"[dr] M12(T)={r['M12'][-1]:.6e} +/- {r['M12_SE'][-1]:.3e}")
    if r['n_nonphys']:
        print(f"[dr] WARNING: {r['n_nonphys']} of {r['nblocks']} blocks have u = 1 - P2/P2eq <= 0 "
              f"inside the fit window; K12 crosses the log singularity there, so those block slopes "
              f"are unphysical and distort the mean. Use fewer, larger blocks (--nblocks).")
    if r['n_skipped']:
        print(f"[dr] WARNING: {r['n_skipped']} block(s) had <2 finite points in the window "
              f"and were skipped.")
    print(f"[dr] wrote {a.out}")


if __name__ == '__main__':
    _main()
