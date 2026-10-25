"""
analyze_tully_rho.py -- the channel-resolved scattering populations of a MASH-RPMD_Tully-Testing run,
built from the stages' density operator rho and read straight from every stage's shards: the Testing
counterpart of MASH-RPMD_Tully-Delta/analyze_tully.py (C only, no M).  Output is one figure
(analyze_tully_rho.png).  (Not named analyze_tully.py: --ref imports that module by name, and spawned
workers must not find this file instead.)

--- the observables --------------------------------------------------------------------------------

    C_Rg(t) = int dq dp  h(-q) [rho_I(q,p,t) - rho_z(q,p,t)]        reflected,   ground  adiabat
    C_Tg(t) = int dq dp  h( q) [rho_I(q,p,t) - rho_z(q,p,t)]        transmitted, ground
    C_Re(t) = int dq dp  h(-q) [rho_I(q,p,t) + rho_z(q,p,t)]        reflected,   excited
    C_Te(t) = int dq dp  h( q) [rho_I(q,p,t) + rho_z(q,p,t)]        transmitted, excited

with h the Heaviside step (h(0) = 1/2): rho_I -+ rho_z = <g|rho|g>, <e|rho|e>, split at the crossing
q = 0, as C_Rg .. C_Te are in the Tully-Delta script.

--- which stage supplies rho(q,p,t) -----------------------------------------------------------------

rho(t) is always built from the CURRENT stage, the most recent one started at or before t:

    stage m  for  t_m <= t < t_m+1          (the last stage up to its final time)

so at a boundary t_m+1 the new stage's first row (right after the resample) is used and the old
stage's last row is not -- the post-jump convention of analyze_tully.py.  Within stage m, every
trajectory j (point i(j) of stage_<m>/init.npz with weights w = (w_I, w_z, w_x, w_y), spin S^(0) at
t_m, then q_j(t), S_j(t)) deposits

    (rho_I -+ rho_z)(q,p,t)  =  sum_j  a_P,j  h(-+S_z,j(t))  delta(q - q_j(t)) delta(p - p_j(t)) / n_spin
    a_P = 4 (w_I |S_z^(0)| + w_z sgn(S_z^(0)) |S_z^(0)| + w_x S_x^(0) + w_y S_y^(0))

(the 1/2-normalized update of resample.py: rho_I - rho_z = 1/2 a_P (1 - sgn S_z(t)) / n_spin), so

    C_Rg(t) = sum_j a_P,j h(-q_j(t)) h(-S_z,j(t)) / n_spin,   and likewise for Tg, Re, Te.

The full time axis therefore evaluates points from every stage, each stage weighted by its own
init.npz.  Beads as in the Tully-Delta script: the channel factor h(+-q_b) h(+-S_z,b) is taken bead by
bead and averaged; the S^(0) inside a_P is the bead mean (sgn: bead mean of sign).  For nbds == 1
(what this tree runs) every bead convention coincides.

Exact identity, checked and reported: within a stage C_Rg + C_Tg + C_Re + C_Te = Tr rho is constant
(sum_j a_P,j / n_spin does not depend on t).  These are also the Rg..Te that `iterate.py resample`
stores in summary.npz (same formula), which this script reproduces from the raw shards.

--- error bars -------------------------------------------------------------------------------------

Per stage, across its points: every point's n_spin trajectories are summed into one contribution
c_i(t), and SE = sqrt(N) std_i(c_i) -- the sampling error of that stage given its initial points
(earlier stages' noise, frozen into those points, is not propagated).

--- input, parallelism, cache ----------------------------------------------------------------------

Every stage_<m>/ next to --config with a config.py, init.npz and a complete shards/ directory.
Stages are taken in order up to the first incomplete one (reported and left out: a stage is a sum
over its points, so missing trajectories would bias it).  Each stage's shards are cut into units of
at most --unit-rows trajectories (workflow.plan_units) and reduced on --ncores workers, with the
partials cached in stage_<m>/<shards>_partials/<version>/ (the version includes a hash of that
stage's init.npz).

--ref <shards> overlays the C channels of a MASH-RPMD_Tully-Delta run (its analyze_tully.py, dashed).
With jump_index at this run's stage boundaries (k = m seg_time / delt) and the same initial state
the two agree on average: the end-to-end test of the stage scheme.

    python analyze_tully_rho.py --config config.py --ncores 8
    python analyze_tully_rho.py --config config.py --ref ../MASH-RPMD_Tully-Delta/<run>/shards
"""

import os
import sys
import glob
import hashlib
import argparse
import functools
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import resample as rs   # noqa: E402

from workflow import plan_units, map_cached, ncores_source, open_readonly   # cached map-reduce


# Bump whenever the observables change: every cached partial is then recomputed.
_VERSION = 'tt_tully_v1'

_NAMES = ('CRg', 'CTg', 'CRe', 'CTe')


def _load_config(path):
    spec = importlib.util.spec_from_file_location('_tt_stage_config', os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


_W0 = {}   # init.npz path -> point weights, loaded once per worker


def _weights(init_path):
    if init_path not in _W0:
        with np.load(init_path) as z:
            _W0[init_path] = np.array(z['w'], float)
    return _W0[init_path]


def _unit_partial(path, lo, hi, init_path, n_spin, time):
    """
    MAP step, run in a worker: per-point sums of the four channels over rows lo..hi of one shard.
    Returns pt (k,) the points present, sums (k, 4, T) and cnt (k,) their trajectory counts.
    """
    with open_readonly(path) as f:                     # no lock: never blocks a running writer
        if not np.array_equal(f['time'][:], time):
            raise ValueError(f'{path}: time grid differs from the rest of the stage')
        idx = f['traj_index'][lo:hi]
        Sx0 = f['mapSx'][lo:hi, 0]                     # (s, nbds)
        Sy0 = f['mapSy'][lo:hi, 0]
        Sz = f['mapSz'][lo:hi]                         # (s, T, nbds)
        R = f['nucR'][lo:hi, :, :, 0]                  # (s, T, nbds)
    i = idx // n_spin
    aP, _ = rs.source_factors(_weights(init_path)[i],
                              (Sx0.mean(-1), Sy0.mean(-1), np.sign(Sz[:, 0]).mean(-1), Sz[:, 0].mean(-1)))
    hg, he = rs.heaviside(-Sz), rs.heaviside(Sz)
    hl, hr = rs.heaviside(-R), rs.heaviside(R)
    c = aP[:, None] / n_spin
    obs = np.stack([c * (hl * hg).mean(-1), c * (hr * hg).mean(-1),
                    c * (hl * he).mean(-1), c * (hr * he).mean(-1)], axis=1)   # (s, 4, T)
    pt, inv = np.unique(i, return_inverse=True)
    sums = np.zeros((pt.size, len(_NAMES), time.size))
    np.add.at(sums, inv, obs)
    return dict(pt=pt, sums=sums, cnt=np.bincount(inv, minlength=pt.size))


def _file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def find_stages(run_dir):
    """[(m, stage dir)] of every stage_<m>/ with a config.py, in stage order."""
    out = []
    for sd in sorted(glob.glob(os.path.join(run_dir, 'stage_*'))):
        tail = os.path.basename(sd)[len('stage_'):]
        if tail.isdigit() and os.path.isfile(os.path.join(sd, 'config.py')):
            out.append((int(tail), sd))
    return out


def compute_stage(sd, ncores=None, unit_rows=1000, cache=True, refresh=False):
    """
    REDUCE step for one stage: per-point sums -> the four channels (4, T), their SE, and Tr.
    Returns None (with the reason) if the stage is not complete.
    """
    scfg = _load_config(os.path.join(sd, 'config.py'))
    init_path = os.path.join(sd, 'init.npz')
    with np.load(init_path) as z:
        n_pts, n_spin = int(z['q'].size), int(z['n_spin'])
    shard_path = os.path.join(sd, scfg.get('shard_dir', 'shards'))
    if not os.path.isdir(shard_path):
        return None, 'no shards yet'
    units = plan_units(shard_path, unit_rows)
    n_done = sum(u.hi - u.lo for u in units)
    if n_done != int(scfg['n_traj']) or not all(u.complete for u in units):
        return None, f'{n_done}/{scfg["n_traj"]} trajectories committed'
    with open_readonly(units[0].file) as f:
        time = f['time'][:]
        nbds = int(f.attrs['nbds'])

    version = f'{_VERSION}_{_file_hash(init_path)}'
    cache_dir = os.path.join(os.path.normpath(shard_path) + '_partials', version)
    fn = functools.partial(_unit_partial, init_path=init_path, n_spin=n_spin, time=time)
    parts, st = map_cached(fn, units, cache_dir, version, ncores=ncores, refresh=refresh,
                           use_cache=cache, tag=f'[tully] stage {int(scfg["stage"])}:')
    for note in st['notes']:
        print(f'[tully] WARNING: {note}', flush=True)

    acc = np.zeros((n_pts, len(_NAMES), time.size))
    cnt = np.zeros(n_pts, dtype=np.int64)
    for p in parts:
        np.add.at(acc, p['pt'], p['sums'])
        np.add.at(cnt, p['pt'], p['cnt'])
    if not np.all(cnt == n_spin):
        raise RuntimeError(f'{sd}: points with {sorted(set(cnt.tolist()))} trajectories, expected '
                           f'{n_spin} each')
    tot = acc.sum(axis=0)
    se = np.sqrt(n_pts) * acc.std(axis=0, ddof=1) if n_pts > 1 else np.zeros_like(tot)
    tr = acc.sum(axis=1)                               # (n_pts, T): per-point trace
    return dict(stage=int(scfg['stage']), t_start=float(scfg['t_start']), time=time, tot=tot, se=se,
                Tr=tr.sum(axis=0), Tr_SE=(np.sqrt(n_pts) * tr.std(axis=0, ddof=1) if n_pts > 1
                                          else np.zeros(time.size)),
                n_points=n_pts, n_spin=n_spin, n_traj=int(scfg['n_traj']), nbds=nbds,
                cache_stats=st), None


def compute(run_dir, ncores=None, unit_rows=1000, cache=True, refresh=False):
    """
    Every complete stage in order (up to the first incomplete one), stitched onto one time axis:
    stage m keeps t < t_m+1, the last included stage keeps everything.
    """
    stages, skipped = [], []
    for m, sd in find_stages(run_dir):
        r, why = compute_stage(sd, ncores=ncores, unit_rows=unit_rows, cache=cache, refresh=refresh)
        if r is None:
            skipped.append((m, why))
            break                                       # a later stage cannot be stitched past a gap
        stages.append(r)
    if not stages:
        raise RuntimeError(f'no complete stage under {run_dir}' +
                           (f' (stage {skipped[0][0]}: {skipped[0][1]})' if skipped else ''))

    t, tot, se, tr, tr_se, sid = [], [], [], [], [], []
    for k, s in enumerate(stages):
        keep = np.ones(s['time'].size, bool)
        if k + 1 < len(stages):
            t_next = stages[k + 1]['t_start']
            keep = s['time'] < t_next - 1e-9 * max(1.0, abs(t_next))
        t.append(s['time'][keep]); tot.append(s['tot'][:, keep]); se.append(s['se'][:, keep])
        tr.append(s['Tr'][keep]); tr_se.append(s['Tr_SE'][keep]); sid.append(np.full(keep.sum(), s['stage']))
    tot, se = np.concatenate(tot, axis=1), np.concatenate(se, axis=1)
    r = {k: tot[i] for i, k in enumerate(_NAMES)}
    r.update({k + '_SE': se[i] for i, k in enumerate(_NAMES)})
    r.update(time=np.concatenate(t), stage_of_t=np.concatenate(sid), Tr=np.concatenate(tr),
             Tr_SE=np.concatenate(tr_se), stages=stages, skipped=skipped,
             t_starts=np.array([s['t_start'] for s in stages]),
             identity_Tr=float(max(np.ptp(s['Tr']) for s in stages)))
    return r


def _load_delta_analyze_tully():
    """MASH-RPMD_Tully-Delta/analyze_tully.py, imported by name: its spawned workers re-import it so,
    from the sys.path they inherit."""
    sys.path.insert(1, os.path.join(os.path.dirname(HERE), 'MASH-RPMD_Tully-Delta'))
    return importlib.import_module('analyze_tully')


def _plot(out, r, ref=None, tmax=None):
    # imported here, not at module level: spawned workers import this module and never plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = r['time']
    msk = np.ones(t.size, bool) if tmax is None else t <= tmax
    t = t[msk]
    colors = ('C0', 'C2', 'C1', 'C3')
    labels = (r'$C_{Rg}(t)=\int dq\,dp\,h(-q)\,(\rho_I-\rho_z)$', r'$C_{Tg}(t)=\int dq\,dp\,h(q)\,(\rho_I-\rho_z)$',
              r'$C_{Re}(t)=\int dq\,dp\,h(-q)\,(\rho_I+\rho_z)$', r'$C_{Te}(t)=\int dq\,dp\,h(q)\,(\rho_I+\rho_z)$')

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.5))
    ax.axhline(0.0, color='0.7', lw=0.8, ls='--')
    for tb in r['t_starts'][1:]:
        if tmax is None or tb <= tmax:
            ax.axvline(tb, color='0.6', lw=0.8, ls=':')
    for k, lab, c in zip(_NAMES, labels, colors):
        y, s = r[k][msk], r[k + '_SE'][msk]
        ax.plot(t, y, color=c, lw=1.6, label=lab)
        ax.fill_between(t, y - s, y + s, color=c, alpha=0.3, lw=0)
    if ref is not None:
        tr = ref['time'] if tmax is None else ref['time'][ref['time'] <= tmax]
        for k, c in zip(_NAMES, colors):
            ax.plot(tr, ref[k][:tr.size], color=c, lw=1.1, ls='--')
        ax.plot([], [], color='0.3', lw=1.1, ls='--', label=r'Tully-Delta $C$ (--ref)')
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$C(t)$')
    ax.legend(loc='best', frameon=False, fontsize=9)

    s0 = r['stages'][0]
    title = (f"{len(r['stages'])} stage(s) x {s0['n_points']} points x {s0['n_spin']} spins, "
             f"nbds={s0['nbds']}")
    if ref is not None:
        title += f"\n--ref: {ref['jump_label']}, {ref['n_traj']} traj"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Channel-resolved Tully populations C_Rg, C_Tg, C_Re, C_Te of a '
                    'MASH-RPMD_Tully-Testing run, stitched across its stages from the shards.')
    ap.add_argument('--config', default='config.py', help="the run's base config (stages sit next to it)")
    ap.add_argument('--ncores', type=int, default=None,
                    help='worker processes (default: the CPUs of the SLURM allocation, else 1)')
    ap.add_argument('--unit-rows', type=int, default=1000,
                    help='trajectories per unit of work / cached partial (default 1000)')
    ap.add_argument('--no-cache', action='store_true',
                    help='neither read nor write the per-unit partials (default: cache them in '
                         'stage_<m>/<shards>_partials/' + _VERSION + '_<init hash>/)')
    ap.add_argument('--refresh', action='store_true', help='recompute (and re-cache) every unit')
    ap.add_argument('--ref', default=None,
                    help='MASH-RPMD_Tully-Delta shards (or data.hdf) whose C channels to overlay')
    ap.add_argument('--tmax', type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--out', default='analyze_tully_rho.png', help='output png')
    a = ap.parse_args()

    if a.ncores is None:
        ncores, why = ncores_source()
    else:
        ncores, why = a.ncores, 'given --ncores'
    run_dir = os.path.dirname(os.path.abspath(a.config))
    try:
        r = compute(run_dir, ncores=ncores, unit_rows=a.unit_rows, cache=not a.no_cache,
                    refresh=a.refresh)
    except RuntimeError as e:
        raise SystemExit(f'[tully] {e}')
    ref = None
    if a.ref:
        ref = _load_delta_analyze_tully().compute(a.ref, ncores=ncores)
    _plot(a.out, r, ref=ref, tmax=a.tmax)

    print(f'[tully] {run_dir}: {len(r["stages"])} stage(s) on {ncores} worker(s) ({why})')
    for m, why_skip in r['skipped']:
        print(f'[tully] stage {m} left out: {why_skip} (and every stage after it)')
    for s in r['stages']:
        st = s['cache_stats']
        print(f"[tully] stage {s['stage']}: t {s['t_start']:g} -> {s['time'][-1]:g}, {s['n_points']} points x "
              f"{s['n_spin']} spins ({st['cached']} unit(s) from cache, {st['computed']} computed)   "
              f"Tr = {s['Tr'][0]:.6f} +/- {s['Tr_SE'][0]:.6f}")
    for k, name in zip(_NAMES, ('C_Rg', 'C_Tg', 'C_Re', 'C_Te')):
        print(f"[tully] {name}(0)={r[k][0]: .6e} +/- {r[k + '_SE'][0]:.3e}   "
              f"{name}(T)={r[k][-1]: .6e} +/- {r[k + '_SE'][-1]:.3e}")
    scale = max(1.0, float(np.abs(r['Tr']).max()))
    print(f"[tully] identity: max over stages of the spread of C_Rg + C_Tg + C_Re + C_Te within the "
          f"stage = {r['identity_Tr']:.3e}")
    if r['identity_Tr'] > 1e-10 * scale:
        print('[tully] WARNING: Tr rho changes within a stage beyond round-off. Something is wrong.')
    if ref is not None:
        kr = np.searchsorted(ref['time'], r['time'])
        ok = (kr < ref['time'].size) & np.isclose(ref['time'][np.minimum(kr, ref['time'].size - 1)], r['time'])
        zs = []
        for k in _NAMES:
            d = r[k][ok] - ref[k][kr[ok]]
            s = np.hypot(r[k + '_SE'][ok], ref[k + '_SE'][kr[ok]])
            zs.append(float(np.max(np.where(s > 0, np.abs(d) / np.where(s > 0, s, 1.0), 0.0))) if ok.any() else np.nan)
        print(f"[tully] vs --ref ({ref['jump_label']}, {ref['n_traj']} traj): max |difference|/SE over "
              f"{int(ok.sum())} common print times   " + '  '.join(f'{n} {z:.1f}' for n, z in zip(_NAMES, zs)))
    print(f'[tully] wrote {a.out}')


if __name__ == '__main__':
    _main()
