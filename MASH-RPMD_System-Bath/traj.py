"""
traj.py -- Part 2 only: production, restarted from the shared thermalization bank.

The equilibration that used to run inline here now lives in therm.py and is executed ONCE per tree.
Its restart states live in workflow.shards files <therm_dir>/shard_<s>.hdf, one row per trajectory.
Part 1 does not depend on `epsil`, so trajectory k of EVERY epsil_* directory in a tree restarts
from the same therm row.  There is no consolidation step: this driver opens the therm shard holding
its trajectory and reads that row directly.

This driver therefore:
  1. takes its trajectory index and run dir from workflow.shards (or, in the old one-dir-per-
     trajectory mode, recovers them from the cwd traj_grid/traj<idx>/),
  2. finds the therm shard s = idx // shard_size (shard_size read from the therm shards' own attrs,
     so it need not match the production shard_size), looks idx up in that shard's traj_index, reads
     the row and validates it against this run's config,
  3. continues the SAME trajectory with the REAL parameters (epsil, delta).

The therm dir is listed once per process and one therm shard is kept open (read-only) at a time; a
task's trajectories run in index order, so it switches shard only when idx crosses into the next.

Because memP and the (already cyclically rolled) noise trajectory Ffluci are carried across,
production starts exactly as if the NVT had been run and the coupling then switched on -- verified
bit-exact against the old single-process two-part driver.

The restart is fail-fast: a missing shard, a trajectory absent from it, or a mismatched record
raises instead of silently re-equilibrating.  Run the thermalize stage first:

    cd <tree>/therm && python -m workflow.runner --config config_therm.py
    cd <tree>/epsil_0.00 && python -m workflow.runner --config config_traj.py

`therm_dir` (default '../therm_shards') is resolved relative to the epsil run directory and must be
the therm stage's shard_dir.
"""

import os
import math

import numpy as np
import h5py
import normal_mode
import mash_rpmd   # via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)


def traj_index(cwd=None):
    """Recover the trajectory index from the cwd (workflow.worker chdirs into .../traj<idx>/)."""
    base = os.path.basename(cwd or os.getcwd())
    if not base.startswith('traj') or not base[4:].isdigit():
        raise RuntimeError(f'expected the cwd to be a trajectory dir named traj<idx>, got {base!r}. '
                           f'This driver must be launched through workflow.worker/runner.')
    return int(base[4:])


def _build(cfg, potparams, nucR, nucP, seed, **restart):
    return mash_rpmd.mash_rpmd(
        nstates=cfg['nstates'], nnuc=cfg['nnuc'], nbds=cfg['nbds'],
        beta=cfg['beta'], mass=np.asarray(cfg['mass'], float),
        potype='ET_with_bath', potparams=potparams, nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        langevin=cfg['langevin'], langevin_params=cfg['langevin_params'],
        Tjump=cfg['Tjump'], full_jump=cfg['full_jump'], seed=seed, **restart)


# the thermalization bank, cached per process: shard number -> path (listed once), the bank's
# shard_size, and the ONE therm shard currently open as (s, path, h5py.File, {idx: row})
_BANK = {'dir': None, 'paths': None, 'size': None, 'cur': None}

_RUN_THERM = ('Run the thermalize stage for this tree first '
              '(cd <tree>/therm && python -m workflow.runner --config config_therm.py).')


def _therm_shard(therm_dir, idx):
    """(path, open file, row of idx) for the therm shard holding trajectory idx."""
    if _BANK['dir'] != therm_dir:
        paths = {}
        for name in (os.listdir(therm_dir) if os.path.isdir(therm_dir) else ()):
            stem = name[len('shard_'):-len('.hdf')]
            if name.startswith('shard_') and name.endswith('.hdf') and stem.isdigit():
                paths[int(stem)] = os.path.join(therm_dir, name)
        if not paths:
            raise FileNotFoundError(f'no thermalization shards (shard_*.hdf) in {therm_dir}\n{_RUN_THERM}')
        with h5py.File(next(iter(paths.values())), 'r') as f:
            if 'shard_size' not in f.attrs or 'traj_index' not in f:
                raise ValueError(f'{f.filename} is not a workflow.shards thermalization shard (an old '
                                 f'format bank?).\n{_RUN_THERM}')
            size = int(f.attrs['shard_size'])
        if _BANK['cur'] is not None:
            _BANK['cur'][2].close()
        _BANK.update(dir=therm_dir, paths=paths, size=size, cur=None)

    s = idx // _BANK['size']
    cur = _BANK['cur']
    if cur is None or cur[0] != s:
        if cur is not None:
            cur[2].close()
            _BANK['cur'] = None
        path = _BANK['paths'].get(s)
        if path is None:
            raise FileNotFoundError(f'no thermalization shard {s} (trajectory {idx}) in {therm_dir}\n'
                                    f'{_RUN_THERM}')
        f = h5py.File(path, 'r')
        rows = {int(k): r for r, k in enumerate(f['traj_index'][:])}
        cur = _BANK['cur'] = (s, path, f, rows)
    s, path, f, rows = cur
    if idx not in rows:
        raise LookupError(f'{path}: trajectory {idx} is not in this thermalization shard '
                          f'({len(rows)} committed). Its therm run failed or did not finish; '
                          f'resubmit the therm stage.')
    return path, f, rows[idx]


def _load_therm(therm_dir, idx, cfg):
    """
    Load and validate trajectory idx's thermalization record from its therm shard.  Returns
    (path, row, nucR, nucP, memP, Ffluci), the arrays float64 and C-contiguous.

    Validation matters here: integrator.__init__ only compares array SHAPES, with a bare exit() and
    no traceback on mismatch, and it would not notice a float32 or non-contiguous array at all.
    """
    path, f, row = _therm_shard(therm_dir, idx)

    delt = float(cfg['delt']); lp = cfg['langevin_params']
    Nmem  = int(math.ceil(float(lp['Tmem']) / delt))
    Nfluc = int(float(lp['Tfluc']) / (2 * delt))

    nucR   = np.ascontiguousarray(f['nucR'][row],   dtype=np.float64)
    nucP   = np.ascontiguousarray(f['nucP'][row],   dtype=np.float64)
    memP   = np.ascontiguousarray(f['memP'][row],   dtype=np.float64)
    Ffluci = np.ascontiguousarray(f['Ffluci'][row], dtype=np.float64)
    a = dict(f.attrs)

    # the physics that must NOT differ between equilibration and production
    for key, want in (('nbds',  int(cfg['nbds'])),   ('nnuc',  int(cfg['nnuc'])),
                      ('delt',  float(cfg['delt'])), ('beta',  float(cfg['beta'])),
                      ('gamma', float(lp['gamma'])),
                      ('Tmem',  float(lp['Tmem'])),  ('Tfluc', float(lp['Tfluc']))):
        if key in a and a[key] != want:
            raise ValueError(f'{path}: {key}={a[key]!r} but this run uses {want!r}. '
                             f'The thermalization bank is not compatible with this config.')

    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])
    if memP.shape != (nbds, nnuc, Nmem):
        raise ValueError(f'{path}: memP has shape {memP.shape}, expected {(nbds, nnuc, Nmem)}')
    if Ffluci.shape != (nbds, 2 * Nfluc):
        raise ValueError(f'{path}: Ffluci has shape {Ffluci.shape}, expected {(nbds, 2 * Nfluc)}')
    if nucR.shape != (nbds, nnuc) or nucP.shape != (nbds, nnuc):
        raise ValueError(f'{path}: nucR/nucP shapes {nucR.shape}/{nucP.shape}, '
                         f'expected {(nbds, nnuc)}')

    # spin_magnus_init rebuilds Fdiss from memP BEFORE re-seeding memP[:,:,-1] from nucP, so the two
    # must already agree -- otherwise the restart is silently wrong rather than an error.
    for i in range(nnuc):
        if not np.allclose(memP[:, i, -1], normal_mode.real_to_normal_mode(nucP[:, i]),
                           rtol=0.0, atol=1e-12):
            raise ValueError(f'{path}: memP[:,{i},-1] does not match the normal-mode transform of '
                             f'nucP[:,{i}]; the restart state is inconsistent.')

    return path, row, nucR, nucP, memP, Ffluci


def run_trajectory(cfg, seed, idx=None, run_dir=None):
    # Shard mode (workflow.shards) runs each trajectory in a temp dir off the run dir and passes idx
    # and run_dir in.  Otherwise the cwd is <epsil run dir>/traj_grid/traj<idx>, and both come from it.
    if idx is None:
        idx = traj_index()
    if run_dir is None:
        run_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    kvec, lbd = cfg['kvec'], cfg['lbd']; bath = np.asarray(cfg['bath'], float)

    # an absolute therm_dir (e.g. on scratch) is used as is
    therm_dir = os.path.abspath(os.path.join(run_dir, cfg.get('therm_dir', '../therm_shards')))

    path, row, nucR, nucP, memP, Ffluci = _load_therm(therm_dir, idx, cfg)
    print(f'[traj] traj {idx} restarting from row {row} of {path}', flush=True)

    # ---- production with the REAL params, continuing from the thermalized state ----
    pot_p = [kvec, cfg['epsil'], lbd, cfg['delta'], bath]
    m2 = _build(cfg, pot_p, nucR, nucP, seed, init_memP=memP, init_Ffluci=Ffluci)
    #m2.init_map_spin(init_state=0)
    m2.init_map_spin(init_state=None)
    Nprod = round(cfg['traj_time'] / cfg['delt'])
    m2.run_dynamics(Nprod, cfg['Nprint'], cfg['delt'], cfg['intype'],
                    init_time=0.0, small_dt_ratio=1)

    # Drop the per-trajectory scratch not needed downstream (memK.dat is the friction kernel, ~4 MB
    # of ASCII, recomputable from config).  Only touch files inside this trajectory dir.
    for _f in ('memK.dat',):
        if os.path.exists(_f):
            os.remove(_f)
