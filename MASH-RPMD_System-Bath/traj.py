"""
traj.py -- Part 2 only: production, restarted from the shared thermalization bank.

The equilibration that used to run inline here now lives in therm.py and is executed ONCE per tree
into per-task shards, which consolidate_therm.py then merges into ONE file, <therm_dir>/therm_data.hdf,
holding every trajectory's restart state as row idx.  Part 1 does not depend on `epsil`, so
trajectory k of EVERY epsil_* directory in a tree restarts from the same row k.

This driver therefore:
  1. recovers its trajectory index from the cwd (workflow.worker chdirs into traj_grid/traj<idx>/
     and calls run_trajectory(cfg, seed) without passing the index),
  2. reads row idx of therm_data.hdf and validates it against this run's config,
  3. continues the SAME trajectory with the REAL parameters (epsil, delta).

therm_data.hdf is opened once per process and kept open (read-only), since a SLURM array task runs
its whole chunk of trajectories through run_trajectory in one process.

Because memP and the (already cyclically rolled) noise trajectory Ffluci are carried across,
production starts exactly as if the NVT had been run and the coupling then switched on -- verified
bit-exact against the old single-process two-part driver.

The restart is fail-fast: a missing file, a trajectory absent from it, or a mismatched record
raises instead of silently re-equilibrating.  Run the thermalize stage and consolidate first:

    cd <tree>/therm && python -m workflow.runner --config config_therm.py
    python consolidate_therm.py --dir <therm_dir>
    cd <tree>/epsil_0.00 && python -m workflow.runner --config config_traj.py

`therm_dir` (default '../therm') is resolved relative to the epsil run directory; `therm_file`
(default 'therm_data.hdf') is the file inside it.
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


_THERM_FILES = {}   # path -> open read-only h5py.File, reused across a task's whole chunk


def _open_therm(path):
    f = _THERM_FILES.get(path)
    if f is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'thermalization file not found: {path}\n'
                f'Run the thermalize stage for this tree and merge its shards first '
                f'(cd <tree>/therm && python -m workflow.runner --config config_therm.py; '
                f'python consolidate_therm.py --dir <therm_dir>).')
        f = _THERM_FILES[path] = h5py.File(path, 'r')
    return f


def _load_therm(path, idx, cfg):
    """
    Load and validate trajectory idx's thermalization record from the consolidated file.  Returns
    (nucR, nucP, memP, Ffluci), all float64 and C-contiguous.

    Validation matters here: integrator.__init__ only compares array SHAPES, with a bare exit() and
    no traceback on mismatch, and it would not notice a float32 or non-contiguous array at all.
    """
    f = _open_therm(path)
    n = f['present'].shape[0]
    if idx >= n or not f['present'][idx]:
        raise LookupError(f'{path}: trajectory {idx} is not in the thermalization bank '
                          f'({f.attrs.get("n_present")} of {n} present). Its therm run failed or '
                          f'its shard was missing at consolidation.')

    delt = float(cfg['delt']); lp = cfg['langevin_params']
    Nmem  = int(math.ceil(float(lp['Tmem']) / delt))
    Nfluc = int(float(lp['Tfluc']) / (2 * delt))

    nucR   = np.ascontiguousarray(f['nucR'][idx],   dtype=np.float64)
    nucP   = np.ascontiguousarray(f['nucP'][idx],   dtype=np.float64)
    memP   = np.ascontiguousarray(f['memP'][idx],   dtype=np.float64)
    Ffluci = np.ascontiguousarray(f['Ffluci'][idx], dtype=np.float64)
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

    return nucR, nucP, memP, Ffluci


def run_trajectory(cfg, seed):
    idx = traj_index()
    kvec, lbd = cfg['kvec'], cfg['lbd']; bath = np.asarray(cfg['bath'], float)

    # cwd is <epsil run dir>/traj_grid/traj<idx>  ->  the run dir is two levels up
    run_dir   = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    therm_dir = os.path.abspath(os.path.join(run_dir, cfg.get('therm_dir', '../therm')))
    path      = os.path.join(therm_dir, cfg.get('therm_file', 'therm_data.hdf'))
    #SCRATCH_DIR = '/storage/home/hcoda1/8/vsuarez6/scratch/'
    #SCRATCH_DIR = '/storage/scratch1/8/vsuarez6/'
    #path      = os.path.join(SCRATCH_DIR, 'therm', 'therm_data.hdf')

    nucR, nucP, memP, Ffluci = _load_therm(path, idx, cfg)
    print(f'[traj] traj {idx} restarting from row {idx} of {path}', flush=True)

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
