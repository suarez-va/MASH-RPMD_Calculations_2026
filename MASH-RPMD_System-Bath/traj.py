"""
traj.py -- Part 2 only: production, restarted from the shared thermalization bank.

The equilibration that used to run inline here now lives in therm.py and is executed ONCE per tree
into <tree>/thermalize/, producing therm<idx>.hdf per trajectory.  Part 1 does not depend on
`epsil`, so trajectory k of EVERY epsil_* directory in a tree restarts from the same
thermalize/therm<k>.hdf.

This driver therefore:
  1. recovers its trajectory index from the cwd (workflow.worker chdirs into traj_grid/traj<idx>/
     and calls run_trajectory(cfg, seed) without passing the index),
  2. loads <tree>/thermalize/therm<idx>.hdf and validates it against this run's config,
  3. continues the SAME trajectory with the REAL parameters (epsil, delta).

Because memP and the (already cyclically rolled) noise trajectory Ffluci are carried across,
production starts exactly as if the NVT had been run and the coupling then switched on -- verified
bit-exact against the old single-process two-part driver.

The restart is fail-fast: a missing or mismatched therm<idx>.hdf raises instead of silently
re-equilibrating.  Run the thermalize stage first:

    cd <tree>/thermalize && python -m workflow.runner --config config_therm.py
    cd <tree>/epsil_0.00 && python -m workflow.runner --config config_traj.py

`therm_dir` (default '../therm') is resolved relative to the epsil run directory.
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


def _load_therm(path, cfg):
    """
    Load and validate a thermalization record.  Returns (nucR, nucP, memP, Ffluci), all float64
    and C-contiguous.

    Validation matters here: integrator.__init__ only compares array SHAPES, with a bare exit() and
    no traceback on mismatch, and it would not notice a float32 or non-contiguous array at all.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'thermalization file not found: {path}\n'
            f'Run the thermalize stage for this tree first '
            f'(cd <tree>/thermalize && python -m workflow.runner --config config_therm.py).')

    delt = float(cfg['delt']); lp = cfg['langevin_params']
    Nmem  = int(math.ceil(float(lp['Tmem']) / delt))
    Nfluc = int(float(lp['Tfluc']) / (2 * delt))

    with h5py.File(path, 'r') as f:
        nucR   = np.ascontiguousarray(f['nucR'][:],   dtype=np.float64)
        nucP   = np.ascontiguousarray(f['nucP'][:],   dtype=np.float64)
        memP   = np.ascontiguousarray(f['memP'][:],   dtype=np.float64)
        Ffluci = np.ascontiguousarray(f['Ffluci'][:], dtype=np.float64)
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
    path      = os.path.join(therm_dir, f'therm_{idx:06d}.hdf')
    #SCRATCH_DIR = '/storage/home/hcoda1/8/vsuarez6/scratch/'
    #path      = os.path.join(SCRATCH_DIR, f'therm/therm_{idx:06d}.hdf')

    nucR, nucP, memP, Ffluci = _load_therm(path, cfg)
    print(f'[traj] traj {idx} restarting from {path}', flush=True)

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
