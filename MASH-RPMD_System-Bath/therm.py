"""
therm.py -- Part 1 only: the shared GLE thermalization stage.

Analytically sample (R,P) from the NVT distribution, then run the GLE with the MODIFIED model
parameters (therm_epsil, therm_delta) for therm_time.  This fills the memory buffer memP with a
valid history and lets (R,P) relax.  run_trajectory RETURNS the resulting restart state

    nucR (nbds, nnuc)   nucP (nbds, nnuc)   memP (nbds, nnuc, Nmem)   Ffluci (nbds, 2*Nfluc)   seed

and workflow.shards (shard mode, config key `shard_size`) stores it as row of the shard
<shard_dir>/shard_<s>.hdf with s = idx // shard_size -- exactly like production output: each SLURM
task owns whole shards, each trajectory runs in a temp dir that is deleted afterwards, a trajectory
is committed only once its traj_index entry is written, and a resubmitted task skips what is
already committed.  shard_attrs() adds this stage's run-constant metadata (Nmem, Nburn, therm_*,
...) to every shard's root attrs.  There is no consolidation step: traj.py opens the shard holding
trajectory idx and reads that row directly.

Why this is a separate stage: Part 1 does not depend on `epsil` at all (it runs with therm_epsil /
therm_delta), and within one tree the epsil_* configs differ ONLY in `epsil`.  So one bank of
thermalized states can seed the production run of every epsil_* directory: trajectory k in every
epsil_* dir restarts from trajectory k of the therm shards.  Previously each of those 9 x n_traj
trajectories redid its own identical 25000-step burn-in.

Usage (from the thermalize run dir, PYTHONPATH=<RP_MASH root>):
    python -m workflow.runner --config config_therm.py
    python -m workflow.slurm  --config config_therm.py   &&  sbatch submit_grid.sh
"""

import os

import numpy as np
import normal_mode
import mash_rpmd   # via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)


def traj_index(cwd=None):
    """Recover the trajectory index from the cwd (workflow.worker chdirs into .../traj<idx>/)."""
    base = os.path.basename(cwd or os.getcwd())
    if not base.startswith('traj') or not base[4:].isdigit():
        raise RuntimeError(f'expected the cwd to be a trajectory dir named traj<idx>, got {base!r}. '
                           f'This driver must be launched through workflow.worker/runner.')
    return int(base[4:])


def buffer_sizes(cfg):
    """(Nmem, Nfluc, period) exactly as integrator.__init__ computes them."""
    delt = float(cfg['delt'])
    lp   = cfg['langevin_params']
    Nmem  = int(np.ceil(float(lp['Tmem']) / delt))     # ceil
    Nfluc = int(float(lp['Tfluc']) / (2 * delt))       # truncation, NOT ceil
    return Nmem, Nfluc, 2 * Nfluc


def _sample_nvt(cfg, rng):
    nbds, nnuc = cfg['nbds'], cfg['nnuc']
    mass = np.asarray(cfg['mass'], float); beta = cfg['beta']
    kvec, lbd = cfg['kvec'], cfg['lbd']; gamma = cfg['langevin_params']['gamma']
    omega_k = normal_mode.calc_normal_mode_freq(beta / nbds, nbds)
    R0   = -np.sqrt(lbd / (2 * kvec))
    dR_k = np.sqrt(nbds / (beta * (kvec + mass[None, :] * (omega_k * (omega_k + gamma))[:, None])))
    nucR_nm = rng.normal(0.0, dR_k, (nbds, nnuc))
    nucR = np.zeros((nbds, nnuc))
    for i in range(nnuc):
        nucR[:, i] = normal_mode.normal_mode_to_real(nucR_nm[:, i])
    nucR[:, 0] += R0
    dP0  = np.sqrt(mass[0] * nbds / beta)
    nucP = rng.normal(0.0, dP0, (nbds, nnuc))
    return nucR, nucP


def _build(cfg, potparams, nucR, nucP, seed, **restart):
    return mash_rpmd.mash_rpmd(
        nstates=cfg['nstates'], nnuc=cfg['nnuc'], nbds=cfg['nbds'],
        beta=cfg['beta'], mass=np.asarray(cfg['mass'], float),
        potype='ET_with_bath', potparams=potparams, nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        langevin=cfg['langevin'], langevin_params=cfg['langevin_params'], seed=seed, **restart)


def _run_attrs(cfg, Nburn):
    """The run-constant metadata stored in a shard's root attrs (identical for every row)."""
    Nmem, Nfluc, period = buffer_sizes(cfg)
    lp = cfg['langevin_params']
    return {
        'nbds'         : int(cfg['nbds']),   'nnuc' : int(cfg['nnuc']),
        'delt'         : float(cfg['delt']), 'beta' : float(cfg['beta']),
        'gamma'        : float(lp['gamma']),
        'Tmem'         : float(lp['Tmem']),  'Tfluc': float(lp['Tfluc']),
        'Nmem'         : int(Nmem),          'Nfluc': int(Nfluc),
        'period'       : int(period),
        'Nburn'        : int(Nburn),
        'noise_offset' : int(Nburn % period),
        'therm_time'   : float(cfg['therm_time']),
        'therm_epsil'  : float(cfg['therm_epsil']),
        'therm_delta'  : float(cfg['therm_delta']),
        'kvec'         : float(cfg['kvec']), 'lbd'  : float(cfg['lbd']),
        'intype'       : str(cfg['intype']),
        'langevin'     : str(cfg['langevin']),
        'centroid_bool': True,
    }


def shard_attrs(cfg):
    """workflow.shards hook: extra root attrs for every shard of this stage."""
    return _run_attrs(cfg, round(cfg['therm_time'] / cfg['delt']))


def run_trajectory(cfg, seed, idx=None, run_dir=None):
    if idx is None:
        idx = traj_index()
    nbds = cfg['nbds']; delt = cfg['delt']
    kvec, lbd = cfg['kvec'], cfg['lbd']; bath = np.asarray(cfg['bath'], float)

    rng = np.random.default_rng(seed)
    nucR, nucP = _sample_nvt(cfg, rng)

    # ---- equilibration with the modified model params -> build memP / Ffluci ----
    pot_e = [kvec, cfg['therm_epsil'], lbd, cfg['therm_delta'], bath]
    m1 = _build(cfg, pot_e, nucR, nucP, seed)
    m1.mapSx = np.zeros(nbds); m1.mapSy = np.zeros(nbds); m1.mapSz = -np.ones(nbds)   # donor
    Nburn = round(cfg['therm_time'] / delt)
    m1.run_dynamics(Nburn, 10**9, delt, cfg['intype'], init_time=0.0, small_dt_ratio=1)

    # Cyclically shift the noise so index 0 is the continuation point (where equilibration ended,
    # Nburn steps in); production then reads Ffluci from index 0 with no phase-offset input.
    period = m1.integ.Ffluci.shape[1]
    Ffluci = np.roll(m1.integ.Ffluci, -(Nburn % period), axis=1)

    print(f'[therm] traj {idx} thermalized ({Nburn} steps)', flush=True)
    return {
        'nucR'  : np.ascontiguousarray(m1.nucR,       dtype=np.float64),
        'nucP'  : np.ascontiguousarray(m1.nucP,       dtype=np.float64),
        'memP'  : np.ascontiguousarray(m1.integ.memP, dtype=np.float64),
        'Ffluci': np.ascontiguousarray(Ffluci,        dtype=np.float64),
        'seed'  : np.int64(-1 if seed is None else int(seed)),
    }
