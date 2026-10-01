"""
therm.py -- Part 1 only: the shared GLE thermalization stage.

Analytically sample (R,P) from the NVT distribution, then run the GLE with the MODIFIED model
parameters (therm_epsil, therm_delta) for therm_time.  This fills the memory buffer memP with a
valid history and lets (R,P) relax.  The resulting restart state is APPENDED as one row to a shard

    <therm_dir>/shard_<SLURM_ARRAY_TASK_ID:04d>.hdf     (under SLURM: one shard per array task)
    <therm_dir>/shard_traj<idx:07d>.hdf                 (locally: one shard per trajectory)

and NOTHING else is kept -- the per-trajectory scratch (.dat files, memK.dat) is deleted.  A SLURM
array task runs its chunk of ~n_traj/495 trajectories sequentially in one process, so each shard has
exactly one writer and the shared therm dir sees ~495 files instead of n_traj.  After the stage
finishes, merge the shards into the single file traj.py reads:

    python consolidate_therm.py --dir <therm_dir>          # -> <therm_dir>/therm_data.hdf

Shard layout: per-trajectory datasets nucR, nucP, memP, Ffluci, seed, idx, all growing along axis 0
(one trajectory per chunk); the run-constant metadata lives in the root attrs.  `idx` is resized and
written LAST, so len(idx) is the committed row count -- a row past it is a write cut off by a kill,
which the next append overwrites and consolidation ignores.  A trajectory already committed to its
shard is skipped, so resubmitting the same submit_grid.sh after a timeout resumes each task.

Why this is a separate stage: Part 1 does not depend on `epsil` at all (it runs with therm_epsil /
therm_delta), and within one tree the epsil_* configs differ ONLY in `epsil`.  So one bank of
thermalized states can seed the production run of every epsil_* directory: trajectory k in every
epsil_* dir restarts from row k of therm_data.hdf.  Previously each of those 9 x n_traj trajectories
redid its own identical 25000-step burn-in.

The trajectory index is recovered from the working directory, because workflow.worker chdirs into
traj_grid/traj<idx>/ and calls run_trajectory(cfg, seed) WITHOUT passing the index.

`therm_dir` (cfg key) overrides where the shards go; the default is the hardcoded scratch path below.

Usage (from the thermalize run dir, PYTHONPATH=<RP_MASH root>):
    python -m workflow.runner --config config_therm.py
    python -m workflow.slurm  --config config_therm.py   &&  sbatch submit_grid.sh
"""

import os
import glob

import numpy as np
import h5py
import normal_mode
import mash_rpmd   # via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)


# files run_dynamics drops into the trajectory dir that this stage does not need
_SCRATCH = ('output.dat', 'nucR.dat', 'nucP.dat', 'mapSx.dat', 'mapSy.dat', 'mapSz.dat',
            'memK.dat', 'checkpoint.npz')


def traj_index(cwd=None):
    """Recover the trajectory index from the cwd (workflow.worker chdirs into .../traj<idx>/)."""
    base = os.path.basename(cwd or os.getcwd())
    if not base.startswith('traj') or not base[4:].isdigit():
        raise RuntimeError(f'expected the cwd to be a trajectory dir named traj<idx>, got {base!r}. '
                           f'This driver must be launched through workflow.worker/runner.')
    return int(base[4:])


#SCRATCH_DIR = '/storage/home/hcoda1/8/vsuarez6/scratch/'
#SCRATCH_DIR = '/storage/scratch1/8/vsuarez6/'
SCRATCH_DIR = '/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/repos/MASH-RPMD_Calculations_2026/MASH-RPMD_System-Bath/Low-Friction/MASH-Classical/'


def therm_dir(cfg):
    """Where the shards go: cfg['therm_dir'] if given, else <SCRATCH_DIR>/therm."""
    return os.path.abspath(cfg.get('therm_dir', os.path.join(SCRATCH_DIR, 'therm')))


def shard_path(therm_dir, idx):
    """
    The shard trajectory `idx` is appended to.  Under SLURM it is the array task's shard (the task
    runs its chunk sequentially, so one writer per file); locally workflow.runner runs every
    trajectory as its own concurrent process, so each gets its own shard instead.
    """
    task = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task is not None:
        return os.path.join(therm_dir, f'shard_{int(task):04d}.hdf')
    return os.path.join(therm_dir, f'shard_traj{idx:07d}.hdf')


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


def already_done(path, idx):
    """True if trajectory `idx` is committed to the shard at `path` (resume after a killed task)."""
    if not os.path.isfile(path):
        return False
    with h5py.File(path, 'r') as f:
        return 'idx' in f and bool(np.any(f['idx'][:] == idx))


def _write_therm(path, m1, Ffluci, cfg, seed, idx, Nburn):
    """
    Append trajectory `idx` as one row of the shard at `path`.  The data rows are written first and
    `idx` LAST, so len(idx) only counts rows that were fully written.  The shard is opened and closed
    per trajectory (not held open across the chunk) so a kill mid-chunk can at worst cut off the row
    being written, not leave the whole file unflushed.
    """
    rows = {
        'nucR'  : np.ascontiguousarray(m1.nucR,       dtype=np.float64),
        'nucP'  : np.ascontiguousarray(m1.nucP,       dtype=np.float64),
        'memP'  : np.ascontiguousarray(m1.integ.memP, dtype=np.float64),
        'Ffluci': np.ascontiguousarray(Ffluci,        dtype=np.float64),
        'seed'  : np.int64(-1 if seed is None else int(seed)),
    }
    attrs = _run_attrs(cfg, Nburn)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'a') as f:
        if 'idx' not in f:
            f.attrs.update(attrs)
            for name, arr in rows.items():
                if name not in f:      # a kill between dataset creations can leave some behind
                    sh = np.shape(arr)
                    f.create_dataset(name, shape=(0,) + sh, maxshape=(None,) + sh,
                                     chunks=(1,) + sh if sh else (4096,), dtype=arr.dtype)
            f.create_dataset('idx', shape=(0,), maxshape=(None,), chunks=(4096,), dtype='i8')
        else:
            for key, want in attrs.items():
                if f.attrs.get(key) != want:
                    raise ValueError(f'{path}: {key}={f.attrs.get(key)!r} but this run uses '
                                     f'{want!r}; this shard came from a different config.')
        n = f['idx'].shape[0]
        for name, arr in rows.items():
            d = f[name]
            if d.shape[0] < n + 1:
                d.resize(n + 1, axis=0)
            d[n] = arr
        f.flush()
        f['idx'].resize(n + 1, axis=0)
        f['idx'][n] = int(idx)


def run_trajectory(cfg, seed):
    idx  = traj_index()
    nbds = cfg['nbds']; delt = cfg['delt']
    kvec, lbd = cfg['kvec'], cfg['lbd']; bath = np.asarray(cfg['bath'], float)

    out = shard_path(therm_dir(cfg), idx)
    if already_done(out, idx):
        print(f'[therm] traj {idx} already in {out}; skipping', flush=True)
        return

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

    _write_therm(out, m1, Ffluci, cfg, seed, idx, Nburn)
    print(f'[therm] traj {idx} -> {out}', flush=True)

    # This stage keeps ONLY the shard row; drop the per-trajectory scratch (memK.dat alone is ~4 MB).
    for _f in _SCRATCH:
        if os.path.exists(_f):
            os.remove(_f)
    for _f in glob.glob('*.dat'):
        os.remove(_f)
