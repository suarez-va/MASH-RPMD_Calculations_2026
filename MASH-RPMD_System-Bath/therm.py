"""
therm.py -- Part 1 only: the shared GLE thermalization stage.

Analytically sample (R,P) from the NVT distribution, then run the GLE with the MODIFIED model
parameters (therm_epsil, therm_delta) for therm_time.  This fills the memory buffer memP with a
valid history and lets (R,P) relax.  The resulting restart state is written to

    <thermalize run dir>/therm<idx>.hdf

and NOTHING else is kept -- the per-trajectory scratch (.dat files, memK.dat) is deleted.

Why this is a separate stage: Part 1 does not depend on `epsil` at all (it runs with therm_epsil /
therm_delta), and within one tree the epsil_* configs differ ONLY in `epsil`.  So one bank of
thermalized states can seed the production run of every epsil_* directory: trajectory k in every
epsil_* dir restarts from therm<k>.hdf.  Previously each of those 9 x n_traj trajectories redid its
own identical 25000-step burn-in.

The trajectory index is recovered from the working directory, because workflow.worker chdirs into
traj_grid/traj<idx>/ and calls run_trajectory(cfg, seed) WITHOUT passing the index.

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


def therm_path(therm_dir, idx):
    """The canonical restart-file path for trajectory `idx`."""
    return os.path.join(therm_dir, f'therm_{idx:06d}.hdf')
    #SCRATCH_DIR = '/storage/home/hcoda1/8/vsuarez6/scratch/'
    #return os.path.join(SCRATCH_DIR, f'therm/therm_{idx:06d}.hdf')


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


def _write_therm(path, m1, Ffluci, cfg, seed, idx, Nburn):
    """Write the restart record atomically (tmp + os.replace) so a killed task leaves no half file."""
    Nmem, Nfluc, period = buffer_sizes(cfg)
    lp = cfg['langevin_params']
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        f.create_dataset('nucR',   data=np.ascontiguousarray(m1.nucR,        dtype=np.float64))
        f.create_dataset('nucP',   data=np.ascontiguousarray(m1.nucP,        dtype=np.float64))
        f.create_dataset('memP',   data=np.ascontiguousarray(m1.integ.memP,  dtype=np.float64))
        f.create_dataset('Ffluci', data=np.ascontiguousarray(Ffluci,         dtype=np.float64))
        a = f.attrs
        a['seed']         = -1 if seed is None else int(seed)
        a['idx']          = int(idx)
        a['nbds']         = int(cfg['nbds']);   a['nnuc'] = int(cfg['nnuc'])
        a['delt']         = float(cfg['delt']); a['beta'] = float(cfg['beta'])
        a['gamma']        = float(lp['gamma'])
        a['Tmem']         = float(lp['Tmem']);  a['Tfluc'] = float(lp['Tfluc'])
        a['Nmem']         = int(Nmem);          a['Nfluc'] = int(Nfluc)
        a['period']       = int(period)
        a['Nburn']        = int(Nburn)
        a['noise_offset'] = int(Nburn % period)
        a['therm_time']   = float(cfg['therm_time'])
        a['therm_epsil']  = float(cfg['therm_epsil'])
        a['therm_delta']  = float(cfg['therm_delta'])
        a['kvec']         = float(cfg['kvec']); a['lbd'] = float(cfg['lbd'])
        a['intype']       = str(cfg['intype'])
        a['langevin']     = str(cfg['langevin'])
        a['centroid_bool'] = True
    os.replace(tmp, path)


def run_trajectory(cfg, seed):
    idx  = traj_index()
    nbds = cfg['nbds']; delt = cfg['delt']
    kvec, lbd = cfg['kvec'], cfg['lbd']; bath = np.asarray(cfg['bath'], float)

    # cwd is <thermalize run dir>/traj_grid/traj<idx>  ->  the run dir is two levels up
    run_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    out     = therm_path(run_dir, idx)

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

    # This stage keeps ONLY the .hdf; drop the per-trajectory scratch (memK.dat alone is ~4 MB).
    for _f in _SCRATCH:
        if os.path.exists(_f):
            os.remove(_f)
    for _f in glob.glob('*.dat'):
        os.remove(_f)
