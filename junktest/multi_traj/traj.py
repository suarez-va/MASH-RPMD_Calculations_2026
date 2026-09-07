"""
system.py -- the physics for one trajectory. The workflow calls run_trajectory(cfg, seed) once per
trajectory, already cd'd into that trajectory's own directory, so RP_MASH's hardcoded output files
(output.dat, nucR.dat, nucP.dat, mapS[xyz].dat, memK.dat, info.txt) land there without collisions.

This is the single-trajectory driver (run_mashrpmd.py) with two changes:
  1. all numbers come from cfg instead of being hardcoded, and
  2. `seed` is threaded into BOTH the sampling RNG and the mash_rpmd object, so trajectory k is
     reproducible and independent of the others.
"""

import numpy as np

import mash_rpmd   # located via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)
import normal_mode

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

def run_trajectory(cfg, seed):
    # one RNG for the initial-condition sampling; seed=None -> OS entropy (non-reproducible)
    rng = np.random.default_rng(seed)

    nbds    = cfg['nbds']
    nnuc    = cfg['nnuc']
    nstates = cfg['nstates']
    mass    = np.asarray(cfg['mass'], dtype=float)
    beta    = cfg['beta']

    kvec, epsil, lbd, delta = cfg['kvec'], cfg['epsil'], cfg['lbd'], cfg['delta']
    bathvec   = np.asarray(cfg['bath'], dtype=float)
    potparams = [kvec, epsil, lbd, delta, bathvec]

    delt   = cfg['delt']
    Nsteps = round(cfg['total_time'] / delt)
    Nprint = cfg['Nprint']

    # initial conditions: reactant (state-0) well of the Marcus parabola
    #R0  = -np.sqrt(lbd / (2 * kvec))
    #dR0 = np.sqrt(1 / (2 * np.sqrt(kvec * mass[0]) * np.tanh(0.5 * beta * np.sqrt(kvec / mass[0]))))
    #dP0 = np.sqrt(mass[0] / beta)
    #nucR = rng.normal(R0, dR0, (nbds, nnuc))
    #nucP = rng.normal(0.0, dP0, (nbds, nnuc))
    nucR, nucP = _sample_nvt(cfg, rng)

    mash_ = mash_rpmd.mash_rpmd(
        nstates=nstates, nnuc=nnuc, nbds=nbds, beta=beta, mass=mass,
        potype='ET_with_bath', potparams=potparams,
        nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        langevin=cfg['langevin'], langevin_params=cfg['langevin_params'],
        seed=seed,                              # <-- makes the GLE noise stream reproducible too
    )

    # reactant electronic state (spin down)
    mash_.mapSx = np.zeros((nbds))
    mash_.mapSy = np.zeros((nbds))
    mash_.mapSz = -np.ones((nbds))

    mash_.run_dynamics(Nsteps, Nprint, delt, cfg['intype'], init_time=0.0, small_dt_ratio=1)
