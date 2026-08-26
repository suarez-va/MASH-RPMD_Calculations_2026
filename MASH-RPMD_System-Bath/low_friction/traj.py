"""
traj.py -- two-part per-trajectory driver (analytic NVT sample -> equilibrate -> production).

Part 1 (equilibration): analytically sample (R,P) from the NVT distribution, then run the GLE with
the MODIFIED model parameters (epsil_equil, delta_equil; delta_equil=0 freezes the spin on the donor)
for equil_time. This fills the memory buffer memP with a valid history and lets (R,P) relax. The full
dynamical state -- (R,P), spin, memP, the noise trajectory Ffluci, the dissipative force Fdiss, and
the noise step offset -- is written to checkpoint.npz.

Part 2 (production): reload checkpoint.npz and continue the SAME trajectory with the REAL parameters
(epsil, delta). Because memP/Ffluci/Fdiss and the noise phase are carried across, production starts
exactly as if the NVT had been run and the coupling then switched on (verified bit-exact when the
parameters are unchanged).
"""

import os
import numpy as np
import normal_mode
import mash_rpmd   # via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)


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


def run_trajectory(cfg, seed):
    rng  = np.random.default_rng(seed)
    nbds = cfg['nbds']; delt = cfg['delt']
    kvec, lbd = cfg['kvec'], cfg['lbd']; bath = np.asarray(cfg['bath'], float)

    nucR, nucP = _sample_nvt(cfg, rng)

    # ---- Part 1: equilibration with the modified model params -> build memP / Ffluci ----
    pot_e = [kvec, cfg.get('epsil_equil', cfg['epsil']), lbd, cfg.get('delta_equil', 0.0), bath]
    m1 = _build(cfg, pot_e, nucR, nucP, seed)
    m1.mapSx = np.zeros(nbds); m1.mapSy = np.zeros(nbds); m1.mapSz = -np.ones(nbds)   # donor
    Nburn = round(cfg['equil_time'] / delt)
    m1.run_dynamics(Nburn, 10**9, delt, cfg['intype'], init_time=0.0, small_dt_ratio=1)

    # Cyclically shift the noise so index 0 is the continuation point (where equilibration ended,
    # Nburn steps in); production then reads Ffluci from index 0 with no phase-offset input.
    period = m1.integ.Ffluci.shape[1]
    Ffluci = np.roll(m1.integ.Ffluci, -(Nburn % period), axis=1)
    np.savez('checkpoint.npz',
             R=m1.nucR, P=m1.nucP, Sx=m1.mapSx, Sy=m1.mapSy, Sz=m1.mapSz,
             memP=m1.integ.memP, Ffluci=Ffluci)

    # ---- Part 2: production with the REAL params, continuing from the checkpoint ----
    ck    = np.load('checkpoint.npz')
    pot_p = [kvec, cfg['epsil'], lbd, cfg['delta'], bath]
    m2 = _build(cfg, pot_p, ck['R'].copy(), ck['P'].copy(), seed,
                init_memP=ck['memP'].copy(), init_Ffluci=ck['Ffluci'].copy())
    m2.init_map_spin(init_state=0)
    #m2.mapSx = ck['Sx'].copy(); m2.mapSy = ck['Sy'].copy(); m2.mapSz = ck['Sz'].copy()
    Nprod = round(cfg['total_time'] / delt)
    m2.run_dynamics(Nprod, cfg['Nprint'], delt, cfg['intype'], init_time=0.0, small_dt_ratio=1)

    # Production done: remove large per-trajectory scratch files not needed for analysis.
    # (checkpoint.npz holds memP/Ffluci; memK.dat is the friction kernel; nucP not used downstream.)
    for _f in ('checkpoint.npz', 'memK.dat'):
        if os.path.exists(_f):
            os.remove(_f)
