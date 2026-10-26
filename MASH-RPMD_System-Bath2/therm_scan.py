"""
therm_scan.py -- ONE thermalization per trajectory, then a production run at EVERY scan point,
each restarted from that same thermalized state held in memory.  Replaces the old two-stage
therm.py (-> therm_shards/) + traj.py (one run per epsil_* directory) workflow: nothing is written
between the stages, and there is no thermalization bank to keep in sync.

Per trajectory idx (one call of run_trajectory):

  1. sample (R,P) from the NVT distribution and run the GLE with the MODIFIED model parameters
     (therm_epsil, therm_delta) for therm_time, spin frozen on the donor.  This fills the memory
     buffer memP and lets (R,P) relax -- exactly the old therm.py;
  2. for every scan point (name, overrides) from scan_points(cfg): continue that state -- nucR, nucP,
     memP and the cyclically rolled noise Ffluci -- with the point's REAL parameters for traj_time,
     writing the point's .dat files into the subdirectory <name>/ of the trajectory's work dir --
     exactly the old traj.py, restarted from memory instead of from a therm shard.

workflow.shards sees scan_points() and stores point <name> of trajectory idx as a row of
<run_dir>/<name>/shards/shard_<s>.hdf, with the point's own config (epsil, delta) in the
shard's config_json.  So <run_dir>/epsil_3.0/shards/ is an ordinary shard directory, and the
analysis runs inside it unchanged:

    cd epsil_3.0 && python ../../../../analyze_p2.py --file shards --ncores 20

--- the scan --------------------------------------------------------------------------------------

The config has exactly one scan list, epsil_scan or delta_scan.  The quantum jumps follow the same
rule at every point, from ONE of two triggers:

    jump_index  the production step indices k at which a jump fires (at t = k*delt from the start of
                production; [] = no jumps) -- the same times for every trajectory;
    jump_gap    Egap: each trajectory arms a jump whenever the diabatic gap |V1 - V0| at its centroid
                drops below Egap and fires it once, on the first step back at |V1 - V0| >= Egap
                (None = off; needs jump_index = []).  Meant for full_jump=False, whose hemisphere
                resample is only safe away from the crossing.

weights.dat is written either way.  The parameter that is not scanned is the fixed config value
(`delta` in an Epsilon-Scan, `epsil` in a Delta-Scan).  Point directories:

    epsil_scan -> epsil_<value>          e.g. epsil_0.0, epsil_3.0, epsil_12.25
    delta_scan -> delta_1e<log10 value>  e.g. delta_1e-1.40, delta_1e-2.00

A one-element list is a one-point scan, and works the same way.

--- random numbers ---------------------------------------------------------------------------------

The trajectory seed is split into a thermalization stream and a production stream.  The production
stream is the SAME at every scan point (same initial spin sample and same jump resamples), as in the old
workflow, where every epsil_* directory used the same seed for trajectory k: correlated sampling
across the scan, so differences between points are less noisy.  base_seed=None still draws the
production seed once per trajectory and shares it across the points.

Usage (from a MASH-Classical / MASH-Quantum dir, whose config_method.py sets system_file here):
    python -m workflow.runner --config config_method.py                    # local
    python -m workflow.slurm  --config config_method.py && sbatch submit_grid.sh
"""

import os
import math

import numpy as np
import normal_mode
import mash_rpmd   # via PYTHONPATH (= RP_MASH root)


# ------------------------------------------------------------------ the scan
def _point_name(key, value):
    if key == 'epsil':
        return f'epsil_{float(value)!r}'
    return f'delta_1e{math.log10(float(value)):.2f}'


def scan_points(cfg):
    """
    workflow.shards hook: [(directory name, config overrides)] for every scan point, in list order.
    Exactly one of epsil_scan / delta_scan; the jumps (jump_index or jump_gap) are shared by every point.
    """
    stale = [k for k in ('Tjump_scan', 'Tjump') if k in cfg]
    if stale:
        raise KeyError(f'{stale} are no longer used: quantum jumps now fire at the predetermined '
                       f'production steps in `jump_index` (e.g. [] or [100, 200]), shared by every point')
    if cfg.get('jump_gap') is not None and list(cfg.get('jump_index', [])):
        raise ValueError(f"jump_gap={cfg['jump_gap']} and jump_index={list(cfg['jump_index'])} are two "
                         f"different jump triggers: set jump_index = [] to use jump_gap")
    keys =[k for k in ('epsil', 'delta') if f'{k}_scan' in cfg]
    if len(keys) != 1:
        raise KeyError(f'the config needs exactly one of epsil_scan / delta_scan, found '
                       f'{[k + "_scan" for k in keys]}')
    key = keys[0]
    fixed = 'delta' if key == 'epsil' else 'epsil'
    if fixed not in cfg:
        raise KeyError(f'a {key} scan needs a fixed `{fixed}` in the config')
    values = list(cfg[f'{key}_scan'])
    if not values:
        raise ValueError(f'{key}_scan is empty')
    if key == 'delta' and min(values) <= 0.0:
        raise ValueError('delta_scan values must be > 0 (the directories are named by log10)')
    return [(_point_name(key, v), {key: float(v), fixed: float(cfg[fixed])}) for v in values]


# ------------------------------------------------------------------ thermalization (old therm.py)
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


def _build(cfg, potparams, nucR, nucP, seed, jump_index=None, jump_gap=None, **restart):
    return mash_rpmd.mash_rpmd(
        nstates=cfg['nstates'], nnuc=cfg['nnuc'], nbds=cfg['nbds'],
        beta=cfg['beta'], mass=np.asarray(cfg['mass'], float),
        potype='ET_with_bath', potparams=potparams, nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        langevin=cfg['langevin'], langevin_params=cfg['langevin_params'],
        jump_index=jump_index, full_jump=cfg.get('full_jump', True), jump_gap=jump_gap, seed=seed,
        verbose=False, **restart)


def thermalize(cfg, seed):
    """Run the burn-in; return the restart state (nucR, nucP, memP, Ffluci rolled to its end)."""
    nbds = cfg['nbds']; delt = cfg['delt']
    rng = np.random.default_rng(seed)
    nucR, nucP = _sample_nvt(cfg, rng)

    pot_e = [cfg['kvec'], float(cfg['therm_epsil']), cfg['lbd'], float(cfg['therm_delta']),
             np.asarray(cfg['bath'], float)]
    m1 = _build(cfg, pot_e, nucR, nucP, seed)
    m1.mapSx = np.zeros(nbds); m1.mapSy = np.zeros(nbds); m1.mapSz = -np.ones(nbds)   # donor
    Nburn = round(cfg['therm_time'] / delt)
    m1.run_dynamics(Nburn, 10**9, delt, cfg['intype'], init_time=0.0, small_dt_ratio=1)

    # Cyclically shift the noise so index 0 is the continuation point (where the burn-in ended,
    # Nburn steps in); production then reads Ffluci from index 0 with no phase-offset input.
    period = m1.integ.Ffluci.shape[1]
    state = dict(nucR=np.array(m1.nucR, dtype=np.float64), nucP=np.array(m1.nucP, dtype=np.float64),
                 memP=np.array(m1.integ.memP, dtype=np.float64),
                 Ffluci=np.roll(m1.integ.Ffluci, -(Nburn % period), axis=1).astype(np.float64))

    # spin_magnus_init rebuilds Fdiss from memP BEFORE re-seeding memP[:,:,-1] from nucP, so the two
    # must agree -- otherwise every production run would restart silently wrong
    for i in range(cfg['nnuc']):
        if not np.allclose(state['memP'][:, i, -1],
                           normal_mode.real_to_normal_mode(state['nucP'][:, i]), rtol=0.0, atol=1e-12):
            raise RuntimeError('thermalized memP[:,:,-1] does not match the normal-mode nucP')
    return state, Nburn


# ------------------------------------------------------------------ one trajectory
def _seeds(seed):
    """(thermalization seed, production seed) from the trajectory seed; concrete ints even for
    seed=None, so the production stream can be shared by every scan point."""
    therm, prod = np.random.SeedSequence(seed).spawn(2)
    return (int(therm.generate_state(1, dtype=np.uint32)[0]),
            int(prod.generate_state(1, dtype=np.uint32)[0]))


def run_trajectory(cfg, seed, idx=None, run_dir=None):
    points = scan_points(cfg)
    therm_seed, prod_seed = _seeds(seed)
    state, Nburn = thermalize(cfg, therm_seed)
    print(f'[therm_scan] traj {idx}: thermalized ({Nburn} steps), running {len(points)} '
          f'scan point(s)', flush=True)

    Nprod = round(cfg['traj_time'] / cfg['delt'])
    bath = np.asarray(cfg['bath'], float)
    home = os.getcwd()
    for name, over in points:
        os.makedirs(name)
        os.chdir(name)
        try:
            pot = [cfg['kvec'], over['epsil'], cfg['lbd'], over['delta'], bath]
            # fresh copies per point: the integrator rolls memP and the production mutates nucR/nucP
            m = _build(cfg, pot, state['nucR'].copy(), state['nucP'].copy(), prod_seed,
                       jump_index=cfg.get('jump_index', []), jump_gap=cfg.get('jump_gap'),
                       init_memP=state['memP'].copy(),
                       init_Ffluci=state['Ffluci'].copy())
            m.init_map_spin(init_state=None)
            m.run_dynamics(Nprod, cfg['Nprint'], cfg['delt'], cfg['intype'],
                           init_time=0.0, small_dt_ratio=1)
            if os.path.exists('memK.dat'):          # the friction kernel: recomputable, ~MB of text
                os.remove('memK.dat')
        finally:
            os.chdir(home)


def shard_attrs(cfg):
    """workflow.shards hook: the run-constant thermalization metadata, in every point's shards."""
    Nmem, Nfluc, period = buffer_sizes(cfg)
    lp = cfg['langevin_params']
    Nburn = round(cfg['therm_time'] / cfg['delt'])
    return {'Tmem': float(lp['Tmem']), 'Tfluc': float(lp['Tfluc']),
            'Nmem': int(Nmem), 'Nfluc': int(Nfluc), 'period': int(period),
            'Nburn': int(Nburn), 'noise_offset': int(Nburn % period),
            'therm_time': float(cfg['therm_time']), 'therm_epsil': float(cfg['therm_epsil']),
            'therm_delta': float(cfg['therm_delta'])}
