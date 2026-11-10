"""
gle_run.py -- the driver of the GLE thermalization test: ONE trajectory per call.

    1. sample (R,P) from the NVT distribution of the donor well (therm_scan._sample_nvt: the
       ring-polymer normal-mode widths include the GLE friction term m*gamma*omega_k);
    2. run BO-RPMD (therm_scan._build_bo: bo_rpmd on the lower adiabat) with the generalized Langevin
       thermostat for total_time, writing every Nprint steps.

No thermalization stage and no scan points: the shards go straight to ./shards/, and the whole
trajectory (from t = 0) is the test -- analyze_gle.py shows how fast it equilibrates.

The sampling and object construction are imported from ../therm_scan.py, so this run uses exactly
the production code path.
"""

import os
import importlib.util

import numpy as np


def _therm_scan():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'therm_scan.py')
    spec = importlib.util.spec_from_file_location('_gle_therm_scan', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ts = _therm_scan()


def run_trajectory(cfg, seed):
    rng = np.random.default_rng(seed)
    nucR, nucP = _ts._sample_nvt(cfg, rng)

    pot = [cfg['kvec'], float(cfg['epsil']), cfg['lbd'], 10**float(cfg['log10_delta']),
           np.asarray(cfg['bath'], float)]
    m = _ts._build_bo(cfg, pot, nucR, nucP, seed)

    Nsteps = round(cfg['total_time'] / cfg['delt'])
    m.run_dynamics(Nsteps, cfg['Nprint'], cfg['delt'], cfg['intype'], init_time=0.0, small_dt_ratio=1)
    if os.path.exists('memK.dat'):          # the friction kernel: recomputable, ~MB of text
        os.remove('memK.dat')


def shard_attrs(cfg):
    """workflow.shards hook: the thermostat settings, in every shard's root attrs."""
    lp = cfg['langevin_params']
    return {'Tmem': float(lp['Tmem']), 'Tfluc': float(lp['Tfluc']), 'epsil': float(cfg['epsil']),
            'log10_delta': float(cfg['log10_delta'])}
