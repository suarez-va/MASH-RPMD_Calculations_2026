"""
config_scan.py -- level 3: the diabatic-coupling scan of the Low-Friction regime.

Inherits ../config_friction.py (and through it ../../config_base.py) and lists the diabatic couplings
to scan, as log10(delta): `log10_delta_scan` is always a list, even for a single point, and point
v runs in delta_1e<v>/ with delta = 10**v.  `epsil` is held fixed.  The jumps are shared by every
scan point and come from ONE trigger: `jump_index` (the production step indices at which quantum
jumps fire; [] = none) or `jump_gap` (Egap: each trajectory jumps once every time it leaves the
region |V1 - V0| < Egap -- meant for full_jump=False; None = off).  This scan uses the energy-gap
trigger with hemisphere jumps (full_jump=False, jump_gap=5).  BO-Classical/ ignores the jumps (it has
no spin) and sets them off itself.
"""

import os
import copy
import importlib.util


def _inherit(relpath):
    """Load the parent config's CONFIG from a path relative to THIS file."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), relpath)
    name = '_cfg_' + os.path.splitext(os.path.basename(p))[0]
    spec = importlib.util.spec_from_file_location(name, p)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return copy.deepcopy(mod.CONFIG)


CONFIG = _inherit('../config_friction.py')

CONFIG.update({
    'n_traj'    : 1000000,      # number of independent trajectories to run
    'shard_size': 100,        # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)

    'log10_delta_scan' : [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'full_jump'  : False,               # hemisphere jumps: resample within the current adiabat
    'jump_index' : [],                  # k -> jump at t = k*delt of production; [] = no jumps
    'jump_gap'   : 5.0,                 # Egap (a.u.): jump once on leaving |V1 - V0| < Egap; None = off, needs jump_index []

    # the parameter this scan holds fixed
    'epsil'      : 0.00,
})
