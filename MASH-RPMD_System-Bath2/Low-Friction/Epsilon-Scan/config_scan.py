"""
config_scan.py -- level 3: the driving-force scan of the Low-Friction regime.

Inherits ../config_friction.py (and through it ../../config_base.py) and lists the epsil values
to scan.  `epsil_scan` is always a list, even for a single point.  The jumps are shared by every
scan point and come from ONE trigger: `jump_index` (the production step indices at which quantum
jumps fire; [] = none) or `jump_gap` (Egap: each trajectory jumps once every time it leaves the
region |V1 - V0| < Egap -- meant for full_jump=False; None = off).  `delta` is held fixed.
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
    'epsil_scan' : [0.00, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0],
    'jump_index' : [],                  # k -> jump at t = k*delt of production; [] = no jumps
    'jump_gap'   : None,                # Egap (a.u.): jump once on leaving |V1 - V0| < Egap; None = off, needs jump_index []

    # the parameter this scan holds fixed
    'delta'      : 10**(-1.4),
})
