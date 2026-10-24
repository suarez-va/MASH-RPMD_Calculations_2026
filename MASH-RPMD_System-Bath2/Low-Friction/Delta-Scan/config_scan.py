"""
config_scan.py -- level 3: the diabatic-coupling scan of the Low-Friction regime.

Inherits ../config_friction.py (and through it ../../config_base.py) and lists the delta values
to scan.  `delta_scan` is always a list, even for a single point.  `jump_index` (the production
step indices at which quantum jumps fire; [] = none) is shared by every scan point.  `epsil` is held fixed.
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
    'delta_scan' : [10**(-1.4)],
    'jump_index' : [],                  # k -> jump at t = k*delt of production; [] = no jumps

    # the parameter this scan holds fixed
    'epsil'      : 0.00,
})
