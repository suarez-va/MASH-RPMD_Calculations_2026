"""
config_scan.py -- level 3: the diabatic-coupling scan of the High-Friction regime.

Inherits ../config_friction.py (and through it ../../config_base.py) and lists the delta values
to scan.  `delta_scan` is always a list, even for a single point, and `Tjump_scan` pairs with
it one-to-one (Tjump_scan[i] is used at delta_scan[i]).  `epsil` is held fixed.
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
    'Tjump_scan' : [45.0],                # one per delta; None = no jumps

    # the parameter this scan holds fixed
    'epsil'      : 0.00,
})
