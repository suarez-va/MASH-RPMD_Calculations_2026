"""
config_friction.py -- level 2: the High-Friction regime.

Inherits ../config_base.py and sets the friction constant `gamma` for every calculation under
High-Friction/ (Epsilon-Scan/ and Delta-Scan/).
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


CONFIG = _inherit('../config_base.py')

CONFIG.update({
    'gamma' : 64.00,           # friction constant
})
