"""
config_method.py -- level 2 of 3: the physics specific to the classical (1-bead) tree.

Inherits Low-Friction/config_base.py and adds everything that distinguishes this method and fixes
the two run stages' timings.  The ONLY value that differs between the Classical and Quantum copies
of this file is `nbds` (1 here).

Consumed by both stages:
  therm_time / therm_delta / therm_epsil  -> part 1, the shared thermalization (therm.py)
  traj_time  / Nprint                     -> part 2, production               (traj.py)
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
    # ------------------------------------------------------------------ ring polymer
    'nbds' : 1,                      # classical limit: a single bead

    # ------------------------------------------------------------------ integrator
    'delt' : 0.001,

    # ------------------------------------------------------------------ part 1: thermalization
    # Modified model params used only while the memory buffer fills and (R,P) relax.
    'therm_time'  : 1.0,          # a.u. of GLE burn-in; >= a few * Tmem so memP fully fills
    'therm_delta' : 10**(-7.0),    # ~0 => NAC~0 => spin frozen on the donor during the burn-in
    'therm_epsil' : -100.0,        # driving force during the burn-in

    # ------------------------------------------------------------------ part 2: production
    'traj_time' : 15.0,            # a.u.; Nsteps = traj_time / delt
    'Nprint'    : 25,

    # ------------------------------------------------------------------ thermostat
    'langevin' : 'generalized',
})

# gamma comes from config_base.py so the friction constant is written down exactly once.
CONFIG['langevin_params'] = {'gamma': CONFIG['gamma'], 'Tmem': 1.0, 'Tfluc': 16.0}
