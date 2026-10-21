"""
config_method.py -- level 4: the physics specific to the quantum (ring-polymer) tree of this scan.

Config hierarchy (each level inherits the one above and overrides/adds a few keys):

    ../../../config_base.py      orchestration + shared physics
    ../../config_friction.py     gamma (the friction regime)
    ../config_scan.py            the scan list (epsil_scan or delta_scan)
    config_method.py             <- you are here: nbds, timings, thermostat

Same values as MASH-RPMD_System-Bath/Low-Friction/MASH-Quantum/config_method.py.  The values that
differ between the Classical and Quantum copies are nbds, therm_time, traj_time, Nprint and the
memory/noise lengths Tmem / Tfluc.
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


CONFIG = _inherit('../config_scan.py')

CONFIG.update({
    # ------------------------------------------------------------------ driver
    # one thermalization, then a production run per scan point (see therm_scan.py)
    'system_file' : '../../../therm_scan.py',
    'grid_dir'    : 'traj_grid',

    # ------------------------------------------------------------------ ring polymer
    'nbds' : 6,                      # ring polymer: 6 beads

    # ------------------------------------------------------------------ integrator
    'delt' : 0.001,

    # ------------------------------------------------------------------ thermalization
    # Modified model params used only while the memory buffer fills and (R,P) relax.
    'therm_time'  : 25.0,          # a.u. of GLE burn-in; >= a few * Tmem so memP fully fills
    'therm_delta' : 10**(-7.0),    # ~0 => NAC~0 => spin frozen on the donor during the burn-in
    'therm_epsil' : -100.0,        # driving force during the burn-in

    # ------------------------------------------------------------------ production
    'traj_time' : 5.0,            # a.u.; Nsteps = traj_time / delt
    'Nprint'    : 25,

    # ------------------------------------------------------------------ thermostat
    'langevin' : 'generalized',
})

# gamma comes from ../../config_friction.py so the friction constant is written down exactly once.
CONFIG['langevin_params'] = {'gamma': CONFIG['gamma'], 'Tmem': 25.0, 'Tfluc': 30.0}
