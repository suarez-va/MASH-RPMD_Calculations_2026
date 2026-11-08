"""
config_method.py -- level 4: Born-Oppenheimer classical (1-bead) dynamics on the LOWER adiabat, for
the same scan as ../MASH-Classical/.

Config hierarchy (each level inherits the one above and overrides/adds a few keys):

    ../../../config_base.py      orchestration + shared physics
    ../../config_friction.py     gamma (the friction regime)
    ../config_scan.py            the scan list (epsil_scan or log10_delta_scan)
    config_method.py             <- you are here: nbds, timings, thermostat, dynamics='bo'

Everything is the same as ../MASH-Classical/config_method.py (same thermalization, timings and
thermostat) except `dynamics = 'bo'`: the production runs RP_MASH's bo_rpmd, which holds the spin
on the south pole (mapSz = -1), so the nuclei move on the lower adiabatic surface Vbar - Vz and never
hop.  There are no quantum jumps, so the scan's jump settings are switched off here.

The populations come from counting trajectories on each side of the crossing, not from weights:

    cd delta_1e-1.10 && python ../../../../analyze_bo_rate.py --file shards --ncores 20
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
    'dynamics'    : 'bo',              # Born-Oppenheimer on the lower adiabat (RP_MASH bo_rpmd)

    # ------------------------------------------------------------------ no quantum jumps (no spin)
    'jump_index'  : [],
    'jump_gap'    : None,
    'full_jump'   : True,              # unused by BO; reset so the shards do not suggest jumps

    # ------------------------------------------------------------------ ring polymer
    'nbds' : 1,                      # classical limit: a single bead

    # ------------------------------------------------------------------ integrator
    'delt' : 0.001,

    # ------------------------------------------------------------------ thermalization
    # Modified model params used only while the memory buffer fills and (R,P) relax.
    'therm_time'  : 15.0,          # a.u. of GLE burn-in; >= a few * Tmem so memP fully fills
    'therm_log10_delta' : -7.0,    # delta = 10**(-7) ~0 => NAC~0 => spin frozen on the donor during the burn-in
    'therm_epsil' : -100.0,        # driving force during the burn-in

    # ------------------------------------------------------------------ production
    'traj_time' : 20.0,            # a.u.; Nsteps = traj_time / delt
    'Nprint'    : 100,

    # ------------------------------------------------------------------ thermostat
    'langevin' : 'generalized',
})

# gamma comes from ../../config_friction.py so the friction constant is written down exactly once.
CONFIG['langevin_params'] = {'gamma': CONFIG['gamma'], 'Tmem': 15.0, 'Tfluc': 35.0}
