"""
config.py -- the GLE thermalization test: many independent BO-RPMD trajectories in the donor well,
thermostatted by the generalized Langevin equation, to check that the GLE samples the right
equilibrium distribution and dynamics (analyze_gle.py).

Inherits ../config_base.py (orchestration, slurm, mass, beta, kvec, lbd, bath).  With epsil = -100
the donor diabat V0 lies 100 below V1, so the lower adiabat IS V0 -- a harmonic well centred at
R0 = -sqrt(lbd/(2 kvec)) with omega = sqrt(kvec/m) -- and the crossing R* = -epsil/sqrt(2 lbd kvec)
is far out of reach.  Every analytic reference in analyze_gle.py then holds exactly.

    python -m workflow.runner --config config.py                    # local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh
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
    # ------------------------------------------------------------------ orchestration
    'n_traj'      : 1000,           # number of independent trajectories to run
    'shard_size'  : 10,             # trajectories per shards/shard_<s>.hdf
    'ncores'      : 5,           # local cores for `python -m workflow.runner`
    'system_file' : 'gle_run.py',
    'grid_dir'    : 'traj_grid',

    # ------------------------------------------------------------------ dynamics
    'dynamics'    : 'bo',            # Born-Oppenheimer on the lower adiabat (RP_MASH bo_rpmd)
    'jump_index'  : [],              # no spin, so no quantum jumps
    'jump_gap'    : None,
    'full_jump'   : True,            # unused by BO

    # ------------------------------------------------------------------ model (donor well only)
    'epsil'       : -100.0,          # driving force: the lower adiabat is the donor parabola
    'log10_delta' : -7.0,            # delta = 10**(-7) ~ 0

    # ------------------------------------------------------------------ ring polymer
    'nbds' : 6,

    # ------------------------------------------------------------------ integrator
    'delt'       : 0.001,
    'total_time' : 30.0,             # a.u.; Nsteps = total_time / delt
    'Nprint'     : 25,               # output every Nprint steps (ACF time resolution = Nprint*delt)

    # ------------------------------------------------------------------ thermostat
    'langevin'        : 'generalized',
    #'langevin_params' : {'gamma': 64.0, 'Tmem': 25.0, 'Tfluc': 30.0},
    'langevin_params' : {'gamma': 2.0, 'Tmem': 15.0, 'Tfluc': 30.0},
})
