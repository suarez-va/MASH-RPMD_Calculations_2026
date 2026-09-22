"""
config_traj.py -- level 3 of 3: the PRODUCTION stage at epsil = 0.00.

Inherits ../config_method.py and names the stage plus the one value that makes this directory
different from its eight siblings: `epsil`.  Keep it equal to the number in the directory name.

Part 1 is NOT run here -- each trajectory restarts from ../thermalize/therm<idx>.hdf, so that
stage must have finished first.

    python -m workflow.slurm --config config_traj.py  &&  sbatch submit_grid.sh
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


CONFIG = _inherit('../config_method.py')

CONFIG.update({
    'grid_dir'   : 'traj_grid',
    'system_file': '../../../traj.py',

    'epsil' : 20.00,
})
