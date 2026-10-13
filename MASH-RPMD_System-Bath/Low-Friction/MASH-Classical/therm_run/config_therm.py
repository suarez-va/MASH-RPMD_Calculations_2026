"""
config_therm.py -- level 3 of 3: the THERMALIZATION stage for this tree.

Inherits ../config_method.py and only names the stage.  Runs part 1 (therm.py) in workflow.shards
mode: each trajectory's restart state is stored as a row of <tree>/therm_shards/shard_<s>.hdf, and
every epsil_*/ run in this tree reads its trajectory's row straight from those shards (no merge
step).  Run this stage to completion BEFORE submitting any epsil_* run.

    python -m workflow.slurm --config config_therm.py  &&  sbatch submit_grid.sh

Note there is deliberately no `epsil` key here: part 1 runs with therm_epsil, so a production
driving force would be meaningless at this stage.
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
    'grid_dir'   : 'therm_grid',
    'system_file': '../../../therm.py',
    # restart states go to workflow.shards files <tree>/therm_shards/shard_<s>.hdf (shard_size from
    # config_base); traj.py's default therm_dir '../therm_shards' points at the same place.  For
    # scratch, set the SAME absolute path here and as 'therm_dir' in ../config_method.py.
    'shard_dir'  : '../therm_shards',
})
