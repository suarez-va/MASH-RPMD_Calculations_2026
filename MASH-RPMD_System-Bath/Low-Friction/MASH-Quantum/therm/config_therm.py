"""
config_therm.py -- level 3 of 3: the THERMALIZATION stage for this tree.

Inherits ../config_method.py and only names the stage.  Runs part 1 (therm.py), which appends
each trajectory to a per-SLURM-task shard_*.hdf in therm_dir; consolidate_therm.py then merges the
shards into therm_data.hdf, which every epsil_*/ run in this tree restarts from.  Run this stage AND
the merge to completion BEFORE submitting any epsil_* run.

    python -m workflow.slurm --config config_therm.py  &&  sbatch submit_grid.sh
    python ../../../consolidate_therm.py --dir <therm_dir> --n-traj <n_traj>

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
})
