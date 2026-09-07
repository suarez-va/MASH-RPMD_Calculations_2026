"""
config.py -- everything the workflow and the physics need for THIS run, in one dict.

The workflow reads the orchestration keys (n_traj, base_seed, grid_dir, ncores, slurm, system_file);
your traj.py reads whatever physics keys you put here. Keep model parameters here (not hardcoded
in traj.py) so a parameter scan is just a copy of this file with a few numbers changed.
"""

import numpy as np

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'     : 100,             # number of independent trajectories to run
    'base_seed'  : None,       # int -> reproducible; None -> fresh OS entropy each trajectory
    'grid_dir'   : 'traj_grid',    # subdir (relative to this file) holding traj0/, traj1/, ...
    'ncores'     : 13,             # local cores for `python -m workflow.runner`
    'system_file': 'traj.py',    # the module providing run_trajectory(cfg, seed)

    # SLURM knobs, used only by `python -m workflow.slurm` (safe to ignore locally)
    'slurm': {
        'job_name'      : 'rpmash_grid',
        'time'          : '02:00:00',
        'cpus_per_task' : 1,
        'max_concurrent': None,        # e.g. 50 -> "--array=0-N%50"
        'partition'     : None,
        'account'       : None,
        'mem'           : None,
        'python'        : 'python',
        'extra_directives': [],        # raw "#SBATCH ..." lines if you need something uncommon
    },

    # ----------------------------------------------------------------- physics (read by traj.py)
    'nbds'    : 8,
    'nnuc'    : 1,
    'nstates' : 2,
    'mass'    : [1.0],
    'beta'    : 1.0,

    # Marcus ET model (atomic units)
    'kvec'  : 4.0,
    'epsil' : 0.0,
    'lbd'   : 500.0,
    'delta' : 10 ** (-5),
    'bath'  : [0, 1.0, 2.0, 2.0],     # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # integrator / thermostat
    #'intype'         : 'vv',
    'intype'         : 'spin_magnus_adaptive',
    'delt'           : 0.002,
    'total_time'     : 25.,        # a.u.; Nsteps = total_time / delt
    'Nprint'         : 1,
    'langevin'       : 'generalized',
    'langevin_params': {'gamma': 2.00, 'Tmem': 10.00, 'Tfluc': 100.0},
}
