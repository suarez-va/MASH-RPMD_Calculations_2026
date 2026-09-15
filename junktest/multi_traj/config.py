"""
config.py -- everything the workflow and the physics need for THIS run, in one dict.

The workflow reads the orchestration keys (n_traj, base_seed, grid_dir, ncores, slurm, system_file);
your traj.py reads whatever physics keys you put here. Keep model parameters here (not hardcoded
in traj.py) so a parameter scan is just a copy of this file with a few numbers changed.
"""

import numpy as np

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'     : 2500,             # number of independent trajectories to run
    'base_seed'  : None,       # int -> reproducible; None -> fresh OS entropy each trajectory
    'grid_dir'   : 'traj_grid',    # subdir (relative to this file) holding traj0/, traj1/, ...
    'ncores'     : 1,             # local cores for `python -m workflow.runner`
    'system_file': 'traj.py',    # the module providing run_trajectory(cfg, seed)

    # SLURM knobs, used only by `python -m workflow.slurm` (safe to ignore locally)
    'slurm': {
        'max_tasks'     : 500,
        'job_name'      : 'rpmash_grid',
        'time'          : '9:00:00',
        'cpus_per_task' : 1,
        'max_concurrent': None,        # e.g. 50 -> "--array=0-N%50"
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem'           : '8GB',
        'python'        : 'python',
        'extra_directives': [],        # raw "#SBATCH ..." lines if you need something uncommon
        'extra_commands': [
            'eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"',
            'conda activate map-rpmd'],
    },

    # ----------------------------------------------------------------- physics (read by traj.py)
    'nbds'    : 6,
    'nnuc'    : 1,
    'nstates' : 2,
    'mass'    : [1.0],
    'beta'    : 1.0,

    # Marcus ET model (atomic units)
    'kvec'  : 4.0,
    'epsil' : -100.0,
    'lbd'   : 12.0,
    'delta' : 10**(-7),
    'bath'  : [0, 1.0, 64.0, 2.0],     # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # integrator / thermostat
    #'intype'         : 'vv',
    'intype'         : 'spin_magnus_adaptive',
    'delt'           : 0.001,
    'total_time'     : 30.,        # a.u.; Nsteps = total_time / delt
    'Nprint'         : 25,
    'langevin'       : 'generalized',
    'langevin_params': {'gamma': 64.00, 'Tmem': 25.00, 'Tfluc': 30.0},
}
