"""
config.py -- everything the workflow and the physics need for THIS run, in one dict.

The workflow reads the orchestration keys (n_traj, base_seed, grid_dir, ncores, slurm, system_file);
your traj.py reads whatever physics keys you put here. Keep model parameters here (not hardcoded
in traj.py) so a parameter scan is just a copy of this file with a few numbers changed.
"""

import numpy as np

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'     : 25,             # number of independent trajectories to run
    'base_seed'  : None,       # int -> reproducible; None -> fresh OS entropy each trajectory
    'grid_dir'   : 'traj_grid',    # subdir (relative to this file) holding traj0/, traj1/, ...
    'ncores'     : 13,             # local cores for `python -m workflow.runner`
    'system_file': 'traj.py',    # the module providing run_trajectory(cfg, seed)

    # SLURM knobs, used only by `python -m workflow.slurm` (safe to ignore locally)
    'slurm': {
        'max_tasks'     : 475,
        'job_name'      : 'rpmash_grid',
        'time'          : '06:00:00',
        'cpus_per_task' : 1,
        'max_concurrent': None,        # e.g. 50 -> "--array=0-N%50"
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem'           : '16GB',
        'python'        : 'python',
        'extra_directives': [],        # raw "#SBATCH ..." lines if you need something uncommon
        'extra_commands': [
            'eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"',
            'conda activate map-rpmd'],
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
    'lbd'   : 12.0,
    'delta' : 10**(-1.4),
    'bath'  : [0, 1.0, 2.0, 2.0],     # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # integrator / thermostat
    'intype'         : 'vv',
    'delt'           : 0.0005,
    'total_time'     : 5.0,        # a.u.; Nsteps = total_time / delt
    'Nprint'         : 1,
    'langevin'       : 'generalized',
    'langevin_params': {'gamma': 0.00, 'Tmem': 0.005, 'Tfluc': 0.005},
    #'langevin_params': {'gamma': 2.00, 'Tmem': 10.0, 'Tfluc': 250.0},

    # part-1 equilibration (modified model params) -- build a valid memP + colored-noise state
    # before production, then carry (R,P,spin,memP,Ffluci,Fdiss,noise offset) into part 2.
    'equil_time'  : 0.01,   # a.u.; make >= a few * Tmem (=10) so memP fully fills and (R,P) relax
    'delta_equil' : 10**(-7.0),    # 0 => NAC=0 => spin frozen on the donor while the memory builds
    'epsil_equil' : -100.0,    # driving force during equilibration
}
