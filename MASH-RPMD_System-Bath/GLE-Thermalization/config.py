"""
    python -m workflow.runner --config config.py                             # Local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh    # HPC
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'      : 10000,                      # number of independent trajectories to run
    'base_seed'   : None,                       # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'      : 10,                         # local cores for `python -m workflow.runner`
    'shard_size'  : 10,                         # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)
    'shard_dir'   : 'shards',
    'system_file' : 'gle_run.py',
    'grid_dir'    : 'traj_grid',

    'slurm': {
        'max_tasks'     : 100,
        'job_name'      : 'GLE-RPMD',
        'time'          : '09:00:00',
        'cpus_per_task' : 10,
        'max_concurrent': None,
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem_per_cpu'   : '8GB', 
        'python'        : 'python',
        'extra_directives': [],
        'extra_commands': [
            'eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"',
            'conda activate map-rpmd'],
    },

    # ------------------------------------------------------------------ dynamics
    'dynamics'          : 'bo',                  # Born-Oppenheimer on the lower adiabat
    'nbds'              : 12,                    # number of ring-polymer beads
    'jump_index'        : [],                    # unused by BO
    'jump_gap'          : None,                  # unused by BO
    'full_jump'         : False,                 # unused by BO

    # ----------------------------------------------------------------- Marcus ET model
    'nnuc'              : 1,
    'nstates'           : 2,
    'mass'              : [1.0],                 # mass
    'beta'              : 1.0,                   # inverse temperature
    #'kvec'              : 2.0,                   # marcus parabola curvature
    #'kvec'              : 2.0,                   # marcus parabola curvature
    'kvec'              : 4.0,                   # marcus parabola curvature
    'lbd'               : 12.0,                  # reorganization energy
    'epsil'             : -100.0,                # driving force
    'log10_delta'       : -7.0,                  # diabatic coupling strength
    'bath'              : [0, 1.0, 2.0, 2.0],    # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # ------------------------------------------------------------------ thermostat + integrator
    #'delt'              : 5e-4,
    #'delt'              : 5e-4,
    'delt'              : 2e-4,
    'intype'            : 'spin_magnus_adaptive',
    'langevin'          : 'generalized',
    #'langevin_params'   : {'gamma': 2.0, 'Tmem': 15.0, 'Tfluc': 35.0},
    #'langevin_params'   : {'gamma': 64.0, 'Tmem': 15.0, 'Tfluc': 35.0},
    'langevin_params'   : {'gamma': 4.0, 'Tmem': 15.0, 'Tfluc': 35.0},
    'total_time'        : 35.0,
    'Nprint'            : 50,  
}
