"""
    python -m workflow.runner --config config.py                             # Local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh    # HPC
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'      : 1000000,                    # number of independent trajectories to run
    'base_seed'   : None,                       # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'      : 10,                         # local cores for `python -m workflow.runner`
    'shard_size'  : 100,                        # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)
    'shard_dir'   : 'shards',
    'system_file' : '../../../therm_scan.py',
    'grid_dir'    : 'traj_grid',

    'slurm': {
        'max_tasks'     : 500,
        'job_name'      : 'MASHRPMD',
        'time'          : '29:00:00',
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
    'nbds'              : 12,                    # number of ring-polymer beads
    'jump_index'        : [],                    # unused
    'jump_gap'          : 5,                     # jump once on leaving |V1 - V0| < Egap; None = off
    'full_jump'         : False,                 # half sphere sample 

    # ----------------------------------------------------------------- Marcus ET model
    'nnuc'              : 1,
    'nstates'           : 2,
    'mass'              : [1.0],                 # mass
    'beta'              : 1.0,                   # inverse temperature
    'kvec'              : 16.0,                   # marcus parabola curvature
    'lbd'               : 12.0,                  # reorganization energy
    'epsil'             : 0.0,                   # driving force
    'log10_delta_scan'  : [-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],    # diabatic coupling strength
    'bath'              : [0, 1.0, 2.0, 2.0],    # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # ------------------------------------------------------------------ thermostat + integrator
    'delt'              : 2e-4,
    'intype'            : 'spin_magnus_adaptive',
    'langevin'          : 'generalized',
    'langevin_params'   : {'gamma': 4.0, 'Tmem': 15.0, 'Tfluc': 35.0},
    'therm_epsil'       : -100.0,
    'therm_log10_delta' : -7.0,
    'therm_time'        : 20.0,
    'traj_time'         : 15.0,
    'Nprint'            : 125,  
}
