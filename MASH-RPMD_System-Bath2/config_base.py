"""
config_base.py -- the top level: everything shared by EVERY run under MASH-RPMD_System-Bath2/.

Config hierarchy (each level inherits the one above and overrides/adds a few keys):

    config_base.py                  <- you are here: orchestration + shared physics
      +-- Low-Friction/             one friction regime
      +-- High-Friction/            the other

Put a value HERE if it is the same for both friction regimes, both methods and every epsil point.
`delta` (the diabatic coupling) and `gamma` (the friction constant) are deliberately NOT here: they
are set lower in the hierarchy, in each regime's / method's config, which is also where `gamma` has
to be wired into `langevin_params`.

SLURM memory is requested per CPU (`mem_per_cpu` -> #SBATCH --mem-per-cpu), so it scales with
`cpus_per_task` instead of being one total shared by every trajectory a task runs at once.
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'    : 100000000,      # number of independent trajectories to run
    'base_seed' : None,        # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'    : 1,           # local cores for `python -m workflow.runner`
    'shard_size': 10000,        # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)
    'shard_dir' : 'shards',    # relative to the run dir; merge with `python -m workflow.shards`

    'slurm': {
        'max_tasks'     : 250,
        'job_name'      : 'rpmash_grid',
        'time'          : '24:00:00',
        'cpus_per_task' : 5,
        'max_concurrent': None,
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem_per_cpu'   : '8GB',     # #SBATCH --mem-per-cpu (do not also set 'mem')
        'python'        : 'python',
        'extra_directives': [],
        'extra_commands': [
            'eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"',
            'conda activate map-rpmd'],
    },

    # ----------------------------------------------------------------- system
    'nnuc'    : 1,
    'nstates' : 2,
    'mass'    : [1.00],
    'beta'    : 1.00,

    # ----------------------------------------------------------------- Marcus ET model (a.u.)
    # `epsil` and `delta` are NOT here -- they are set lower in the hierarchy.
    'kvec'  : 4.00,
    'lbd'   : 12.00,
    'bath'  : [0, 1.00, 2.00, 2.00],     # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # ----------------------------------------------------------------- integrator
    'intype'    : 'spin_magnus_adaptive',

    # ----------------------------------------------------------------- quantum jumps
    # The jump trigger is set in each config_scan.py: jump_index (fixed production step indices) or
    # jump_gap (Egap: jump once on leaving |V1 - V0| < Egap, per trajectory -- use it with
    # full_jump=False, so the hemisphere resample happens away from the crossing).
    'full_jump' : True,
}
