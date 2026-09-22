"""
config_base.py -- level 1 of 3: everything shared by EVERY run under Low-Friction/.

Config hierarchy (each level inherits the one above and overrides/adds a few keys):

    Low-Friction/config_base.py                    &lt;- you are here: orchestration + shared physics
      +-- MASH-{Classical,Quantum}/config_method.py   per-tree physics (nbds, timings, langevin)
            +-- thermalize/config_therm.py            stage: thermalization  (therm.py)
            +-- epsil_*/config_traj.py                stage: production      (traj.py)

Put a value HERE if it is the same for both methods and every epsil point.  Anything that varies
per tree belongs in config_method.py; anything that varies per directory belongs in the leaf.

`gamma` is defined here as a plain top-level key and is wired into langevin_params by
config_method.py, so the friction constant is written down exactly once.
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'    : 100000,      # number of independent trajectories to run
    'base_seed' : None,        # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'    : 1,           # local cores for `python -m workflow.runner`

    'slurm': {
        'max_tasks'     : 500,
        'job_name'      : 'rpmash_grid',
        'time'          : '24:00:00',
        'cpus_per_task' : 1,
        'max_concurrent': None,
        'partition'     : None,
        'account'       : 'gts-jkretchmer3-chemx',
        'mem'           : '8GB',
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
    # `epsil` is NOT here -- it is the one thing each epsil_*/config_traj.py sets for itself.
    'kvec'  : 4.00,
    'lbd'   : 10.00,
    'delta' : 10**(-1.40),
    'bath'  : [0, 1.00, 2.00, 2.00],     # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

    # ----------------------------------------------------------------- integrator / thermostat
    'intype'    : 'spin_magnus_adaptive',
    'Tjump'     : 1.0,
    'full_jump' : False,
    'gamma'     : 2.00,           # friction constant; config_method.py feeds this into langevin_params
}
