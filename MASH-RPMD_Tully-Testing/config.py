"""
config.py -- the base config of MASH-RPMD_Tully-Testing: the Tully single avoided crossing run as a
sequence of STAGES, each restarting from a resampled density operator instead of carrying the
quantum-jump weights along every trajectory.

The physics (potential, mass, integrator) is MASH-RPMD_Tully-Delta's.  What is new is the initial
condition, a density operator with four phase-space components in the adiabatic basis,

    rho(q,p) = rho_I(q,p) I + rho_z(q,p) sigma_z + rho_x(q,p) sigma_x + rho_y(q,p) sigma_y

(sigma_z = +1 on the upper adiabat, i.e. S_z > 0), and the stage loop driven by iterate.py:

    stage m:  rho^(m-1) as n_points signed phase-space points, each carrying (w_I, w_z, w_x, w_y)
              -> n_spin full-sphere spin samples per point, run for seg_time
              -> every endpoint weighted with the 16-term update (see resample.py)
              -> histogram the signed weights in (q,p) and resample n_points new points = rho^(m)

This file is NOT run directly: iterate.py writes stage_<m>/config.py, which inherits everything here
and fixes the per-stage keys (stage, t_start, n_traj, base_seed, init_file).

    python iterate.py local  --config config.py --ncores 14     # every stage, locally
    python iterate.py chain  --config config.py                 # every stage, as a SLURM chain
    python analyze_testing.py --config config.py                # stitched observables (from summary.npz)
    python analyze_tully_rho.py --config config.py --ncores 8   # C_Rg, C_Tg, C_Re, C_Te from the shards
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'    : None,        # set per stage by iterate.py: n_points * n_spin
    'base_seed' : None,        # int -> reproducible (per-stage seeds derive from it); None -> OS entropy
    'ncores'    : 10,           # local cores for `python -m workflow.runner` (iterate.py local --ncores)
    'shard_size': 2000,        # trajectories per stage_<m>/shards/shard_<s>.hdf (REQUIRED: traj.py needs idx)
    'shard_dir' : 'shards',

    'slurm': {
        'max_tasks'     : 50,
        'job_name'      : 'MASHTEST',
        'time'          : '0:59:00',
        'cpus_per_task' : 4,
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
    # the single-task job that resamples one stage and submits the next (iterate.py chain)
    'resample_slurm': {
        'time'          : '0:59:00',
        'cpus_per_task' : 1,
        'mem'           : '16GB',
    },

    # ----------------------------------------------------------------- stage wiring
    'grid_dir'   : 'traj_grid',
    'system_file': 'traj.py',   # stage configs point this at ../traj.py

    # ----------------------------------------------------------------- system
    'nnuc'    : 1,             # tully_1ac is a single-nuclear-DOF model (enforced by error_check)
    'nstates' : 2,
    'mass'    : [2000.00],
    'beta'    : 1.00,

    # ----------------------------------------------------------------- Tully single avoided crossing (a.u.)
    # V11 =  A*tanh(B*R),  V22 = -V11,  V12 = V21 = C*exp(-D*R^2)
    'A' : 0.01,
    'B' : 1.60,
    'C' : 0.005,
    'D' : 1.00,

    # ----------------------------------------------------------------- ring polymer / integrator
    'nbds'  : 1,               # every bead starts at the sampled point (centroid is binned)
    'delt'  : 5.0,
    'intype': 'spin_magnus_adaptive',
    'Nprint': 2,              # print every Nprint steps within a stage (the last step is always printed)

    # ----------------------------------------------------------------- stages
    'n_stages'  : 100,           # stages 0 .. n_stages-1
    'seg_time'  : 10.0,      # a.u. per stage (t_m = m * seg_time); seg_time / delt steps
    'n_points'  : 100,         # phase-space points per stage (per nonzero component if sharing='separate')
    'n_spin'    : 1000,        # full-sphere spin samples per point -> n_traj = n_points * n_spin
    'sharing'   : 'shared',    # 'shared': each point carries all four weights; 'separate': one-hot points per component
    'bins'      : [64, 64],    # (q, p) histogram for the signed-weight cancellation; None -> no binning
    'placement' : 'endpoint',  # where a draw from a bin lands: 'endpoint' (a real endpoint), 'uniform', 'center'
    'n_pool_init': 100000,     # samples per nonzero component drawn from rho_init before resampling stage 0

    # ----------------------------------------------------------------- initial density operator
    # Per component: None (zero) or {'coef', 'q0', 'p0', 'ga'}, meaning coef * g(q,p) with g a
    # normalized delta(q-q0) delta(p-p0) when ga is None, else the Gaussian of MASH-RPMD_Tully
    # (q ~ N(q0, 1/(2 ga)), p ~ N(p0, ga/2)).  Tr rho = 2 int rho_I, so coef_I = 1/2 is normalized.
    # Default: the LOWER adiabat at (-10, 10), as in Tully-Delta:  |g><g| = (I - sigma_z)/2.
    'rho_init': {
        'I': {'coef':  0.5, 'q0': -2.0, 'p0': 7.0, 'ga': None},
        'z': {'coef': -0.5, 'q0': -2.0, 'p0': 7.0, 'ga': None},
        'x': None,
        'y': None,
    },
}
