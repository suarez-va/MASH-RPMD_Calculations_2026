"""
config.py -- the single config for the Tully single-avoided-crossing grid.

MASH-RPMD_System-Bath needs a three-level config hierarchy (config_base -> config_method ->
config_traj) because it runs two stages across two ring-polymer sizes and nine values of `epsil`.
This tree runs one stage at one state point, so everything lives in one flat file.

Relative to Low-Friction/MASH-Classical/epsil_10.0 the differences are:

  * potential  'ET_with_bath' -> 'tully_1ac', so `kvec`/`lbd`/`delta`/`bath`/`epsil` are replaced by
    the four Tully parameters A, B, C, D;
  * no thermostat, so `langevin`, `langevin_params` and `gamma` are gone entirely;
  * no thermalization stage, so `therm_time`/`therm_delta`/`therm_epsil`/`therm_dir` are gone and
    `system_file` points at the local traj.py, which draws its own initial conditions;
  * `mass` 1.00 -> 2000.00.

Every other value is carried over from the System-Bath config unchanged.

Output is saved in shards (workflow.shards): trajectory idx is a row of shards/shard_<s>.hdf with
s = idx // shard_size.  With n_traj=2M and shard_size=1000 that is 2000 shards (shard_0000 ..
shard_1999), run by 400 SLURM tasks x 5 shards each.  Merge them into one consolidate-style
data.hdf for the analysis scripts:

    python -m workflow.runner  --config config.py                     # local
    python -m workflow.slurm   --config config.py && sbatch submit_grid.sh
    python -m workflow.shards  --config config.py                     # merge shards -> data.hdf
"""

CONFIG = {
    # ----------------------------------------------------------------- orchestration (workflow)
    'n_traj'    : 100000000,     # number of independent trajectories to run
    'base_seed' : None,        # int -> reproducible; None -> fresh OS entropy each trajectory
    'ncores'    : 1,           # local cores for `python -m workflow.runner`
    'shard_size': 10000,        # trajectories per shards/shard_<s>.hdf (None -> one dir per trajectory)
    'shard_dir' : 'shards',    # relative to this dir; merge with `python -m workflow.shards`

    'slurm': {
        'max_tasks'     : 250,
        'job_name'      : 'MASHRPMD',
        'time'          : '4:59:00',
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

    # ----------------------------------------------------------------- stage wiring
    'grid_dir'   : 'traj_grid',
    'system_file': 'traj.py',   # resolved relative to THIS file

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

    # ----------------------------------------------------------------- initial conditions (see traj.py)
    # nucR ~ Normal(nucR_init, sqrt(1/(2*ga))); nucP = nucP_init on every bead.
    'nucR_init' : -10.00,      # wavepacket centre, to the left of the crossing at R = 0
    'nucP_init' :  10.00,      # incident momentum, directed toward the crossing
    'ga'        :   0.25,      # wavepacket width parameter

    # ----------------------------------------------------------------- ring polymer / integrator
    'nbds'  : 1,               # classical limit: a single bead
    'delt'  : 5.0,
    'intype': 'spin_magnus_adaptive',

    # ----------------------------------------------------------------- production
    'traj_time' : 4200.0,        # a.u.; Nsteps = traj_time / delt
    'Nprint'    : 5,

    # ----------------------------------------------------------------- quantum jumps
    'Tjump'     : 1050.0,
    'full_jump' : True,
}
