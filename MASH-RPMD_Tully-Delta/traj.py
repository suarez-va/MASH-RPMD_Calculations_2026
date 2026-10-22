"""
traj.py -- the ONLY stage for the Tully single-avoided-crossing grid with DELTA-FUNCTION initial
conditions (MASH-RPMD_Tully-Delta).

Identical to MASH-RPMD_Tully/traj.py except for the nuclear initial state.  There the trajectories
sample a Gaussian wavepacket of width `ga`; here every trajectory, and every bead, starts EXACTLY at

    nucR = nucR_init        nucP = nucP_init

so there is no `ga` and no initial-condition random draw.  The trajectories differ only through the
sampled initial spin (init_map_spin(0): lower hemisphere) and the quantum-jump times, both drawn
inside mash_rpmd from the dynamics seed.

There is no thermalization stage and no thermostat: a Tully scattering problem has no equilibrium
ensemble to generate, so the initial state is set right here and propagated microcanonically.

In shard mode (config `shard_size`, see workflow.shards) each trajectory runs in its own temp dir
(under $TMPDIR), where mash_rpmd writes its .dat files; workflow.shards parses them into a row of
shards/shard_<s>.hdf and deletes the dir.  Nothing here depends on the cwd, so run_trajectory(cfg,
seed) is the same in shard mode and the old one-dir-per-trajectory mode (shard_size: None).

    python -m workflow.runner --config config.py                        # local
    python -m workflow.slurm  --config config.py && sbatch submit_grid.sh  # cluster
    python -m workflow.shards --config config.py                        # merge -> data.hdf
"""

import numpy as np
import mash_rpmd   # via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)


def _seeds(seed):
    """
    Split the one trajectory seed into (initial conditions, dynamics), exactly as in
    MASH-RPMD_Tully/traj.py.  The initial-condition half is unused here (the initial state is
    deterministic), but keeping the same split means a given `base_seed` gives trajectory k the SAME
    spin/jump streams as the Gaussian-wavepacket run: correlated sampling between the two workflows.
    seed=None -> OS entropy.
    """
    if seed is None:
        return None, None
    ic, dyn = np.random.SeedSequence(int(seed)).generate_state(2, dtype=np.uint32)
    return int(ic), int(dyn)


def run_trajectory(cfg, seed):
    _, dyn_seed = _seeds(seed)

    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])

    # ---- delta-function initial state: every bead exactly at (nucR_init, nucP_init) ----
    nucR = np.full((nbds, nnuc), float(cfg['nucR_init']))
    nucP = np.full((nbds, nnuc), float(cfg['nucP_init']))

    # tully_1ac wants each parameter as its own (nnuc,) array -- see potential.tully_1ac.error_check
    potparams = [np.array([cfg['A']]), np.array([cfg['B']]),
                 np.array([cfg['C']]), np.array([cfg['D']])]

    # No langevin/langevin_params: mash_rpmd defaults langevin=None, which switches off every GLE
    # branch in the integrator (and with it the per-trajectory memK.dat that the System-Bath driver
    # has to delete afterwards).
    m = mash_rpmd.mash_rpmd(
        nstates=cfg['nstates'], nnuc=nnuc, nbds=nbds,
        beta=cfg['beta'], mass=np.asarray(cfg['mass'], float),
        potype='tully_1ac', potparams=potparams, nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        Tjump=cfg['Tjump'], full_jump=cfg['full_jump'], seed=dyn_seed)

    # init_state=0 -> cos(theta) drawn from (-1, 0): the particle starts on the LOWER adiabat,
    # matching run_mashrpmd.py.  (System-Bath uses init_state=None because it samples the whole
    # sphere for its equilibrium correlation function; a scattering problem has a definite entrance
    # channel.)
    m.init_map_spin(0)

    Nprod = round(cfg['traj_time'] / cfg['delt'])
    m.run_dynamics(Nprod, cfg['Nprint'], cfg['delt'], cfg['intype'],
                   init_time=0.0, small_dt_ratio=1)
