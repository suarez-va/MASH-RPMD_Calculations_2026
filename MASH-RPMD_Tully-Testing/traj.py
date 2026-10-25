"""
traj.py -- the trajectory driver of one STAGE of MASH-RPMD_Tully-Testing.

Every stage directory stage_<m>/ holds a generated config.py (system_file -> ../traj.py) and the
stage's initial points init.npz (q, p, w: see resample.py).  Trajectory idx of the stage starts at
point

    i = idx // n_spin            (the n_spin trajectories of a point are consecutive indices)

with every bead at exactly (q_i, p_i), as in MASH-RPMD_Tully-Delta, and a spin sampled uniformly
on the WHOLE sphere (init_map_spin() with no state).  It runs for seg_time with no quantum jumps,
starting the clock at the stage's t_start so the shard /time is absolute.  The point's weights w_i
are not needed here: iterate.py applies them afterwards from init.npz and the trajectory's
spin at the first (S^(0)) and later print times.

Nothing is returned, so workflow.shards parses the usual .dat files into the stage's shards.  The
driver needs the trajectory index, which only the shard path hands over: shard_size is required.
"""

import os

import numpy as np
import mash_rpmd   # via PYTHONPATH (= /home/victorwsl/repos/RP_MASH)


_POINTS = {}   # init.npz path -> (q, p), loaded once per process


def _points(path):
    if path not in _POINTS:
        with np.load(path) as z:
            _POINTS[path] = (np.array(z['q'], float), np.array(z['p'], float))
    return _POINTS[path]


def run_trajectory(cfg, seed, idx, run_dir):
    if not cfg.get('shard_size'):
        raise ValueError('MASH-RPMD_Tully-Testing needs shard mode (shard_size): the trajectory '
                         'index selects the initial point')
    q, p = _points(os.path.join(run_dir, cfg['init_file']))
    i = int(idx) // int(cfg['n_spin'])
    if i >= q.size:
        raise IndexError(f'trajectory {idx} -> point {i}, but init has {q.size} points')

    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])
    nucR = np.full((nbds, nnuc), q[i])
    nucP = np.full((nbds, nnuc), p[i])

    # tully_1ac wants each parameter as its own (nnuc,) array -- see potential.tully_1ac.error_check
    potparams = [np.array([cfg['A']]), np.array([cfg['B']]),
                 np.array([cfg['C']]), np.array([cfg['D']])]

    m = mash_rpmd.mash_rpmd(
        nstates=cfg['nstates'], nnuc=nnuc, nbds=nbds,
        beta=cfg['beta'], mass=np.asarray(cfg['mass'], float),
        potype='tully_1ac', potparams=potparams, nucR=nucR, nucP=nucP,
        spinmap_bool=True, centroid_bool=True, bead_bool=False,
        jump_index=[], seed=seed, verbose=False)

    # whole-sphere sample: the stage's electronic state lives in the weights, not in the sampling
    m.init_map_spin()

    Nseg = round(cfg['seg_time'] / cfg['delt'])
    m.run_dynamics(Nseg, cfg['Nprint'], cfg['delt'], cfg['intype'],
                   init_time=float(cfg['t_start']), small_dt_ratio=1)
