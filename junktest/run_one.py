"""
run_one.py -- flat, single-trajectory MASH-RPMD driver.

Everything is hardcoded below. Analytically sample (R,P) from the NVT distribution,
build the model with the real parameters, initialize the spin uniformly on the sphere,
and run production dynamics printing every timestep. No equilibration, no checkpoint.

Run with:  python run_one.py          (needs PYTHONPATH=/home/victorwsl/repos/RP_MASH)
Writes:    output.dat nucR.dat nucP.dat mapSx.dat mapSy.dat mapSz.dat info.txt [memK.dat]
"""

import numpy as np
import normal_mode
import mash_rpmd

# ----------------------------------------------------------------------- PARAMETERS
SEED = 0                # int -> reproducible; None -> fresh OS entropy

# system
NBDS    = 8
NNUC    = 1
NSTATES = 2
MASS    = np.array([1.3])
BETA    = 1.1

# Marcus ET model (atomic units)
KVEC  = 4.0
EPSIL = 0.0
LBD   = 12.0
DELTA = 10**(-1.4)
BATH  = np.array([0, 1.0, 2.0, 2.0])    # [N_bath, mass, gamma, w_b]; N_bath=0 -> no explicit bath

# integrator / thermostat
#INTYPE          = 'vv'
INTYPE          = 'spin_magnus_adaptive'
DELT            = 0.002
TOTAL_TIME      = 8.8                   # a.u.; Nsteps = TOTAL_TIME / DELT
NPRINT          = 1                     # 1 -> print every timestep
LANGEVIN        = None
#LANGEVIN        = 'generalized'
LANGEVIN_PARAMS = {'gamma': 2.00, 'Tmem': 10.0, 'Tfluc': 125}

INIT_STATE = 0       # None -> spin sampled over the whole sphere; 0 -> donor, 1 -> acceptor
# -----------------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

# ---- analytic NVT sample of (R,P) ----
omega_k = normal_mode.calc_normal_mode_freq(BETA / NBDS, NBDS)
R0      = -np.sqrt(LBD / (2 * KVEC))

dR_k    = np.sqrt(NBDS / (BETA * (KVEC + MASS[None, :] * (omega_k**2)[:, None])))
nucR_nm = rng.normal(0.0, dR_k, (NBDS, NNUC))
nucR    = np.zeros((NBDS, NNUC))
for i in range(NNUC):
    nucR[:, i] = normal_mode.normal_mode_to_real(nucR_nm[:, i])
nucR[:, 0] += R0
nucR[:, 0] += 0.5

dP0  = np.sqrt(MASS[0] * NBDS / BETA)
nucP = rng.normal(0.0, dP0, (NBDS, NNUC))
nucP += 11.0

# ---- build the model with the real parameters ----
mash = mash_rpmd.mash_rpmd(
    nstates=NSTATES, nnuc=NNUC, nbds=NBDS,
    beta=BETA, mass=MASS,
    potype='ET_with_bath', potparams=[KVEC, EPSIL, LBD, DELTA, BATH],
    nucR=nucR, nucP=nucP,
    spinmap_bool=True, centroid_bool=True, bead_bool=False,
    langevin = LANGEVIN, langevin_params = LANGEVIN_PARAMS, seed=SEED)

mash.init_map_spin(init_state=INIT_STATE)

# ---- production ----
Nsteps = round(TOTAL_TIME / DELT)
mash.run_dynamics(Nsteps, NPRINT, DELT, INTYPE, init_time=0.0, small_dt_ratio=1)
