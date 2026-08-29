"""
analyze_pop.py -- population/coherence correlation functions C_11(t) and C_12(t) from data.hdf.

Each is a SINGLE whole-bracket ensemble average over trajectories (no prefactor):

 C_11(t) = < 1.5 A Sx a(t)Sx(t)
             + A Sx (1 + B(t) sgn(Sz(t)))
             + (1 + B sgn(Sz)) a(t)Sx(t)
             + |Sz| (1 + B sgn(Sz)) (1 + B(t) sgn(Sz(t))) >

 C_12(t) = < -1.5 A Sx a(t)Sx(t)
             + A Sx (1 - B(t) sgn(Sz(t)))
             - (1 + B sgn(Sz)) a(t)Sx(t)
             + |Sz| (1 + B sgn(Sz)) (1 - B(t) sgn(Sz(t))) >

where A=a(R), B=b(R), Sx, sgn(Sz), |Sz| are t=0 (row 0 of `time`) and the "(t)" quantities are the
full time series. <...> is an average over trajectories with t=0 a prepared initial condition (not a
sliding origin). The standard error of each is ONE combined block-average of the whole bracket over
trajectories (not per-term quadrature).

Bead reduction (per user): A(t)=mean_b a(R_b(t)), B(t)=mean_b b(R_b(t)) -- the AVERAGE OVER BEADS of
the per-bead a,b (compute a,b per bead, then mean over beads), NOT a(Rbar)/b(Rbar). Sx(t), sgn(Sz(t)),
|Sz(t)| are likewise bead means of the per-bead value. The electronic-mixing functions use the
corrected definition from analyze_c12.py:
    kappa(R) = sqrt(lbd*kvec/2)*R + epsil/2,   norm = sqrt(kappa^2 + delta^2)
    a(R) = -delta/norm,   b(R) = kappa/norm
with production params (kvec, epsil, lbd, delta) from the HDF config_json.

For checking, the 4 sub-terms of each bracket are written as diagnostic columns so that
C_11 == sum(U_k) and C_12 == sum(V_k) hold exactly.

Usage (from the run dir, PYTHONPATH=<RP_MASH root>):
    python analyze_pop.py --file data.hdf --nblocks 7 --tmax 50
"""

import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from workflow import open_run            # h5py.File(path, 'r')


def block_sem(per_traj, n_blocks):
    """SE over trajectories by block averaging (same helper as the other analyze_* scripts)."""
    ntraj = per_traj.shape[0]
    n_blocks = ntraj if n_blocks is None else max(1, min(int(n_blocks), ntraj))
    if n_blocks < 2:
        return np.zeros(per_traj.shape[1:]), n_blocks
    blocks = np.array_split(per_traj, n_blocks, axis=0)
    block_means = np.stack([b.mean(axis=0) for b in blocks], axis=0)
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks), n_blocks


def a_b_functions(R, kvec, epsil, lbd, delta):
    """a(R), b(R) elementwise (corrected definition, matches analyze_c12.py)."""
    kappa = np.sqrt(lbd * kvec / 2.0) * R + epsil / 2.0
    norm  = np.sqrt(kappa**2 + delta**2)
    return -delta / norm, kappa / norm


U_LABELS = ['1.5 A Sx a(t)Sx(t)', 'A Sx (1+B(t)sgn(t))', '(1+B sgn) a(t)Sx(t)',
            '|Sz|(1+B sgn)(1+B(t)sgn(t))']
V_LABELS = ['-1.5 A Sx a(t)Sx(t)', 'A Sx (1-B(t)sgn(t))', '-(1+B sgn) a(t)Sx(t)',
            '|Sz|(1+B sgn)(1-B(t)sgn(t))']


def compute_pop(path, nblocks=None, tmax=None):
    with open_run(path) as f:
        time = f['time'][:]                     # (T,)
        cfg  = json.loads(f.attrs['config_json'])
        kvec  = float(cfg['kvec']); epsil = float(cfg['epsil'])
        lbd   = float(cfg['lbd']);  delta = float(cfg['delta'])
        nbds  = int(f.attrs['nbds'])

        R  = f['nucR'][:, :, :, 0]              # (n, T, nbds)
        Sx = f['mapSx'][:]                      # (n, T, nbds)
        Sz = f['mapSz'][:]                      # (n, T, nbds)

    n_traj = R.shape[0]

    # ---- bead reduction: A,B = AVERAGE OVER BEADS of per-bead a,b ----
    Rbar = R.mean(axis=2)
    A, B = a_b_functions(Rbar, kvec, epsil, lbd, delta)
    #aR, bR = a_b_functions(R, kvec, epsil, lbd, delta)   # per bead (n,T,nbds)
    #A     = aR.mean(axis=2)                     # A(t) = mean_b a(R_b(t))
    #B     = bR.mean(axis=2)                     # B(t) = mean_b b(R_b(t))
    Sx_   = Sx.mean(axis=2)                     # Sx(t)
    sgnSz = np.sign(Sz).mean(axis=2)            # sgn(Sz(t))  in [-1,1]
    absSz = np.abs(Sz).mean(axis=2)             # |Sz(t)|

    # ---- t=0 scalars (row 0), shape (n,1) to broadcast against (n,T) ----
    A0, Sx0, B0, sgn0, abs0 = (X[:, :1] for X in (A, Sx_, B, sgnSz, absSz))

    aSx_t   = A * Sx_                            # a(t) Sx(t)
    proj0   = 1.0 + B0 * sgn0                    # (1 + B sgn(Sz))  at t=0
    projt_p = 1.0 + B * sgnSz                    # (1 + B(t) sgn(Sz(t)))
    projt_m = 1.0 - B * sgnSz                    # (1 - B(t) sgn(Sz(t)))
    ASx0    = A0 * Sx0                           # A Sx at t=0

    # ---- C_11 sub-terms (n,T) ----
    U = [ 1.5 * ASx0 * aSx_t,
          ASx0 * projt_p,
          proj0 * aSx_t,
          abs0 * proj0 * projt_p ]
    # ---- C_12 sub-terms (n,T) ----
    V = [ -1.5 * ASx0 * aSx_t,
          ASx0 * projt_m,
          -proj0 * aSx_t,
          abs0 * proj0 * projt_m ]

    G11 = sum(U); G12 = sum(V)                   # full brackets per trajectory (n,T)

    C11 = G11.mean(axis=0);  C11_SE, used_nb = block_sem(G11, nblocks)
    C12 = G12.mean(axis=0);  C12_SE, _       = block_sem(G12, nblocks)
    U_means = [Uk.mean(axis=0) for Uk in U]
    V_means = [Vk.mean(axis=0) for Vk in V]

    if tmax is not None:
        m = time <= tmax
        time = time[m]
        C11, C11_SE, C12, C12_SE = C11[m], C11_SE[m], C12[m], C12_SE[m]
        U_means = [u[m] for u in U_means]; V_means = [v[m] for v in V_means]

    return dict(time=time, C11=C11, C11_SE=C11_SE, C12=C12, C12_SE=C12_SE,
                U_means=U_means, V_means=V_means,
                n_traj=n_traj, nbds=nbds, nblocks=used_nb,
                params=dict(kvec=kvec, epsil=epsil, lbd=lbd, delta=delta))


def _write_dat(out, r):
    cols = [r['time'], r['C11'], r['C11_SE'], r['C12'], r['C12_SE']] + r['U_means'] + r['V_means']
    data = np.column_stack(cols)
    p = r['params']
    header = (
        'C_11(t), C_12(t): single whole-bracket ensemble averages (see module docstring; no prefactor)\n'
        'C11_SE/C12_SE = blockSEM of the whole bracket (one combined block SE, not per-term quadrature)\n'
        f"n_traj={r['n_traj']} nbds={r['nbds']} nblocks={r['nblocks']} "
        f"kvec={p['kvec']} epsil={p['epsil']} lbd={p['lbd']} delta={p['delta']:.6e}\n"
        'A=mean_b a(R_b), B=mean_b b(R_b) [avg over beads]; a=-delta/norm, b=kappa/norm\n'
        'C11 == U1+U2+U3+U4 ; C12 == V1+V2+V3+V4 (diagnostic sub-term means below)\n'
        'columns: time C11 C11_SE C12 C12_SE  U1 U2 U3 U4  V1 V2 V3 V4'
    )
    np.savetxt(out, data, fmt='%20.10e', header=header)


def _plot(out, r):
    t = r['time']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)

    #ax1.axhline(0.0, color='0.7', lw=0.8)
    #ax1.plot(t, r['C11'], color='C0', lw=1.6, label=r'$C_{11}(t)$')
    #ax1.fill_between(t, r['C11']-r['C11_SE'], r['C11']+r['C11_SE'], color='C0', alpha=0.3, lw=0)
    ax1.plot(t, r['C12'], color='C3', lw=1.6, label=r'$C_{12}(t)$')
    ax1.fill_between(t, r['C12']-r['C12_SE'], r['C12']+r['C12_SE'], color='C3', alpha=0.3, lw=0)
    ax1.set_ylabel('correlation function')
    ax1.legend(loc='best', frameon=False)
    ax1.set_title(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}")

    for u, lab in zip(r['U_means'], U_LABELS):
        ax2.plot(t, u, lw=1.0, ls='-',  label='C11: ' + lab)
    for v, lab in zip(r['V_means'], V_LABELS):
        ax2.plot(t, v, lw=1.0, ls='--', label='C12: ' + lab)
    ax2.axhline(0.0, color='0.7', lw=0.8)
    ax2.set_xlabel('time (a.u.)'); ax2.set_ylabel('sub-term averages')
    ax2.legend(loc='best', frameon=False, ncol=2, fontsize=6)

    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(description='Compute C_11(t) and C_12(t) from an RP-MASH data.hdf.')
    ap.add_argument('--file',    default='data.hdf')
    ap.add_argument('--nblocks', type=int, default=None,
                    help='trajectory blocks for the SE (default: one per trajectory = plain SEM)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate output/plot to t <= tmax')
    ap.add_argument('--out',     default='pop', help='output basename (writes <out>.dat and <out>.png)')
    a = ap.parse_args()

    r = compute_pop(a.file, nblocks=a.nblocks, tmax=a.tmax)
    _write_dat(a.out + '.dat', r)
    _plot(a.out + '.png', r)

    p = r['params']
    print(f"[pop] n_traj={r['n_traj']} nbds={r['nbds']} nblocks={r['nblocks']}")
    print(f"[pop] params: kvec={p['kvec']} epsil={p['epsil']} lbd={p['lbd']} delta={p['delta']:.6e}")
    print(f"[pop] C11(0)={r['C11'][0]:.6e} +/- {r['C11_SE'][0]:.3e}")
    print(f"[pop] C12(0)={r['C12'][0]:.6e} +/- {r['C12_SE'][0]:.3e}")
    print(f"[pop] wrote {a.out}.dat and {a.out}.png  ({r['time'].size} time points)")


if __name__ == '__main__':
    _main()
