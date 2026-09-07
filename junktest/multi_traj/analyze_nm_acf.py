"""
Example analysis script -- a TEMPLATE you copy and edit per plot.

It shows the intended pattern once you've built data.hdf with
    python -m workflow.consolidate --config config.py :

  1. open the HDF5 read-only (data stays on disk; slices load lazily),
  2. pull one trajectory at a time and transform its bead coordinates -> normal modes,
  3. accumulate a per-trajectory, time-origin-averaged correlation function,
  4. average over trajectories for the mean + a BETWEEN-trajectory standard error.

Here the observable is the normal-mode MOMENTUM autocorrelation  C_k(t) = <P_k(0) P_k(t)>.
Swap nucP->nucR (in the read loop) for the position ACF, or replace the correlator entirely.

Error bars: each trajectory contributes ONE time-origin-averaged ACF curve (the raw data point,
per lag). The error is the spread of those curves BETWEEN uncorrelated trajectories -- never across
lag pairs within a single trajectory. It is estimated by block averaging: the trajectories are
split into --nblocks groups, each group's mean ACF is one super-sample, and the reported bar is
std(block means, ddof=1)/sqrt(nblocks). With --nblocks equal to the trajectory count (the default),
this reduces exactly to the plain per-trajectory standard error of the mean.

    python analyze_nm_acf.py --file=data.hdf --lags=20000 --tmax=5.0 --tskip=10.0 --nblocks=7

    --file    HDF5 produced by `python -m workflow.consolidate` (default: data.hdf)
    --lags    how many lag points to COMPUTE (default: all remaining rows after tskip). Larger ->
              longer ACF, but the tail gets noisier (fewer time-origin pairs contribute at long lag).
    --tmax    final time (a.u.) to PLOT out to (default: the whole computed range). Only windows
              the figure; it does not change what was computed.
    --tskip   discard the first tskip a.u. of each trajectory before averaging -- i.e. start from
              index round(tskip/dt), dropping the equilibration transient (default: 0).
    --nblocks number of trajectory blocks for the between-trajectory error bar (e.g. 77 trajectories
              with --nblocks=7 -> 7 blocks of 11). Default: one block per trajectory.
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import normal_mode                       # from RP_MASH (PYTHONPATH)
from workflow import open_run            # h5py.File(path, 'r')


def MACF(tmax, tpts, beta_p, mass, omega_k, gamma, omega_ext=0.0):
    eps = 0.001
    N = 1 + 2*round(tpts-1)
    t_ar = np.linspace(-tmax, tmax, N); dt = (t_ar[-1] - t_ar[0]) / (N - 1); i0 = round(N/2)-1
    w_ar = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, d=dt)); dw = (w_ar[-1] - w_ar[0]) / (N - 1)
    s_ar = eps + 1j*w_ar
    Kw_ar = gamma*s_ar/(omega_k + np.sqrt(omega_k**2 + s_ar**2))
    Cw_ar = 2*(mass*s_ar/beta_p*(1/(s_ar**2 + s_ar*Kw_ar/mass + (omega_k**2 + gamma*omega_k + omega_ext**2)))).real
    Ct_ar = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Cw_ar)))*N*dw/(2*np.pi)
    return t_ar, Ct_ar













def nm_transform_traj(R):
    """R: (T, nbds, nnuc) bead values (position OR momentum) -> (T, nbds, nnuc) real normal modes."""
    T, nbds, nnuc = R.shape
    Q = np.empty_like(R)
    for t in range(T):
        for a in range(nnuc):
            Q[t, :, a] = normal_mode.real_to_normal_mode(R[t, :, a])
    return Q


def acf_all_pairs(x, n_lags):
    """Unbiased all-pairs autocorrelation: C(j) = mean over EVERY pair (i, i+j) of x[i]*x[i+j],
    every time point weighted equally (no special origin). The FFT (Wiener-Khinchin, zero-padded
    to 2*nt so the correlation is linear not circular) just makes it O(n log n) vs the O(n^2) double
    loop -- the result is bit-for-bit the same. counts = nt-j is the true number of pairs at lag j."""
    nt = x.shape[0]
    n_lags = int(min(n_lags, nt))
    fx = np.fft.rfft(x, 2 * nt)
    raw = np.fft.irfft(fx * np.conjugate(fx), 2 * nt)[:n_lags]
    counts = nt - np.arange(n_lags)                      # number of pairs at each lag
    return raw / counts


def block_sem(per_traj, n_blocks):
    """Between-trajectory standard error of the trajectory-mean ACF, via block averaging.

    per_traj : (ntraj, nbds, n_lags); per_traj[i,k] is trajectory i's OWN time-origin-averaged ACF
               for mode k -- one independent raw data point per trajectory, per lag.
    n_blocks : split the ntraj trajectories into this many contiguous groups; each group's mean ACF
               is one super-sample and SEM = std(block means, ddof=1)/sqrt(n_blocks). n_blocks=ntraj
               -> plain per-trajectory SEM. Every lag is treated independently, so NO error is
               propagated across lag pairs within a trajectory.

    Returns (sem, n_blocks_used); sem has shape (nbds, n_lags).
    """
    ntraj = per_traj.shape[0]
    n_blocks = ntraj if n_blocks is None else max(1, min(int(n_blocks), ntraj))
    if n_blocks < 2:
        return np.zeros(per_traj.shape[1:]), n_blocks
    blocks = np.array_split(per_traj, n_blocks, axis=0)               # groups of whole trajectories
    block_means = np.stack([b.mean(axis=0) for b in blocks], axis=0)  # (n_blocks, nbds, n_lags)
    sem = block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    return sem, n_blocks


def main():
    ap = argparse.ArgumentParser(description='Normal-mode momentum ACF across trajectories.')
    ap.add_argument('--file',  default='data.hdf', help='consolidated HDF5 (default: data.hdf)')
    ap.add_argument('--lags',  type=int,   default=None, help='lags to compute (default: full time)')
    ap.add_argument('--tmax',  type=float, default=None, help='final time to plot, a.u. (default: all)')
    ap.add_argument('--tskip', type=float, default=0.0,  help='discard data before this time, a.u. (default: 0)')
    ap.add_argument('--nblocks', type=int, default=None,
                    help='trajectory blocks for the between-trajectory error bar (default: one per trajectory)')
    a = ap.parse_args()
    path, n_lags, tmax, tskip, nblocks = a.file, a.lags, a.tmax, a.tskip, a.nblocks

    with open_run(path) as f:
        time  = f['time'][:]
        delt  = float(time[1] - time[0])
        nbds  = int(f.attrs['nbds'])
        nnuc  = int(f.attrs['nnuc'])
        ntraj = f['nucR'].shape[0]
        T     = f['nucR'].shape[1]

        # discard the first tskip a.u. of each trajectory (equilibration transient)
        i_skip = int(round(tskip / delt))
        nt_eff = T - i_skip                                   # rows left to average over
        if nt_eff <= 1:
            raise ValueError(f'tskip={tskip} leaves {nt_eff} rows (T={T}, dt={delt}); too large')
        n_lags = nt_eff if n_lags is None else min(n_lags, nt_eff)

        # parameters for the analytic MACF overlay (from the stored config)
        beta  = float(f.attrs['beta'])
        gamma = float(f.attrs['gamma'])
        cfg   = json.loads(f.attrs['config_json'])
        mass  = float(np.atleast_1d(cfg['mass'])[0])
        kvec  = float(cfg['kvec'])

        # accumulate per-mode ACF across trajectories: (ntraj, nbds, n_lags), averaged over nuclei
        per_traj = np.empty((ntraj, nbds, n_lags))
        for i in range(ntraj):
            #R = f['nucR'][i][i_skip:]                     # lazy read: only trajectory i (T, nbds, nnuc)
            #Q = nm_transform_traj(R)
            P = f['nucP'][i][i_skip:]                      # lazy read from i_skip onward (nt_eff, nbds, nnuc)
            print(P.shape)
            Pnm = nm_transform_traj(P)
            for k in range(nbds):
                c = np.zeros(n_lags)
                for a in range(nnuc):
                    c += acf_all_pairs(Pnm[:, k, a], n_lags)
                per_traj[i, k] = c / nnuc
            print(f'[analyze] trajectory {i + 1}/{ntraj}', flush=True)

    mean = per_traj.mean(axis=0)                          # (nbds, n_lags): trajectory-averaged ACF
    sem, nblk = block_sem(per_traj, nblocks)              # between-trajectory error via block averaging
    print(f'[analyze] error bars: {nblk} blocks over {ntraj} trajectories '
          f'(~{ntraj / nblk:.1f} traj/block), between-trajectory only', flush=True)
    tau  = np.arange(n_lags) * delt
    print("Here")
    print(tau)
    print(tau[-1])

    # window the PLOT to tmax (computation above is unaffected)
    n_plot = n_lags if tmax is None else min(n_lags, int(round(tmax / delt)) + 1)
    plot_tmax = tau[n_plot - 1]
    print(n_plot)
    print(mean[0, :n_plot])

    # analytic-MACF parameters shared by every mode
    beta_p    = beta / nbds
    omega_k   = normal_mode.calc_normal_mode_freq(beta_p, nbds)   # (nbds,), k=0 -> 0
    omega_ext = np.sqrt(kvec / mass)

    # ---- plot: one panel per normal mode, mean +/- SE band -----------------
    ncol = 3
    nrow = int(np.ceil(nbds / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for k in range(nbds):
        ax = axes[k // ncol][k % ncol]
        # simulated normal-mode momentum ACF (mean +/- SE over trajectories)
        ax.plot(tau[:n_plot], mean[k, :n_plot], color='k', lw=2.5, label='simulation')
        ax.fill_between(tau[:n_plot], (mean[k] - sem[k])[:n_plot], (mean[k] + sem[k])[:n_plot],
                        color='tab:red', alpha=0.3)
        # analytic MACF -- one call per mode (MACF is not vectorized over omega_k)
        #tv, Cv = MACF(100.0, 1000000, beta_p, mass, omega_k[k], gamma, omega_ext)
        #tv, Cv = MACF(125.0, 1000000, beta_p, mass, omega_k[k], gamma, omega_ext)
        tv, Cv = MACF(125.0, 1000000, beta_p, mass, omega_k[k], gamma, omega_ext)
        #tv, Cv = MACF(125.0, 1000000, beta_p, mass, omega_k[k], gamma)
        m = (tv >= 0.0) & (tv <= plot_tmax)
        ax.plot(tv[m], Cv[m].real, color='tab:blue', lw=1.2, label='analytic MACF')
        ax.axhline(0.0, color='gray', lw=0.6)
        ax.set_title(fr'normal-mode $k={k}$  ($\omega_k={omega_k[k]:.3f}$)', loc='left')
        ax.set_xlabel('t (a.u.)')
        ax.set_ylabel(r'$\langle\tilde{P}_k(0)\,\tilde{P}_k(t)\rangle$')
        ax.legend(loc='best', fontsize=8, frameon=True, framealpha=0.85, facecolor='white')
    for k in range(nbds, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'Momentum Autocorrelation Function; γ={gamma}', fontsize=14)
    fig.tight_layout()
    plt.savefig('nm_momentum_acf.png', dpi=150)
    #plt.show()


if __name__ == '__main__':
    main()
