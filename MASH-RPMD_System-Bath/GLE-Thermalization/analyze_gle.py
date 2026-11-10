"""
analyze_gle.py -- convergence checks of the GLE thermostat, from the GLE-Thermalization run
(BO-RPMD in the harmonic donor well; see config.py).  Three figures and a printed summary:

  histogram.png       left: the bead positions pooled over beads, trajectories and times t >= tskip;
                      right: the same for the centroid.  Overlaid (all centred on the analytic
                      R0 = -sqrt(lbd/(2 kvec)), not on the sample mean):
                         classical          var = 1/(beta kvec)
                         quantum harmonic   var = 1/(2 m omega tanh(beta omega/2))   (no friction)
                         GLE-RPMD analytic  var = (1/nbds) sum_k 1/(beta_p (kvec + m w_k^2 + m gamma w_k))
                      (the GLE analytic is the distribution the run should reproduce -- the same
                      normal-mode widths therm_scan._sample_nvt draws from; the centroid is always
                      classical, var = 1/(beta kvec)).
  nm_momentum_acf.png the normal-mode momentum ACF <P_k(0) P_k(t)> per mode (FFT all-pairs ACF over
                      t >= tskip, mean +/- block SE between trajectories) vs the analytic GLE MACF.
                      Both as in junktest/multi_traj/analyze_nm_acf.py.
  equilibration.png   trajectory averages vs time, with SE bands: <R_centroid>(t) vs R0, and per
                      normal mode <P_k^2>(t) / (m/beta_p) and <(Q_k-Q_k^eq)^2>(t) / var_k^GLE, which
                      both -> 1 at equilibrium.  Use it to choose --tskip.

The data are read one unit (a shard, or <= --unit-rows rows of a data.hdf) at a time by --ncores
worker processes (workflow.plan_units / map_files): each unit is reduced to sums, and the error bars
come from --nblocks contiguous blocks of whole units (std of block means / sqrt(nblocks)).

Usage (from GLE-Thermalization/, PYTHONPATH=<RP_MASH root>):
    python analyze_gle.py                              # shards/ if present, else data.hdf
    python analyze_gle.py --ncores 16 --tskip 10 --nblocks 20 --tmax 5
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import normal_mode                                          # from RP_MASH (PYTHONPATH)
from workflow import plan_units, map_files, open_readonly


# ------------------------------------------------------------------ analytic references
def MACF(tmax, tpts, beta_p, mass, omega_k, gamma, omega_ext=0.0):
    """Analytic normal-mode momentum ACF of the GLE ring polymer in a harmonic well (verbatim from
    junktest/multi_traj/analyze_nm_acf.py)."""
    eps = 0.001
    N = 1 + 2*round(tpts-1)
    t_ar = np.linspace(-tmax, tmax, N); dt = (t_ar[-1] - t_ar[0]) / (N - 1)
    w_ar = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(N, d=dt)); dw = (w_ar[-1] - w_ar[0]) / (N - 1)
    s_ar = eps + 1j*w_ar
    Kw_ar = gamma*s_ar/(omega_k + np.sqrt(omega_k**2 + s_ar**2))
    Cw_ar = 2*(mass*s_ar/beta_p*(1/(s_ar**2 + s_ar*Kw_ar/mass + (omega_k**2 + gamma*omega_k + omega_ext**2)))).real
    Ct_ar = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Cw_ar)))*N*dw/(2*np.pi)
    return t_ar, Ct_ar


def analytic(cfg, nbds, beta, gamma):
    """Every equilibrium reference of the harmonic donor well."""
    mass = float(np.atleast_1d(cfg['mass'])[0])
    kvec, lbd = float(cfg['kvec']), float(cfg['lbd'])
    beta_p = beta / nbds
    omega = np.sqrt(kvec / mass)
    w_k = normal_mode.calc_normal_mode_freq(beta_p, nbds)
    varQ = 1.0 / (beta_p * (kvec + mass * w_k**2 + mass * gamma * w_k))      # (nbds,)
    return dict(mass=mass, kvec=kvec, omega=omega, beta_p=beta_p, w_k=w_k,
                R0=-np.sqrt(lbd / (2 * kvec)),
                var_cl=1.0 / (beta * kvec),
                var_qm=1.0 / (2 * mass * omega * np.tanh(beta * omega / 2)),
                var_gle=varQ.mean(),                  # orthonormal NM transform: bead var = mean_k
                var_cen=varQ[0] / nbds,               # = 1/(beta kvec)
                varQ=varQ, varP=mass / beta_p)


def nm_matrix(nbds):
    """U with Q = U @ R for one bead vector R (normal_mode.real_to_normal_mode, orthonormal)."""
    return np.stack([normal_mode.real_to_normal_mode(e) for e in np.eye(nbds)], axis=1)


def gauss(x, x0, var):
    return np.exp(-0.5 * (x - x0)**2 / var) / np.sqrt(2 * np.pi * var)


# ------------------------------------------------------------------ map step (one unit)
def _unit_partial(path, lo, hi, i_skip, n_lags, edges_b, edges_c, R0, U):
    with open_readonly(path) as f:
        R = f['nucR'][lo:hi, :, :, 0].astype(np.float64)       # (n, T, nbds), reaction coordinate
        P = f['nucP'][lo:hi, :, :, 0].astype(np.float64)
    n = R.shape[0]
    Rc = R.mean(axis=2)                                         # (n, T) centroid
    Q  = np.einsum('kb,ntb->ntk', U, R - R0)                    # NM positions about the well
    Pn = np.einsum('kb,ntb->ntk', U, P)                         # NM momenta

    Rb_eq, Rc_eq = R[:, i_skip:].ravel(), Rc[:, i_skip:].ravel()
    hb = np.histogram(Rb_eq, edges_b)[0].astype(np.float64)
    hc = np.histogram(Rc_eq, edges_c)[0].astype(np.float64)

    # unbiased all-pairs ACF per trajectory and mode over t >= tskip (zero-padded FFT)
    x = Pn[:, i_skip:, :]
    nt = x.shape[1]
    fx = np.fft.rfft(x, 2 * nt, axis=1)
    raw = np.fft.irfft(fx * np.conjugate(fx), 2 * nt, axis=1)[:, :n_lags, :]
    acf = (raw / (nt - np.arange(n_lags))[None, :, None]).sum(axis=0).T     # (nbds, n_lags)

    return dict(n=np.float64(n), hb=hb, hc=hc, acf=acf,
                Rc=Rc.sum(0), P2=(Pn**2).sum(0), Q2=(Q**2).sum(0),          # (T,), (T,nbds)
                mom_b=np.array([Rb_eq.size, Rb_eq.sum(), (Rb_eq**2).sum()]),
                mom_c=np.array([Rc_eq.size, Rc_eq.sum(), (Rc_eq**2).sum()]))


def block_stats(parts, key, nblocks):
    """(mean over all trajectories, block SE) of the per-trajectory sums `key`."""
    tot = sum(p[key] for p in parts) / sum(p['n'] for p in parts)
    nb = min(nblocks, len(parts))
    if nb < 2:
        return tot, np.zeros_like(tot), nb
    groups = np.array_split(np.arange(len(parts)), nb)
    bm = np.stack([sum(parts[i][key] for i in g) / sum(parts[i]['n'] for i in g) for g in groups])
    return tot, bm.std(axis=0, ddof=1) / np.sqrt(nb), nb


def _band(ax, x, y, se, color, label=None, lw=1.6):
    ax.plot(x, y, color=color, lw=lw, label=label)
    ax.fill_between(x, y - se, y + se, color=color, alpha=0.25, lw=0)


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, color='k', alpha=0.07, lw=0.7)
    ax.tick_params(direction='in')


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description='GLE thermalization convergence checks.')
    ap.add_argument('--file',    default=None, help='shard dir or data.hdf (default: shards/ if present)')
    ap.add_argument('--tskip',   type=float, default=10.0,
                    help='discard t < tskip (a.u.) from the histograms and the ACF (default 10)')
    ap.add_argument('--bins',    type=int,   default=99, help='histogram bins (default 99)')
    ap.add_argument('--lags',    type=int,   default=None, help='ACF lags to compute (default: all)')
    ap.add_argument('--tmax',    type=float, default=None, help='ACF plot range, a.u. (default: all)')
    ap.add_argument('--nblocks', type=int,   default=20, help='blocks for the error bars (default 20)')
    ap.add_argument('--ncores',  type=int,   default=None, help='worker processes (default: SLURM CPUs, else 1)')
    ap.add_argument('--unit-rows', type=int, default=1000, help='max trajectories per unit of work')
    a = ap.parse_args()
    path = a.file or ('shards' if os.path.isdir('shards') else 'data.hdf')

    units = [u for u in plan_units(path, a.unit_rows) if u.hi > u.lo]
    if not units:
        raise SystemExit(f'[gle] no trajectories in {path}')
    with open_readonly(units[0].file) as f:
        time = f['time'][:]
        cfg  = json.loads(f.attrs['config_json'])
        nbds = int(f.attrs['nbds'])
        beta = float(f.attrs['beta'])
        gamma = float(f.attrs['gamma'])
    dt = float(time[1] - time[0])
    i_skip = int(round(a.tskip / dt))
    nt_eff = time.size - i_skip
    if nt_eff <= 1:
        raise SystemExit(f'[gle] tskip={a.tskip} leaves {nt_eff} rows (T={time[-1]:g})')
    n_lags = nt_eff if a.lags is None else min(a.lags, nt_eff)

    A = analytic(cfg, nbds, beta, gamma)
    half = 6.0 * np.sqrt(max(A['var_cl'], A['var_qm'], A['var_gle']))
    edges_b = np.linspace(A['R0'] - half, A['R0'] + half, a.bins + 1)
    edges_c = np.linspace(A['R0'] - 6 * np.sqrt(A['var_cen']), A['R0'] + 6 * np.sqrt(A['var_cen']),
                          a.bins + 1)
    U = nm_matrix(nbds)

    print(f'[gle] {path}: {len(units)} unit(s), nbds={nbds} beta={beta:g} gamma={gamma:g} '
          f"Tmem={cfg['langevin_params'].get('Tmem')} Tfluc={cfg['langevin_params'].get('Tfluc')} "
          f'dt_out={dt:g} T={time[-1]:g} tskip={a.tskip:g}', flush=True)
    tasks = [(u.file, u.lo, u.hi, i_skip, n_lags, edges_b, edges_c, A['R0'], U) for u in units]
    parts = []
    for i, p in enumerate(map_files(_unit_partial, tasks, a.ncores)):
        parts.append(p)
        print(f'[gle] unit {i + 1}/{len(units)}', flush=True)
    ntraj = int(sum(p['n'] for p in parts))

    # ---------------------------------------------------------------- summary
    def moments(key):
        m = sum(p[key] for p in parts)
        mean = m[1] / m[0]
        return mean, m[2] / m[0] - mean**2, m[0]
    mb, vb, nb_pts = moments('mom_b')
    mc, vc, nc_pts = moments('mom_c')
    P2, P2_se, nblk = block_stats(parts, 'P2', a.nblocks)
    Q2, Q2_se, _    = block_stats(parts, 'Q2', a.nblocks)
    Rc, Rc_se, _    = block_stats(parts, 'Rc', a.nblocks)
    acf, acf_se, _  = block_stats(parts, 'acf', a.nblocks)
    rP = P2 / A['varP']
    rQ = Q2 / A['varQ'][None, :]
    print(f'[gle] {ntraj} trajectories, {nblk} blocks for the error bars')
    print(f"[gle] bead     mean {mb:+.5f} (R0 {A['R0']:+.5f})   var {vb:.5f}   "
          f"GLE {A['var_gle']:.5f}  classical {A['var_cl']:.5f}  quantum {A['var_qm']:.5f}")
    print(f"[gle] centroid mean {mc:+.5f} (R0 {A['R0']:+.5f})   var {vc:.5f}   analytic {A['var_cen']:.5f}")
    out_b = 1 - sum(p['hb'].sum() for p in parts) / nb_pts
    if out_b > 0:
        print(f'[gle] {100 * out_b:.3g}% of bead positions fell outside the histogram range')
    for k in range(nbds):
        print(f"[gle] mode k={k} (w_k={A['w_k'][k]:.4g}): <P_k^2>/(m/beta_p) = "
              f"{rP[i_skip:, k].mean():.4f}   <Q_k^2>/var_k = {rQ[i_skip:, k].mean():.4f}   (t >= tskip)")

    # ---------------------------------------------------------------- histogram.png
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, key, edges, var_list, title in (
            (axs[0], 'hb', edges_b,
             (('Classical', A['var_cl'], '#e34948'), ('Quantum (no friction)', A['var_qm'], '#1baf7a'),
              ('GLE-RPMD analytic', A['var_gle'], 'k')), 'all beads'),
            (axs[1], 'hc', edges_c, (('Classical = analytic', A['var_cen'], '#e34948'),), 'centroid')):
        h = sum(p[key] for p in parts)
        width = np.diff(edges)
        dens = h / (h.sum() * width) if h.sum() > 0 else h
        ax.bar(edges[:-1], dens, width=width, align='edge', color='#2a78d6', alpha=0.75,
               edgecolor='white', lw=0.4, label='simulation')
        xg = np.linspace(edges[0], edges[-1], 600)
        for lab, var, c in var_list:
            ax.plot(xg, gauss(xg, A['R0'], var), color=c, lw=1.8,
                    ls='--' if lab.startswith('GLE') else '-', label=lab)
        ax.set_xlabel('position (a.u.)')
        ax.set_ylabel('probability density')
        ax.set_title(f'{title}: {ntraj} traj, t >= {a.tskip:g}', loc='left')
        ax.legend(frameon=False)
        _style(ax)
    fig.suptitle(f'GLE position distribution   nbds={nbds}, beta={beta:g}, gamma={gamma:g}')
    fig.tight_layout()
    fig.savefig('histogram.png', dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- nm_momentum_acf.png
    tau = np.arange(n_lags) * dt
    n_plot = n_lags if a.tmax is None else min(n_lags, int(round(a.tmax / dt)) + 1)
    ncol = min(3, nbds)
    nrow = int(np.ceil(nbds / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow), squeeze=False)
    for k in range(nbds):
        ax = axes[k // ncol][k % ncol]
        _band(ax, tau[:n_plot], acf[k, :n_plot], acf_se[k, :n_plot], 'k', 'simulation', lw=2.2)
        tv, Cv = MACF(125.0, 1000000, A['beta_p'], A['mass'], A['w_k'][k], gamma, A['omega'])
        msk = (tv >= 0.0) & (tv <= tau[n_plot - 1])
        ax.plot(tv[msk], Cv[msk].real, color='tab:blue', lw=1.2, label='analytic MACF')
        dev = np.max(np.abs(acf[k, :n_plot] - np.interp(tau[:n_plot], tv[msk], Cv[msk].real)))
        print(f'[gle] mode k={k}: max |ACF - MACF| = {dev:.4g}  (ACF(0) = {acf[k, 0]:.4f}, '
              f"m/beta_p = {A['varP']:.4f})")
        ax.axhline(0.0, color='gray', lw=0.6)
        ax.set_title(fr"$k={k}$  ($\omega_k={A['w_k'][k]:.3f}$)", loc='left')
        ax.set_xlabel('t (a.u.)')
        ax.set_ylabel(r'$\langle\tilde{P}_k(0)\,\tilde{P}_k(t)\rangle$')
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
    for k in range(nbds, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'Normal-mode momentum ACF   gamma={gamma:g}, {ntraj} traj, t >= {a.tskip:g}')
    fig.tight_layout()
    fig.savefig('nm_momentum_acf.png', dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- equilibration.png
    cols = plt.cm.viridis(np.linspace(0, 0.9, nbds))
    fig, axs = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    _band(axs[0], time, Rc, Rc_se, 'k', r'$\langle \bar R \rangle(t)$')
    axs[0].axhline(A['R0'], color='#e34948', lw=1.2, ls='--', label='$R_0$')
    axs[0].set_ylabel('centroid position')
    for k in range(nbds):
        _band(axs[1], time, rP[:, k], P2_se[:, k] / A['varP'], cols[k], f'k={k}', lw=1.2)
        _band(axs[2], time, rQ[:, k], Q2_se[:, k] / A['varQ'][k], cols[k], f'k={k}', lw=1.2)
    axs[1].set_ylabel(r'$\langle \tilde P_k^2 \rangle / (m/\beta_p)$')
    axs[2].set_ylabel(r'$\langle \tilde Q_k^2 \rangle / \sigma_k^2$ (GLE)')
    axs[2].set_xlabel('t (a.u.)')
    for ax in axs:
        if ax is not axs[0]:
            ax.axhline(1.0, color='#e34948', lw=1.2, ls='--')
        ax.axvline(a.tskip, color='gray', lw=0.8, ls=':')
        ax.legend(frameon=False, fontsize=8, ncol=min(nbds, 6) if ax is not axs[0] else 2)
        _style(ax)
    fig.suptitle(f'Equilibration   nbds={nbds}, beta={beta:g}, gamma={gamma:g}, {ntraj} traj '
                 f'(dotted: tskip)')
    fig.tight_layout()
    fig.savefig('equilibration.png', dpi=160)
    plt.close(fig)
    print('[gle] wrote histogram.png, nm_momentum_acf.png, equilibration.png')


if __name__ == '__main__':
    main()
