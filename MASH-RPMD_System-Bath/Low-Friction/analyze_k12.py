"""
analyze_k12.py -- plot C_12(t) and the transformed rate integrand
    K_12(t) = -P2eq * ln| 1 - C_12(t)/P2eq |
from pop_c12.dat, with error bars propagated through the logarithm, and a least-squares line fit of
K_12(t) over a chosen time window (default t=10..20).

  top panel    : C_12(t) with its +/- SE band (from pop_c12.dat column 2)
  bottom panel : K_12(t) with its propagated +/- SE band, plus the m*t+b least-squares fit over
                 [fitmin, fitmax]; the slope m (rate), intercept b, and Pearson r are reported.

P2eq = 1 / (1 + exp(-beta*epsil)), with beta (FULL inverse temperature, not beta/nbds) and epsil read
from the run config.py by default (override with --beta / --epsil / --config).

Error propagation: with u(t) = 1 - C_12(t)/P2eq,
    K_12 = -P2eq * ln|u|   ->   dK/dC_12 = 1/u   ->   sigma_K = sigma_C12 / |u|.
Where |u| = 0 (division by zero), that point's error is set to 0.

Usage (from the run dir):
    python analyze_k12.py                       # fit t=10..20, beta/epsil from config.py
    python analyze_k12.py --fitmin 10 --fitmax 20 --tmax 50
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from workflow import load_config           # reads config.py -> CONFIG dict


def _parse_header_epsil(path):
    """Fallback: pull the 'epsil=<value>' token from the .dat comment header, if present."""
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith('#'):
                    break
                for tok in line.replace(',', ' ').split():
                    if tok.startswith('epsil='):
                        return float(tok.split('=', 1)[1])
    except Exception:
        pass
    return None


def _main():
    ap = argparse.ArgumentParser(description='Plot C_12(t), K_12(t)=-P2eq*ln|1-C12/P2eq|, and a line fit over a window.')
    ap.add_argument('--file',   default='pop_c12.dat', help='input .dat: time, C12, C12_SE')
    ap.add_argument('--tmax',   type=float, default=None, help='truncate output/plot to t <= tmax')
    ap.add_argument('--fitmin', type=float, default=10.0, help='fit window start (default 10)')
    ap.add_argument('--fitmax', type=float, default=20.0, help='fit window end (default 20)')
    ap.add_argument('--config', default='config.py', help='run config providing beta/epsil (default config.py)')
    ap.add_argument('--beta',   type=float, default=None, help='override full inverse temperature (default: config.py)')
    ap.add_argument('--epsil',  type=float, default=None, help='override driving force epsil (default: config.py)')
    ap.add_argument('--out',    default=None, help='output png (default: <file stem>_k12.png)')
    a = ap.parse_args()

    data = np.loadtxt(a.file)
    time = data[:, 0]; C12 = data[:, 1]; C12_SE = data[:, 2]

    # beta/epsil from config.py by default; CLI overrides; header/1.0 fallbacks
    cfg = None
    if a.beta is None or a.epsil is None:
        try:
            cfg = load_config(a.config)
        except Exception as ex:
            print(f"[k12] WARNING: could not load {a.config} ({ex}); using overrides/fallbacks")

    def _resolve(name, cli_val, fallback):
        if cli_val is not None:
            return float(cli_val)
        if cfg is not None and name in cfg:
            return float(cfg[name])
        return fallback

    beta  = _resolve('beta',  a.beta,  1.0)
    epsil = _resolve('epsil', a.epsil, _parse_header_epsil(a.file) or 0.0)
    P2eq  = 1.0 / (1.0 + np.exp(-beta * epsil))

    # K_12(t) and its propagated error; guard the 1/|u| division
    u = 1.0 - C12 / P2eq
    with np.errstate(divide='ignore', invalid='ignore'):
        K12    = -P2eq * np.log(np.abs(u))
        K12_SE = C12_SE / np.abs(u)
    K12_SE = np.where(np.isfinite(K12_SE), K12_SE, 0.0)   # |u|=0 -> divide by zero -> error 0

    # ---- least-squares line fit m*t + b of K_12 over [fitmin, fitmax] (finite points only) ----
    fmask = (time >= a.fitmin) & (time <= a.fitmax) & np.isfinite(K12)
    if int(fmask.sum()) >= 2:
        fit = linregress(time[fmask], K12[fmask])
        m, b, r = float(fit.slope), float(fit.intercept), float(fit.rvalue)
    else:
        m = b = r = float('nan')
        print(f"[k12] WARNING: only {int(fmask.sum())} finite points in [{a.fitmin},{a.fitmax}]; no fit")

    if a.tmax is not None:
        msk = time <= a.tmax
        time, C12, C12_SE, K12, K12_SE = time[msk], C12[msk], C12_SE[msk], K12[msk], K12_SE[msk]

    out = a.out or (os.path.splitext(a.file)[0] + '_k12.png')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)

    ax1.axhline(0.0, color='0.7', lw=0.8, ls='--')
    ax1.plot(time, C12, color='C3', lw=1.6, label=r'$C_{12}(t)$')
    ax1.fill_between(time, C12 - C12_SE, C12 + C12_SE, color='C3', alpha=0.3, lw=0)
    ax1.set_ylabel(r'$C_{12}(t)$')
    ax1.legend(loc='best', frameon=False)
    ax1.set_title(f"{os.path.basename(a.file)}   P2eq={P2eq:.4f}  (beta={beta}, epsil={epsil})")

    ax2.axhline(0.0, color='0.7', lw=0.8, ls='--')
    ax2.plot(time, K12, color='C0', lw=1.6, label=r'$K_{12}(t)=-P_2^{\mathrm{eq}}\ln|1-C_{12}/P_2^{\mathrm{eq}}|$')
    ax2.fill_between(time, K12 - K12_SE, K12 + K12_SE, color='C0', alpha=0.3, lw=0)
    if np.isfinite(m):
        tf = np.array([a.fitmin, a.fitmax])
        ax2.plot(tf, m * tf + b, color='k', lw=2.0, ls='--',
                 label=fr'fit [{a.fitmin:g},{a.fitmax:g}]:  m={m:.4g}, b={b:.4g}, r={r:.4f}')
        ax2.axvspan(a.fitmin, a.fitmax, color='0.9', zorder=0)   # shade the fit window
    ax2.set_xlabel('time (a.u.)'); ax2.set_ylabel(r'$K_{12}(t)$')
    ax2.legend(loc='best', frameon=False)

    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)

    src = 'config.py' if (cfg is not None) else 'override/fallback'
    print(f"[k12] beta={beta}, epsil={epsil} (from {src}), P2eq={P2eq:.6f}")
    print(f"[k12] line fit of K12 over t=[{a.fitmin},{a.fitmax}]:  m={m:.6g}  b={b:.6g}  r={r:.6f}  (r^2={r*r:.6f})")
    print(f"[k12] wrote {out}  ({time.size} points)")


if __name__ == '__main__':
    _main()
