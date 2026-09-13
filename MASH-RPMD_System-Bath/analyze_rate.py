"""
analyze_rate.py -- C_11(t), C_12(t), M_12(t), the complex k(t) and Mk(t), the rate integrand
K_12(t), and a BLOCK ANALYSIS of
the rate, all in one pass over data.hdf.  The only output is a single 6-panel figure
(analyze_rate.png).

This merges what used to be two scripts (analyze_pop.py -> pop_c12.dat -> analyze_k12.py) and adds
an uncertainty on the extracted rate.

--- correlation functions (identical algebra to analyze_pop.py) -------------------------------------

 C_11(t) = < 1.5 A Sx a(t)Sx(t)
             + A Sx (1 + B(t) sgn(Sz(t)))
             + (1 + B sgn(Sz)) a(t)Sx(t)
             + |Sz| (1 + B sgn(Sz)) (1 + B(t) sgn(Sz(t))) >

 C_12(t) = < -1.5 A Sx a(t)Sx(t)
             + A Sx (1 - B(t) sgn(Sz(t)))
             - (1 + B sgn(Sz)) a(t)Sx(t)
             + |Sz| (1 + B sgn(Sz)) (1 - B(t) sgn(Sz(t))) >

 M_12(t) = < (|Sz(t)| - |Sz|) (1 + B sgn(Sz)) (1 - B(t) sgn(Sz(t))) >

M_12 is exactly the 4th term of C_12 with |Sz| at t=0 replaced by the difference |Sz(t)| - |Sz|;
the two projector factors are untouched.  That leading factor vanishes at t=0, so M_12(0) == 0
identically.  Its error bar is the same one-combined-block SE over --nblocks used for C_11/C_12.

--- complex correlation functions (i is the imaginary unit) -----------------------------------------

 k(t)  = delta * < 1.5 (i B Sx + Sy) (A(t) Sx(t))
                  -   (i B Sx + Sy) (1 - B(t) sgn(Sz(t)))
                  -   (i A sgn(Sz)) (A(t) Sx(t))
                  +   |Sz| (i A sgn(Sz)) (1 - B(t) sgn(Sz(t))) >

 Mk(t) = delta * < (|Sz(t)| - |Sz|) (i A sgn(Sz)) (1 - B(t) sgn(Sz(t))) >

Mk stands to k's 4th term exactly as M_12 stands to C_12's, so Mk(0) == 0 identically.  Note k's
sign pattern (+1.5, -1, -1, +1) is NOT C_12's (-1.5, +1, -1, +1).  Because np.std on a complex
array collapses to one real number, the block SE is taken separately on the real and imaginary
parts; the magnitude band is propagated as sigma_|z| = sqrt((Re*s_Re)^2 + (Im*s_Im)^2)/|z|.

A=a(Rbar), B=b(Rbar) are evaluated at the BEAD-MEAN (centroid) position Rbar = mean_b R_b, and
Sx(t), sgn(Sz(t)), |Sz(t)| are bead means of the per-bead value.  Unprimed quantities are t=0
(row 0 of `time`, a prepared initial condition -- not a sliding time origin).  <...> averages over
trajectories.  The electronic-mixing functions use the corrected definition

    kappa(R) = sqrt(lbd*kvec/2)*R + epsil/2,   norm = sqrt(kappa^2 + delta^2)
    a(R) = -delta/norm,   b(R) = kappa/norm

with (kvec, epsil, lbd, delta) read from the HDF config_json.

--- rate integrand ---------------------------------------------------------------------------------

    P2eq    = 1 / (1 + exp(-beta*epsil))
    K_12(t) = -P2eq * ln| 1 - C_12(t)/P2eq |

with errors propagated through the logarithm: u = 1 - C12/P2eq, dK/dC12 = 1/u, so sigma_K =
sigma_C12 / |u|.  The rate is the slope m of a least-squares line fit of K_12 over [fitmin, fitmax].

--- figure layout (analyze_rate.png, the only output) -----------------------------------------------

                col 0                col 1                        col 2
    row 0   C_11(t) +/- SE       k(t):  Re / Im / |k|        Mk(t): Re / Im / |Mk|
    row 1   C_12(t) +/- SE       K_12(t): trajectory avg     M_12(t) +/- SE
                                 + full line fit, overlaid
                                 on the per-block curves
                                 and their per-block fits

--- block analysis of the rate ----------------------------------------------------------------------

The trajectory axis is split into `nblocks` contiguous blocks (np.array_split, the same split the
block-SEM uses).  Each block gets its OWN C_12 -> K_12 -> line fit, giving one slope per block:

    m_bar   = mean(slopes)
    sigma_m = std(slopes, ddof=1) / sqrt(nblocks)

reported as m = m_bar +/- sigma_m.  A final fit to the entire (trajectory-averaged) K_12 is done
separately and shown in the top-right panel.

CAVEAT: K_12 is only meaningful while u > 0.  A block built from very few trajectories can have
C_12 > P2eq, which drives u through zero and K_12 through the log singularity; those blocks produce
meaningless slopes and badly distort m_bar.  The script warns when it happens -- if you see that
warning, use fewer, larger blocks.

Usage (from the run dir, PYTHONPATH=<RP_MASH root>):
    python analyze_rate.py
    python analyze_rate.py --nblocks 5 --fitmin 20 --fitmax 40
"""

import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
    """a(R), b(R) elementwise (corrected definition, matches analyze_pop.py/analyze_c12.py)."""
    kappa = np.sqrt(lbd * kvec / 2.0) * R + epsil / 2.0
    norm  = np.sqrt(kappa**2 + delta**2)
    return -delta / norm, kappa / norm


def k12_from_c12(C12, P2eq, C12_SE=None):
    """K_12 = -P2eq*ln|1 - C12/P2eq| and, if given an error, its propagated SE = sigma_C12/|u|."""
    u = 1.0 - C12 / P2eq
    with np.errstate(divide='ignore', invalid='ignore'):
        K12 = -P2eq * np.log(np.abs(u))
        SE  = None if C12_SE is None else C12_SE / np.abs(u)
    if SE is not None:
        SE = np.where(np.isfinite(SE), SE, 0.0)     # |u|=0 -> divide by zero -> error 0
    return K12, SE, u


def _mag_and_se(z, se_re, se_im):
    """|z| and its error propagated from the real/imaginary block SEs. |z|=0 -> error 0."""
    mag = np.abs(z)
    with np.errstate(divide='ignore', invalid='ignore'):
        se = np.sqrt((z.real * se_re)**2 + (z.imag * se_im)**2) / mag
    return mag, np.where(np.isfinite(se), se, 0.0)


def line_fit(time, K12, fitmin, fitmax):
    """Least-squares m*t+b of K12 over [fitmin, fitmax], finite points only. NaNs if too few."""
    mask = (time >= fitmin) & (time <= fitmax) & np.isfinite(K12)
    if int(mask.sum()) < 2:
        return float('nan'), float('nan'), float('nan'), int(mask.sum())
    fit = linregress(time[mask], K12[mask])
    return float(fit.slope), float(fit.intercept), float(fit.rvalue), int(mask.sum())


def compute(path, nblocks=None, fitmin=20.0, fitmax=40.0, beta_ovr=None, epsil_ovr=None):
    with open_run(path) as f:
        time = f['time'][:]                     # (T,)
        cfg  = json.loads(f.attrs['config_json'])
        kvec  = float(cfg['kvec']); epsil = float(cfg['epsil'])
        lbd   = float(cfg['lbd']);  delta = float(cfg['delta'])
        beta  = float(cfg['beta'])
        nbds  = int(f.attrs['nbds'])

        R  = f['nucR'][:, :, :, 0]              # (n, T, nbds)
        Sx = f['mapSx'][:]                      # (n, T, nbds)
        Sy = f['mapSy'][:]                      # (n, T, nbds)
        Sz = f['mapSz'][:]                      # (n, T, nbds)

    n_traj = R.shape[0]
    if beta_ovr  is not None: beta  = float(beta_ovr)
    if epsil_ovr is not None: epsil = float(epsil_ovr)

    # ---- bead reduction: A,B evaluated at the centroid Rbar = mean over beads ----
    Rbar  = R.mean(axis=2)
    A, B  = a_b_functions(Rbar, kvec, epsil, lbd, delta)
    Sx_   = Sx.mean(axis=2)                     # Sx(t)
    Sy_   = Sy.mean(axis=2)                     # Sy(t)
    sgnSz = np.sign(Sz).mean(axis=2)            # sgn(Sz(t))  in [-1,1]
    absSz = np.abs(Sz).mean(axis=2)             # |Sz(t)|

    # ---- t=0 scalars (row 0), shape (n,1) to broadcast against (n,T) ----
    A0, Sx0, Sy0, B0, sgn0, abs0 = (X[:, :1] for X in (A, Sx_, Sy_, B, sgnSz, absSz))

    aSx_t   = A * Sx_                            # a(t) Sx(t)
    proj0   = 1.0 + B0 * sgn0                    # (1 + B sgn(Sz))  at t=0
    projt_p = 1.0 + B * sgnSz                    # (1 + B(t) sgn(Sz(t)))
    projt_m = 1.0 - B * sgnSz                    # (1 - B(t) sgn(Sz(t)))
    ASx0    = A0 * Sx0                           # A Sx at t=0

    # ---- full brackets per trajectory (n,T) ----
    G11 = ( 1.5 * ASx0 * aSx_t + ASx0 * projt_p + proj0 * aSx_t + abs0 * proj0 * projt_p)
    G12 = (-1.5 * ASx0 * aSx_t + ASx0 * projt_m - proj0 * aSx_t + abs0 * proj0 * projt_m)

    # M_12: C12's 4th term with |Sz| at t=0 replaced by the difference (|Sz(t)| - |Sz|).
    # The leading factor vanishes at t=0, so M12(0) == 0 identically.
    GM12 = (absSz - abs0) * proj0 * projt_m

    # ---- complex brackets: k(t) and Mk(t).  'i' is the imaginary unit. ----
    W0 = 1j * B0 * Sx0 + Sy0                     # (i B Sx + Sy)  at t=0
    Z0 = 1j * A0 * sgn0                          # (i A sgn(Sz))  at t=0

    # NOTE the sign pattern (+1.5, -1, -1, +1) is NOT C12's (-1.5, +1, -1, +1).
    Gk  = delta * (1.5 * W0 * aSx_t - W0 * projt_m - Z0 * aSx_t + abs0 * Z0 * projt_m)
    # Mk is k's 4th term with |Sz| -> (|Sz(t)| - |Sz|), so Mk(0) == 0 identically.
    GMk = delta * (absSz - abs0) * Z0 * projt_m

    C11 = G11.mean(axis=0);  C11_SE, used_nb = block_sem(G11, nblocks)
    C12 = G12.mean(axis=0);  C12_SE, _       = block_sem(G12, nblocks)
    M12 = GM12.mean(axis=0); M12_SE, _       = block_sem(GM12, nblocks)

    # complex: block_sem on a complex array collapses to ONE real number, so split Re/Im
    kt = Gk.mean(axis=0)
    kt_SE_re, _ = block_sem(Gk.real,  nblocks);  kt_SE_im, _ = block_sem(Gk.imag,  nblocks)
    Mk = GMk.mean(axis=0)
    Mk_SE_re, _ = block_sem(GMk.real, nblocks);  Mk_SE_im, _ = block_sem(GMk.imag, nblocks)
    kt_mag, kt_mag_SE = _mag_and_se(kt, kt_SE_re, kt_SE_im)
    Mk_mag, Mk_mag_SE = _mag_and_se(Mk, Mk_SE_re, Mk_SE_im)

    P2eq = 1.0 / (1.0 + np.exp(-beta * epsil))

    # ---- full-average K12 and its single line fit ----
    K12, K12_SE, u_full = k12_from_c12(C12, P2eq, C12_SE)
    m_full, b_full, r_full, n_full = line_fit(time, K12, fitmin, fitmax)

    # ---- per-block K12 and per-block line fits (same array_split as block_sem) ----
    blocks   = np.array_split(G12, max(1, min(int(used_nb), n_traj)), axis=0)
    win      = (time >= fitmin) & (time <= fitmax)
    K12_blks = []
    blk_m, blk_b = [], []          # per-block, NaN where the fit failed (stays aligned with blocks)
    n_nonphys, n_skipped = 0, 0
    for blk in blocks:
        C12_b = blk.mean(axis=0)
        K12_b, _, u_b = k12_from_c12(C12_b, P2eq)
        K12_blks.append(K12_b)
        if np.any(u_b[win] <= 0.0):
            n_nonphys += 1
        m_b, b_b, _, npts = line_fit(time, K12_b, fitmin, fitmax)
        blk_m.append(m_b); blk_b.append(b_b)
        if not np.isfinite(m_b):
            n_skipped += 1

    blk_m = np.asarray(blk_m); blk_b = np.asarray(blk_b)
    good  = np.isfinite(blk_m)
    slopes, intercepts = blk_m[good], blk_b[good]
    if slopes.size >= 2:
        m_bar   = float(slopes.mean())
        b_bar   = float(intercepts.mean())
        sigma_m = float(slopes.std(ddof=1) / np.sqrt(slopes.size))
    elif slopes.size == 1:
        m_bar, b_bar, sigma_m = float(slopes[0]), float(intercepts[0]), float('nan')
    else:
        m_bar = b_bar = sigma_m = float('nan')

    return dict(time=time, C11=C11, C11_SE=C11_SE, C12=C12, C12_SE=C12_SE,
                M12=M12, M12_SE=M12_SE,
                kt=kt, kt_SE_re=kt_SE_re, kt_SE_im=kt_SE_im,
                kt_mag=kt_mag, kt_mag_SE=kt_mag_SE,
                Mk=Mk, Mk_SE_re=Mk_SE_re, Mk_SE_im=Mk_SE_im,
                Mk_mag=Mk_mag, Mk_mag_SE=Mk_mag_SE,
                K12=K12, K12_SE=K12_SE, K12_blks=K12_blks,
                m_full=m_full, b_full=b_full, r_full=r_full, n_full=n_full,
                slopes=slopes, intercepts=intercepts, blk_m=blk_m, blk_b=blk_b,
                m_bar=m_bar, b_bar=b_bar, sigma_m=sigma_m,
                n_nonphys=n_nonphys, n_skipped=n_skipped,
                P2eq=P2eq, beta=beta, n_traj=n_traj, nbds=nbds, nblocks=used_nb,
                fitmin=fitmin, fitmax=fitmax,
                params=dict(kvec=kvec, epsil=epsil, lbd=lbd, delta=delta))


def _fmt_sci(x, sig=4):
    """Format x as mathtext, e.g. -2.442\\times10^{-3}; plain decimal for moderate magnitudes."""
    if not np.isfinite(x):
        return 'nan'
    if x == 0.0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    if -3 < e < 4:
        return f'{x:.{sig}g}'
    return rf'{x / 10.0**e:.{sig}g}\times10^{{{e}}}'


def _block_colors(nb):
    """Distinct color per block: tab10 up to 10 blocks, else a viridis ramp."""
    if nb <= 10:
        return [plt.cm.tab10(i % 10) for i in range(nb)]
    return [plt.cm.viridis(x) for x in np.linspace(0.0, 0.9, nb)]


def _complex_panel(axp, t, z, se_re, se_im, mag, mag_se, label):
    """Re / Im / |.| of a complex correlation function, each with its own +/- SE band."""
    axp.axhline(0.0, color='0.7', lw=0.8, ls='--')
    axp.plot(t, z.real, color='C0', lw=1.4, label=rf'$\mathrm{{Re}}\,{label}$')
    axp.fill_between(t, z.real - se_re, z.real + se_re, color='C0', alpha=0.3, lw=0)
    axp.plot(t, z.imag, color='C1', lw=1.4, label=rf'$\mathrm{{Im}}\,{label}$')
    axp.fill_between(t, z.imag - se_im, z.imag + se_im, color='C1', alpha=0.3, lw=0)
    axp.plot(t, mag, color='k', lw=1.4, label=rf'$|{label}|$')
    axp.fill_between(t, mag - mag_se, mag + mag_se, color='k', alpha=0.2, lw=0)
    axp.legend(loc='best', frameon=False, fontsize=9)


def _plot(out, r, tmax=None):
    t = r['time']
    C11, C11_SE, C12, C12_SE = r['C11'], r['C11_SE'], r['C12'], r['C12_SE']
    M12, M12_SE              = r['M12'], r['M12_SE']
    K12, K12_SE, K12_blks    = r['K12'], r['K12_SE'], r['K12_blks']
    kt, kt_re, kt_im, kt_mag, kt_mag_SE = (r['kt'], r['kt_SE_re'], r['kt_SE_im'],
                                           r['kt_mag'], r['kt_mag_SE'])
    Mk, Mk_re, Mk_im, Mk_mag, Mk_mag_SE = (r['Mk'], r['Mk_SE_re'], r['Mk_SE_im'],
                                           r['Mk_mag'], r['Mk_mag_SE'])

    if tmax is not None:                         # truncate for plotting only (fits already done)
        msk = t <= tmax
        t, C11, C11_SE, C12, C12_SE = t[msk], C11[msk], C11_SE[msk], C12[msk], C12_SE[msk]
        M12, M12_SE = M12[msk], M12_SE[msk]
        K12, K12_SE = K12[msk], K12_SE[msk]
        K12_blks    = [k[msk] for k in K12_blks]
        kt, kt_re, kt_im, kt_mag, kt_mag_SE = (X[msk] for X in
                                               (kt, kt_re, kt_im, kt_mag, kt_mag_SE))
        Mk, Mk_re, Mk_im, Mk_mag, Mk_mag_SE = (X[msk] for X in
                                               (Mk, Mk_re, Mk_im, Mk_mag, Mk_mag_SE))

    fmin, fmax = r['fitmin'], r['fitmax']
    tf = np.array([fmin, fmax])

    # sharex=True across all six panels; sharey left at its default (False)
    fig, ax = plt.subplots(2, 3, figsize=(19, 9), sharex=True)

    # ---------------- top-left: C11 ----------------
    a11 = ax[0, 0]
    a11.axhline(1.0, color='0.7', lw=0.8, ls='--')
    a11.plot(t, C11, color='C0', lw=1.6, label=r'$C_{11}(t)$')
    a11.fill_between(t, C11 - C11_SE, C11 + C11_SE, color='C0', alpha=0.3, lw=0)
    a11.set_ylabel(r'$C_{11}(t)$')
    a11.legend(loc='best', frameon=False)

    # ---------------- bottom-left: C12 ----------------
    a12 = ax[1, 0]
    a12.axhline(0.0, color='0.7', lw=0.8, ls='--')
    a12.plot(t, C12, color='C3', lw=1.6, label=r'$C_{12}(t)$')
    a12.fill_between(t, C12 - C12_SE, C12 + C12_SE, color='C3', alpha=0.3, lw=0)
    a12.set_xlabel('time (a.u.)'); a12.set_ylabel(r'$C_{12}(t)$')
    a12.legend(loc='best', frameon=False)

    # ---- bottom-middle: K12 -- full average + fit AND the per-block curves + fits ----
    ak = ax[1, 1]
    ak.axvspan(fmin, fmax, color='0.92', zorder=0)
    ak.axhline(0.0, color='0.7', lw=0.8, ls='--')

    # per-block curves first, each with its own fit in the SAME colour (unlabelled)
    colors = _block_colors(len(K12_blks))
    for i, K_b in enumerate(K12_blks):
        c = colors[i]
        ak.plot(t, K_b, color=c, lw=0.9, alpha=0.55)
        if np.isfinite(r['blk_m'][i]):
            ak.plot(tf, r['blk_m'][i] * tf + r['blk_b'][i], color=c, lw=1.6, alpha=1.0)

    # the trajectory-averaged K12 sits on top of the block traces
    ak.plot(t, K12, color='C0', lw=1.6, zorder=4,
            label=r'$K_{12}(t)=-P_2^{\mathrm{eq}}\ln|1-C_{12}/P_2^{\mathrm{eq}}|$')
    ak.fill_between(t, K12 - K12_SE, K12 + K12_SE, color='C0', alpha=0.3, lw=0, zorder=3)
    if np.isfinite(r['m_full']):
        ak.plot(tf, r['m_full'] * tf + r['b_full'], color='k', lw=2.0, ls='--', zorder=6,
                label=(fr"fit [{fmin:g},{fmax:g}]:  m={r['m_full']:.4g}, "
                       fr"b={r['b_full']:.4g}, r={r['r_full']:.4f}"))
    # average of the block fits: black SOLID, unlabelled, with its equation and the block result
    if np.isfinite(r['m_bar']):
        ak.plot(tf, r['m_bar'] * tf + r['b_bar'], color='k', lw=2.5, zorder=5)
        ak.text(0.03, 0.12,
                rf"$\bar{{y}} = {_fmt_sci(r['m_bar'])}\,x + {r['b_bar']:.4g}$",
                transform=ak.transAxes, fontsize=10, va='bottom', ha='left')
        ak.text(0.03, 0.03,
                rf"$m = {_fmt_sci(r['m_bar'])} \pm {_fmt_sci(r['sigma_m'])}$",
                transform=ak.transAxes, fontsize=10, va='bottom', ha='left')
    ak.set_xlabel('time (a.u.)'); ak.set_ylabel(r'$K_{12}(t)$')
    ak.legend(loc='best', frameon=False, fontsize=8)   # only the full-average items are labelled

    # Blocks that cross the log singularity spike arbitrarily high and would flatten every fit
    # line into an unreadable smear, so clip the view to a robust range (all curves are still
    # drawn -- this only sets the y-limits). The pool includes the trajectory-averaged curve and
    # every fit line so neither can be clipped out of frame by the spiky block percentiles.
    finite = np.concatenate([k[np.isfinite(k)] for k in K12_blks if np.isfinite(k).any()]) \
        if any(np.isfinite(k).any() for k in K12_blks) else np.array([0.0])
    lo, hi = np.percentile(finite, [1.0, 99.0])
    band = np.concatenate([K12 - K12_SE, K12 + K12_SE])
    band = band[np.isfinite(band)]
    if band.size:
        lo, hi = min(lo, band.min()), max(hi, band.max())
    ends = [v for i in range(len(K12_blks)) if np.isfinite(r['blk_m'][i])
            for v in (r['blk_m'][i] * tf + r['blk_b'][i])]
    if np.isfinite(r['m_bar']):
        ends += list(r['m_bar'] * tf + r['b_bar'])
    if np.isfinite(r['m_full']):
        ends += list(r['m_full'] * tf + r['b_full'])
    if ends:
        lo, hi = min(lo, min(ends)), max(hi, max(ends))
    pad = 0.12 * (hi - lo) if hi > lo else 1.0
    ak.set_ylim(lo - pad, hi + pad)

    # ---------------- bottom-right: M12 ----------------
    am = ax[1, 2]
    am.axhline(0.0, color='0.7', lw=0.8, ls='--')
    am.plot(t, M12, color='C2', lw=1.6, label=r'$M_{12}(t)$')
    am.fill_between(t, M12 - M12_SE, M12 + M12_SE, color='C2', alpha=0.3, lw=0)
    am.set_xlabel('time (a.u.)'); am.set_ylabel(r'$M_{12}(t)$')
    am.legend(loc='best', frameon=False)

    # ---------------- top-middle: k(t) (complex) ----------------
    _complex_panel(ax[0, 1], t, kt, kt_re, kt_im, kt_mag, kt_mag_SE, r'k(t)')
    ax[0, 1].set_ylabel(r'$k(t)$')

    # ---------------- top-right: Mk(t) (complex) ----------------
    _complex_panel(ax[0, 2], t, Mk, Mk_re, Mk_im, Mk_mag, Mk_mag_SE, r'M_k(t)')
    ax[0, 2].set_ylabel(r'$M_k(t)$')

    p = r['params']
    fig.suptitle(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}   "
                 f"P2eq={r['P2eq']:.4f} (beta={r['beta']:g}, epsil={p['epsil']:g})   "
                 f"fit window [{fmin:g}, {fmax:g}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='C11, C12, M12, k, Mk, K12 and a block analysis of the rate from an RP-MASH data.hdf.')
    ap.add_argument('--file',    default='data.hdf')
    ap.add_argument('--nblocks', type=int, default=10, help='trajectory blocks (default 10)')
    ap.add_argument('--fitmin',  type=float, default=20.0, help='fit window start (default 20)')
    ap.add_argument('--fitmax',  type=float, default=40.0, help='fit window end (default 40)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--beta',    type=float, default=None, help='override full inverse temperature')
    ap.add_argument('--epsil',   type=float, default=None, help='override driving force epsil')
    ap.add_argument('--out',     default='analyze_rate.png', help='output png')
    a = ap.parse_args()

    r = compute(a.file, nblocks=a.nblocks, fitmin=a.fitmin, fitmax=a.fitmax,
                beta_ovr=a.beta, epsil_ovr=a.epsil)
    _plot(a.out, r, tmax=a.tmax)

    p = r['params']
    print(f"[rate] n_traj={r['n_traj']} nbds={r['nbds']} nblocks={r['nblocks']}")
    print(f"[rate] params: kvec={p['kvec']} epsil={p['epsil']} lbd={p['lbd']} delta={p['delta']:.6e}")
    print(f"[rate] beta={r['beta']:g}  P2eq={r['P2eq']:.6f}")
    print(f"[rate] C11(0)={r['C11'][0]:.6e} +/- {r['C11_SE'][0]:.3e}")
    print(f"[rate] C12(0)={r['C12'][0]:.6e} +/- {r['C12_SE'][0]:.3e}   "
          f"C12 range [{r['C12'].min():.6e}, {r['C12'].max():.6e}]")
    print(f"[rate] M12(0)={r['M12'][0]:.6e} (exact 0 expected) +/- {r['M12_SE'][0]:.3e}   "
          f"M12 range [{r['M12'].min():.6e}, {r['M12'].max():.6e}]")
    print(f"[rate] k(0)={r['kt'][0].real:+.6e}{r['kt'][0].imag:+.6e}j   "
          f"|k| range [{r['kt_mag'].min():.6e}, {r['kt_mag'].max():.6e}]")
    print(f"[rate] Mk(0)={r['Mk'][0].real:+.6e}{r['Mk'][0].imag:+.6e}j (exact 0 expected)   "
          f"|Mk| range [{r['Mk_mag'].min():.6e}, {r['Mk_mag'].max():.6e}]")
    print(f"[rate] FULL fit of K12 over t=[{r['fitmin']:g},{r['fitmax']:g}] "
          f"({r['n_full']} pts):  m={r['m_full']:.6g}  b={r['b_full']:.6g}  r={r['r_full']:.6f}")
    print(f"[rate] BLOCK analysis ({r['slopes'].size} usable blocks):  "
          f"m = {r['m_bar']:.6g} +/- {r['sigma_m']:.6g}")
    if r['slopes'].size:
        print(f"[rate] block slopes: {np.array2string(r['slopes'], precision=4)}")
    if r['n_nonphys']:
        print(f"[rate] WARNING: {r['n_nonphys']} of {r['nblocks']} blocks have "
              f"u = 1 - C12/P2eq <= 0 inside the fit window; K12 crosses the log singularity "
              f"there, so those block slopes are unphysical and distort the mean. "
              f"Use fewer, larger blocks (--nblocks).")
    if r['n_skipped']:
        print(f"[rate] WARNING: {r['n_skipped']} block(s) had <2 finite points in the window "
              f"and were skipped.")
    print(f"[rate] wrote {a.out}")


if __name__ == '__main__':
    _main()
