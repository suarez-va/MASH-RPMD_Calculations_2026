"""
analyze_p2.py -- the quantum-jump-weighted population P2(t), the rate integrand K(t), and a BLOCK
ANALYSIS of the rate, in one streaming pass over data.hdf.  Output is a single 3-panel figure
(analyze_p2.png).

This is the jump-aware sibling of analyze_rate.py.  Where that script's four terms carry constant
coefficients and a |Sz(0)| factor, here each term carries one of the four quantum-jump weights
W_AB^(m), which depend on every jump the trajectory has taken up to t.

--- the observable ---------------------------------------------------------------------------------

 P2(t) =  1/2 < (1 + K Sgn) W_PP^(m) (1 - K(t) sgn(Sz(t))) >
        - 1/2 < D Sx        W_CP^(m) (1 - K(t) sgn(Sz(t))) >
        + 1/2 < (1 + K Sgn) W_PC^(m) D(t) Sx(t) >
        - 1/2 < D Sx        W_CC^(m) D(t) Sx(t) >

with Sgn = sgn(Sz(0)), and unprimed quantities evaluated at t=0 (row 0 of `time`, a prepared initial
condition -- not a sliding time origin).  <...> averages over trajectories.  The electronic-mixing
functions are

    kappa(R) = sqrt(lbd*kvec/2)*R + epsil/2,   Delta = delta,   norm = sqrt(kappa^2 + Delta^2)
    K(R) = kappa/norm,    D(R) = Delta/norm

(Delta is a constant in ET_with_bath despite the D(R(t)) notation.)  NOTE the relation to
analyze_rate.py's a_b_functions, which returns a = -delta/norm and b = kappa/norm: K == b but
D == -a.  K and D are built directly here so that sign cannot be lost in translation.

The first subscript of W_AB is the t=0 factor and the second is the t factor: W_PP multiplies the
projector-like (1 + K Sgn) at BOTH ends, W_CC the coherence-like D Sx at both ends, and W_CP / W_PC
the mixed pairs.  So the integrand factorizes, which is how it is evaluated:

    G = 1/2 [ (P0 W_PP - C0 W_CP) Pt  +  (P0 W_PC - C0 W_CC) Ct ]
    P0 = 1 + K(0) sgn(Sz(0))    C0 = D(0) Sx(0)
    Pt = 1 - K(t) sgn(Sz(t))    Ct = D(t) Sx(t)

--- the weights ------------------------------------------------------------------------------------

W_AB^(m) is taken at m(t) = #{ jumps with t_jump <= t }, i.e. every jump up to AND INCLUDING t.
That matches mash_rpmd, which writes the post-jump state at t_jump, so W and the spin refer to the
same instant.  The weights live in data.hdf as a ragged per-jump table (/weights, /n_weights) and are
expanded onto the time grid by workflow.expand_weights.

A run consolidated WITHOUT the weights datasets is refused rather than silently analysed, since the
result would look plausible and be wrong.  --w0-only forces the no-jump baseline W^(0) =
(2|Sz(0)|, 2, 2, 3) instead, which is the right comparison curve when you want to see what the jumps
are doing.

--- rate integrand ---------------------------------------------------------------------------------

    P2eq = 1 / (1 + exp(-beta*epsil))
    K12(t) = -P2eq * ln| 1 - P2(t)/P2eq |

with errors propagated through the logarithm: u = 1 - P2/P2eq, dK/dP2 = 1/u, so sigma_K =
sigma_P2 / |u|.  The rate is the slope of a least-squares line fit of K12 over [fitmin, fitmax].

--- block analysis and memory ----------------------------------------------------------------------

The trajectory axis is split into `nblocks` contiguous blocks (np.array_split, the same split
analyze_rate.py's block-SEM uses).  Each block gets its own P2 -> K12 -> line fit, giving one slope
per block:

    m_bar   = mean(slopes)
    sigma_m = std(slopes, ddof=1) / sqrt(nblocks)

Unlike analyze_rate.py this script never materializes the (n_traj, T) arrays.  At n_traj = 1e5 and
T = 601 those would be several GB.  Instead trajectories are read in slabs and only two accumulators
are kept -- the running total and the per-block running sums -- so peak memory is O(nblocks*T +
slab*T) at any n_traj.  Block membership comes from the same contiguous split, so the result is
IDENTICAL to the whole-array form, not an approximation.

CAVEAT: K12 is only meaningful while u > 0.  A block built from few trajectories can have P2 > P2eq,
driving u through zero and K12 through the log singularity; those blocks produce meaningless slopes.
The script warns when it happens -- if you see that warning, use fewer, larger blocks.

Usage (from the run dir, PYTHONPATH=<RP_MASH root>):
    python analyze_p2.py
    python analyze_p2.py --nblocks 100 --fitmin 5 --fitmax 15
    python analyze_p2.py --w0-only --out analyze_p2_nojump.png
"""

import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

from workflow import open_run, expand_weights   # h5py.File(path,'r'); per-jump -> dense weights


def block_sem_from_means(block_means, n_blocks):
    """SE over trajectories from ALREADY-FORMED block means (same estimator as analyze_rate.py)."""
    if n_blocks < 2:
        return np.zeros(block_means.shape[1:])
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def k_d_functions(R, kvec, epsil, lbd, delta):
    """
    K(R) = kappa/norm and D(R) = Delta/norm, elementwise.

    Related to analyze_rate.py's a_b_functions by K == b and D == -a (that script returns
    a = -delta/norm). Written out here so the sign of D is explicit.
    """
    kappa = np.sqrt(lbd * kvec / 2.0) * R + epsil / 2.0
    norm  = np.sqrt(kappa**2 + delta**2)
    return kappa / norm, delta / norm


def k12_from_p2(P2, P2eq, P2_SE=None):
    """K12 = -P2eq*ln|1 - P2/P2eq| and, if given an error, its propagated SE = sigma_P2/|u|."""
    u = 1.0 - P2 / P2eq
    with np.errstate(divide='ignore', invalid='ignore'):
        K12 = -P2eq * np.log(np.abs(u))
        SE  = None if P2_SE is None else P2_SE / np.abs(u)
    if SE is not None:
        SE = np.where(np.isfinite(SE), SE, 0.0)     # |u|=0 -> divide by zero -> error 0
    return K12, SE, u


def line_fit(time, K12, fitmin, fitmax):
    """Least-squares m*t+b of K12 over [fitmin, fitmax], finite points only. NaNs if too few."""
    mask = (time >= fitmin) & (time <= fitmax) & np.isfinite(K12)
    if int(mask.sum()) < 2:
        return float('nan'), float('nan'), float('nan'), int(mask.sum())
    fit = linregress(time[mask], K12[mask])
    return float(fit.slope), float(fit.intercept), float(fit.rvalue), int(mask.sum())


def compute(path, nblocks=None, fitmin=5.0, fitmax=15.0, beta_ovr=None, epsil_ovr=None,
            slab=5000, w0_only=False):
    with open_run(path) as f:
        time = f['time'][:]                     # (T,)
        cfg  = json.loads(f.attrs['config_json'])
        kvec  = float(cfg['kvec']); epsil = float(cfg['epsil'])
        lbd   = float(cfg['lbd']);  delta = float(cfg['delta'])
        beta  = float(cfg['beta'])
        nbds  = int(f.attrs['nbds'])
        n_traj = f['nucR'].shape[0]

        if beta_ovr  is not None: beta  = float(beta_ovr)
        if epsil_ovr is not None: epsil = float(epsil_ovr)

        has_w = 'weights' in f
        if not has_w and not w0_only:
            raise KeyError(
                f"{path} has no /weights dataset. Either the run was made without Tjump, or it was "
                f"consolidated with a version of workflow.consolidate that predates the quantum-jump "
                f"weights. Re-run `python -m workflow.consolidate --config <config.py>` to pick them "
                f"up, or pass --w0-only to deliberately analyse the no-jump baseline W^(0).")

        # contiguous blocks, identical to np.array_split(..., axis=0) over the trajectory axis
        nb = n_traj if nblocks is None else max(1, min(int(nblocks), n_traj))
        block_of = np.concatenate([np.full(len(b), i, dtype=int)
                                   for i, b in enumerate(np.array_split(np.arange(n_traj), nb))])
        block_sizes = np.bincount(block_of, minlength=nb).astype(float)

        T = time.size
        tot  = np.zeros(T)                       # running sum of G over all trajectories
        blk  = np.zeros((nb, T))                 # per-block running sums
        terms = np.zeros((4, T))                 # the four terms separately, for the diagnostic panel

        # ---- streaming pass: one slab of trajectories at a time ----
        for lo in range(0, n_traj, slab):
            hi  = min(lo + slab, n_traj)
            sel = slice(lo, hi)

            R  = f['nucR'][sel, :, :, 0]        # (s, T, nbds)
            Sx = f['mapSx'][sel]                # (s, T, nbds)
            Sz = f['mapSz'][sel]                # (s, T, nbds)   (Sy is not used by P2)

            # bead reduction, matching analyze_rate.py: K,D at the centroid Rbar; sign/abs before
            # the bead mean. For nbds == 1 (the Classical runs) every convention coincides.
            Rbar  = R.mean(axis=2)
            K, D  = k_d_functions(Rbar, kvec, epsil, lbd, delta)
            Sx_   = Sx.mean(axis=2)
            sgnSz = np.sign(Sz).mean(axis=2)

            K0, D0, Sx0, sgn0 = (X[:, :1] for X in (K, D, Sx_, sgnSz))
            P0 = 1.0 + K0 * sgn0                 # (s,1)  projector-like, t=0
            C0 = D0 * Sx0                        # (s,1)  coherence-like, t=0
            Pt = 1.0 - K * sgnSz                 # (s,T)  projector-like, t
            Ct = D * Sx_                         # (s,T)  coherence-like, t

            if w0_only:
                # the no-jump baseline: W^(0) = (2|Sz(0)|, 2, 2, 3), constant in t
                absSz0 = np.abs(Sz[:, :1].mean(axis=2))
                W = {'W_PP': 2.0 * absSz0, 'W_CP': np.full_like(absSz0, 2.0),
                     'W_PC': np.full_like(absSz0, 2.0), 'W_CC': np.full_like(absSz0, 3.0)}
            else:
                W = expand_weights(f, time=time, sel=sel)

            t1 =  0.5 * P0 * W['W_PP'] * Pt
            t2 = -0.5 * C0 * W['W_CP'] * Pt
            t3 =  0.5 * P0 * W['W_PC'] * Ct
            t4 = -0.5 * C0 * W['W_CC'] * Ct
            G  = t1 + t2 + t3 + t4               # (s,T) -- the weighted integrand, per trajectory

            tot += G.sum(axis=0)
            for i, tt in enumerate((t1, t2, t3, t4)):
                terms[i] += tt.sum(axis=0)
            # scatter-add each trajectory's G into its block's running sum
            np.add.at(blk, block_of[lo:hi], G)

    P2          = tot / n_traj
    block_means = blk / block_sizes[:, None]
    P2_SE       = block_sem_from_means(block_means, nb)
    terms      /= n_traj

    P2eq = 1.0 / (1.0 + np.exp(-beta * epsil))

    # ---- full-average K12 and its single line fit ----
    K12, K12_SE, u_full = k12_from_p2(P2, P2eq, P2_SE)
    m_full, b_full, r_full, n_full = line_fit(time, K12, fitmin, fitmax)

    # ---- per-block K12 and per-block line fits ----
    win = (time >= fitmin) & (time <= fitmax)
    K12_blks, blk_m, blk_b = [], [], []
    n_nonphys, n_skipped = 0, 0
    for i in range(nb):
        K12_b, _, u_b = k12_from_p2(block_means[i], P2eq)
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

    return dict(time=time, P2=P2, P2_SE=P2_SE, terms=terms,
                K12=K12, K12_SE=K12_SE, K12_blks=K12_blks,
                m_full=m_full, b_full=b_full, r_full=r_full, n_full=n_full,
                slopes=slopes, intercepts=intercepts, blk_m=blk_m, blk_b=blk_b,
                m_bar=m_bar, b_bar=b_bar, sigma_m=sigma_m,
                n_nonphys=n_nonphys, n_skipped=n_skipped,
                P2eq=P2eq, beta=beta, n_traj=n_traj, nbds=nbds, nblocks=nb,
                fitmin=fitmin, fitmax=fitmax, w0_only=w0_only,
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


def _plot(out, r, tmax=None):
    t = r['time']
    P2, P2_SE = r['P2'], r['P2_SE']
    K12, K12_SE, K12_blks = r['K12'], r['K12_SE'], r['K12_blks']
    terms = r['terms']

    if tmax is not None:                         # truncate for plotting only (fits already done)
        msk = t <= tmax
        t, P2, P2_SE = t[msk], P2[msk], P2_SE[msk]
        K12, K12_SE  = K12[msk], K12_SE[msk]
        K12_blks     = [k[msk] for k in K12_blks]
        terms        = terms[:, msk]

    fmin, fmax = r['fitmin'], r['fitmax']
    tf = np.array([fmin, fmax])

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), sharex=True)

    # ---------------- left: P2 ----------------
    a0 = ax[0]
    a0.axhline(0.0, color='0.7', lw=0.8, ls='--')
    a0.axhline(r['P2eq'], color='0.5', lw=0.9, ls=':', label=r'$P_2^{\mathrm{eq}}$')
    a0.plot(t, P2, color='C0', lw=1.6, label=r'$P_2(t)$')
    a0.fill_between(t, P2 - P2_SE, P2 + P2_SE, color='C0', alpha=0.3, lw=0)
    a0.set_xlabel('time (a.u.)'); a0.set_ylabel(r'$P_2(t)$')
    a0.legend(loc='best', frameon=False)

    # ---------------- middle: K12, full + per-block ----------------
    ak = ax[1]
    ak.axvspan(fmin, fmax, color='0.92', zorder=0)
    ak.axhline(0.0, color='0.7', lw=0.8, ls='--')
    colors = _block_colors(len(K12_blks))
    for i, K_b in enumerate(K12_blks):
        ak.plot(t, K_b, color=colors[i], lw=0.9, alpha=0.55)
        if np.isfinite(r['blk_m'][i]):
            ak.plot(tf, r['blk_m'][i] * tf + r['blk_b'][i], color=colors[i], lw=1.6)
    ak.plot(t, K12, color='C0', lw=1.6, zorder=4,
            label=r'$K_{12}(t)=-P_2^{\mathrm{eq}}\ln|1-P_2/P_2^{\mathrm{eq}}|$')
    ak.fill_between(t, K12 - K12_SE, K12 + K12_SE, color='C0', alpha=0.3, lw=0, zorder=3)
    if np.isfinite(r['m_full']):
        ak.plot(tf, r['m_full'] * tf + r['b_full'], color='k', lw=2.0, ls='--', zorder=6,
                label=(fr"fit [{fmin:g},{fmax:g}]:  m={r['m_full']:.4g}, "
                       fr"b={r['b_full']:.4g}, r={r['r_full']:.4f}"))
    if np.isfinite(r['m_bar']):
        ak.plot(tf, r['m_bar'] * tf + r['b_bar'], color='k', lw=2.5, zorder=5)
        ak.text(0.03, 0.03, rf"$m = {_fmt_sci(r['m_bar'])} \pm {_fmt_sci(r['sigma_m'])}$",
                transform=ak.transAxes, fontsize=10, va='bottom', ha='left')
    ak.set_xlabel('time (a.u.)'); ak.set_ylabel(r'$K_{12}(t)$')
    ak.legend(loc='best', frameon=False, fontsize=8)

    # clip the view: blocks crossing the log singularity spike arbitrarily high
    finite = np.concatenate([k[np.isfinite(k)] for k in K12_blks if np.isfinite(k).any()]) \
        if any(np.isfinite(k).any() for k in K12_blks) else np.array([0.0])
    lo, hi = np.percentile(finite, [1.0, 99.0])
    band = np.concatenate([K12 - K12_SE, K12 + K12_SE]); band = band[np.isfinite(band)]
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

    # ---------------- right: the four weighted terms ----------------
    at = ax[2]
    at.axhline(0.0, color='0.7', lw=0.8, ls='--')
    for i, (lab, c) in enumerate(((r'$+\frac{1}{2}P_0 W_{PP} P_t$', 'C0'),
                                  (r'$-\frac{1}{2}C_0 W_{CP} P_t$', 'C3'),
                                  (r'$+\frac{1}{2}P_0 W_{PC} C_t$', 'C2'),
                                  (r'$-\frac{1}{2}C_0 W_{CC} C_t$', 'C1'))):
        at.plot(t, terms[i], color=c, lw=1.4, label=lab)
    at.plot(t, terms.sum(axis=0), color='k', lw=1.8, ls='--', label=r'sum $= P_2(t)$')
    at.set_xlabel('time (a.u.)'); at.set_ylabel('term contribution')
    at.legend(loc='best', frameon=False, fontsize=8)

    p = r['params']
    fig.suptitle(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}   "
                 f"P2eq={r['P2eq']:.6f} (beta={r['beta']:g}, epsil={p['epsil']:g})   "
                 f"fit window [{fmin:g}, {fmax:g}]"
                 + ("   [W^(0) ONLY -- no-jump baseline]" if r['w0_only'] else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Jump-weighted P2(t), K12(t) and a block analysis of the rate from an RP-MASH data.hdf.')
    ap.add_argument('--file',    default='data.hdf')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--fitmin',  type=float, default=5.0, help='fit window start (default 5)')
    ap.add_argument('--fitmax',  type=float, default=15.0, help='fit window end (default 15)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--beta',    type=float, default=None, help='override full inverse temperature')
    ap.add_argument('--epsil',   type=float, default=None, help='override driving force epsil')
    ap.add_argument('--slab',    type=int, default=5000,
                    help='trajectories read per streaming pass (default 5000; lower it if RAM is tight)')
    ap.add_argument('--w0-only', action='store_true',
                    help='ignore the jump weights and use W^(0)=(2|Sz0|,2,2,3): the no-jump baseline')
    ap.add_argument('--out',     default='analyze_p2.png', help='output png')
    a = ap.parse_args()

    r = compute(a.file, nblocks=a.nblocks, fitmin=a.fitmin, fitmax=a.fitmax,
                beta_ovr=a.beta, epsil_ovr=a.epsil, slab=a.slab, w0_only=a.w0_only)
    _plot(a.out, r, tmax=a.tmax)

    p = r['params']
    print(f"[p2] n_traj={r['n_traj']} nbds={r['nbds']} nblocks={r['nblocks']}"
          + ("  [W^(0) ONLY]" if r['w0_only'] else ""))
    print(f"[p2] params: kvec={p['kvec']} epsil={p['epsil']} lbd={p['lbd']} delta={p['delta']:.6e}")
    print(f"[p2] beta={r['beta']:g}  P2eq={r['P2eq']:.6f}")
    print(f"[p2] P2(0)={r['P2'][0]:.6e} +/- {r['P2_SE'][0]:.3e}   "
          f"P2 range [{r['P2'].min():.6e}, {r['P2'].max():.6e}]")
    print(f"[p2] term means at t=0: " + ', '.join(f'{v:.4e}' for v in r['terms'][:, 0]))
    print(f"[p2] FULL fit of K12 over t=[{r['fitmin']:g},{r['fitmax']:g}] "
          f"({r['n_full']} pts):  m={r['m_full']:.6g}  b={r['b_full']:.6g}  r={r['r_full']:.6f}")
    print(f"[p2] BLOCK analysis ({r['slopes'].size} usable blocks):  "
          f"m = {r['m_bar']:.6g} +/- {r['sigma_m']:.6g}")
    if r['n_nonphys']:
        print(f"[p2] WARNING: {r['n_nonphys']} of {r['nblocks']} blocks have u = 1 - P2/P2eq <= 0 "
              f"inside the fit window; K12 crosses the log singularity there, so those block slopes "
              f"are unphysical and distort the mean. Use fewer, larger blocks (--nblocks).")
    if r['n_skipped']:
        print(f"[p2] WARNING: {r['n_skipped']} block(s) had <2 finite points in the window "
              f"and were skipped.")
    print(f"[p2] wrote {a.out}")


if __name__ == '__main__':
    _main()
