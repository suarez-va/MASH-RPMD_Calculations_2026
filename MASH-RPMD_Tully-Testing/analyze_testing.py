"""
analyze_testing.py -- the observables of a MASH-RPMD_Tully-Testing run, stitched across its stages
from the stage_<m>/summary.npz files that `iterate.py resample` writes (no trajectory is read here).

Figure 1 (analyze_testing.png), 2x2:
    top-left     Rg, Tg, Re, Te: int h(-+q) <g|rho|g>, int h(-+q) <e|rho|e>  (R/T = q </> 0)
    top-right    Tr rho = Pg + Pe, and Pg, Pe
    bottom-left  int rho_x, int rho_y
    bottom-right the cancellation at every resampling: per component sum|W| over the endpoints
                 (raw), sum|B| over the histogram cells (binned), and |sum W| (net) -- the closer
                 binned is to net, the less sign noise goes into the next stage

Figure 2 (analyze_testing_points.png): every stage's initial points in (q,p), one row per component
(I, z, x, y), one column per stage, coloured by the signed weight.

Stage m covers [t_m, t_m+1): its last row (just before the resampling) is dropped in favour of the
next stage's first row, the convention of a quantum jump in analyze_tully.py.  Bands are the SEs of
each stage given its initial points (see iterate.py).

--ref <shards> overlays analyze_tully.py's C_Rg, C_Tg, C_Re, C_Te of a MASH-RPMD_Tully-Delta run
(dashed).  With the stage boundaries as its jump_index (k = m * seg_time / delt) and the same
initial state, the two agree on average, so this is the end-to-end test of the stage update.

    python analyze_testing.py --config config.py
    python analyze_testing.py --config config.py --ref ../MASH-RPMD_Tully-Delta/run/shards --ref-ncores 8
"""

import os
import sys
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMPONENTS = ('I', 'z', 'x', 'y')


def load_stages(run_dir):
    """The summaries of every resampled stage, in stage order."""
    files = sorted(glob.glob(os.path.join(run_dir, 'stage_*', 'summary.npz')))
    if not files:
        raise SystemExit(f'[test] no stage_*/summary.npz under {run_dir}: run iterate.py first')
    out = []
    for fn in files:
        with np.load(fn) as z:
            out.append({k: z[k] for k in z.files})
    return out


def stitch(stages):
    """One time axis over all stages: each stage minus its last row, the final stage whole."""
    names = list(stages[0]['names'])
    t, tot, se = [], [], []
    for m, s in enumerate(stages):
        keep = slice(None) if m == len(stages) - 1 else slice(0, -1)
        t.append(s['time'][keep]); tot.append(s['tot'][:, keep]); se.append(s['se'][:, keep])
    t, tot, se = np.concatenate(t), np.concatenate(tot, axis=1), np.concatenate(se, axis=1)
    r = {k: tot[i] for i, k in enumerate(names)}
    r.update({k + '_SE': se[i] for i, k in enumerate(names)})
    r['time'] = t
    r['t_starts'] = np.array([float(s['t_start']) for s in stages])
    return r


def _panel(ax, t, r, keys, labels, colors, ylabel, bounds):
    ax.axhline(0.0, color='0.7', lw=0.8, ls='--')
    for tb in bounds:
        ax.axvline(tb, color='0.6', lw=0.8, ls=':')
    for k, lab, c in zip(keys, labels, colors):
        ax.plot(t, r[k], color=c, lw=1.6, label=lab)
        ax.fill_between(t, r[k] - r[k + '_SE'], r[k] + r[k + '_SE'], color=c, alpha=0.3, lw=0)
    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(ylabel)


def _plot(out, r, stages, ref=None, tmax=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = r['time']
    if tmax is not None:
        msk = t <= tmax
        t = t[msk]
        r = {k: (v[msk] if isinstance(v, np.ndarray) and v.shape == msk.shape else v) for k, v in r.items()}
    bounds = r['t_starts'][1:]

    colors = ('C0', 'C2', 'C1', 'C3')
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.0))
    _panel(axs[0, 0], t, r, ('Rg', 'Tg', 'Re', 'Te'),
           (r'$R_g=\int h(-q)\,\langle g|\hat\rho|g\rangle$', r'$T_g=\int h(q)\,\langle g|\hat\rho|g\rangle$',
            r'$R_e=\int h(-q)\,\langle e|\hat\rho|e\rangle$', r'$T_e=\int h(q)\,\langle e|\hat\rho|e\rangle$'),
           colors, 'population', bounds)
    if ref is not None:
        tr = ref['time'] if tmax is None else ref['time'][ref['time'] <= tmax]
        for k, c in zip(('CRg', 'CTg', 'CRe', 'CTe'), colors):
            axs[0, 0].plot(tr, ref[k][:tr.size], color=c, lw=1.2, ls='--')
        axs[0, 0].plot([], [], color='0.3', lw=1.2, ls='--', label=r'analyze_tully $C$ (--ref)')
    axs[0, 0].legend(loc='center left', frameon=False, fontsize=9)

    _panel(axs[0, 1], t, r, ('Tr', 'Pg', 'Pe'),
           (r'$\mathrm{Tr}\,\hat\rho$', r'$P_g=\int\langle g|\hat\rho|g\rangle$', r'$P_e=\int\langle e|\hat\rho|e\rangle$'),
           ('k', 'C0', 'C3'), 'population', bounds)
    axs[0, 1].legend(loc='best', frameon=False, fontsize=9)

    _panel(axs[1, 0], t, r, ('X', 'Y'), (r'$\int\rho_x$', r'$\int\rho_y$'), ('C4', 'C5'),
           'coherence', bounds)
    axs[1, 0].legend(loc='best', frameon=False, fontsize=9)

    ax = axs[1, 1]
    rs_st = [s for s in stages if 'resample_raw_l1' in s]
    if rs_st:
        x = np.array([int(s['stage']) for s in rs_st]) + 0.5
        for k, (c, col) in enumerate(zip(COMPONENTS, ('k', 'C0', 'C4', 'C5'))):
            ax.plot(x, [s['resample_raw_l1'][k] for s in rs_st], color=col, marker='o', lw=1.4,
                    label=rf'$\rho_{c}$: $\sum|W|$ raw')
            ax.plot(x, [s['resample_binned_l1'][k] for s in rs_st], color=col, marker='s', lw=1.4, ls='--',
                    label=r'$\sum|B|$ binned')
            ax.plot(x, [abs(s['resample_net'][k]) for s in rs_st], color=col, marker='^', lw=1.0, ls=':',
                    label=r'$|\sum W|$ net')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v - 0.5)}' + r'$\to$' + f'{int(v + 0.5)}' for v in x])
        ax.legend(loc='best', frameon=False, fontsize=7, ncol=4)
    ax.set_xlabel('resampling (stage)')
    ax.set_ylabel('total |weight|')

    s0 = stages[0]
    fig.suptitle(f"{len(stages)} stage(s), {int(s0['n_points'])} points x {int(s0['n_spin'])} spins per stage"
                 + ('' if ref is None else f"    --ref: {ref['jump_label']}, {ref['n_traj']} traj"), fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_points(out, stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(stages)
    fig, axs = plt.subplots(4, n, figsize=(3.2 * n + 1.0, 10.0), squeeze=False, sharex='col', sharey='col')
    for m, s in enumerate(stages):
        q, p, w = s['init_q'], s['init_p'], s['init_w']
        for k, c in enumerate(COMPONENTS):
            ax = axs[k, m]
            nz = w[:, k] != 0.0
            if nz.any():
                vmax = np.abs(w[nz, k]).max()
                sc = ax.scatter(q[nz], p[nz], c=w[nz, k], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                s=12 + 40 * np.abs(w[nz, k]) / vmax, edgecolors='none')
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(rf"stage {int(s['stage'])}, $w_{c}$  (t={float(s['t_start']):g})", fontsize=9)
            if k == 3:
                ax.set_xlabel('q')
            if m == 0:
                ax.set_ylabel('p')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(description='Stitched observables of a MASH-RPMD_Tully-Testing run.')
    ap.add_argument('--config', default='config.py', help='the run\'s base config (stages sit next to it)')
    ap.add_argument('--ref', default=None, help='MASH-RPMD_Tully-Delta shards to overlay (analyze_tully C)')
    ap.add_argument('--ref-ncores', type=int, default=None, help='workers for the --ref analysis')
    ap.add_argument('--tmax', type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--out', default='analyze_testing.png')
    ap.add_argument('--out-points', default='analyze_testing_points.png')
    a = ap.parse_args()

    run_dir = os.path.dirname(os.path.abspath(a.config))
    stages = load_stages(run_dir)
    r = stitch(stages)
    ref = None
    if a.ref:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'MASH-RPMD_Tully-Delta'))
        import analyze_tully
        ref = analyze_tully.compute(a.ref, ncores=a.ref_ncores)

    _plot(a.out, r, stages, ref=ref, tmax=a.tmax)
    _plot_points(a.out_points, stages)

    for s in stages:
        names = list(s['names'])
        fin = '  '.join(f"{k}={s['tot'][names.index(k), -1]:.4f}+-{s['se'][names.index(k), -1]:.4f}"
                        for k in ('Rg', 'Tg', 'Re', 'Te', 'Tr', 'X', 'Y'))
        print(f"[test] stage {int(s['stage'])} (t {float(s['t_start']):g} -> {s['time'][-1]:g}): {fin}")
        if 'resample_raw_l1' in s:
            print(f"[test]   -> stage {int(s['stage']) + 1}: sum|W| raw " + ' '.join(f'{v:.3e}' for v in s['resample_raw_l1'])
                  + '  binned ' + ' '.join(f'{v:.3e}' for v in s['resample_binned_l1'])
                  + '  net ' + ' '.join(f'{v:+.3e}' for v in s['resample_net'])
                  + f"   ({int(s['resample_n_distinct'])} distinct points)")
    if ref is not None:
        zs = []
        for k in ('Rg', 'Tg', 'Re', 'Te'):
            kr = np.searchsorted(ref['time'], r['time'])
            ok = (kr < ref['time'].size) & np.isclose(ref['time'][np.minimum(kr, ref['time'].size - 1)], r['time'])
            d = r[k][ok] - ref['C' + k][kr[ok]]
            s = np.hypot(r[k + '_SE'][ok], ref['C' + k + '_SE'][kr[ok]])
            zs.append(np.where(s > 0, np.abs(d) / np.where(s > 0, s, 1.0), 0.0).max())
        print(f"[test] vs --ref ({ref['jump_label']}, {ref['n_traj']} traj): max |difference|/SE over the "
              f"common print times  Rg {zs[0]:.1f}  Tg {zs[1]:.1f}  Re {zs[2]:.1f}  Te {zs[3]:.1f}")
    print(f'[test] wrote {a.out} and {a.out_points}')


if __name__ == '__main__':
    _main()
