"""
analyze_cpm.py -- the quantum-jump-weighted adiabat populations c_+(t) and c_-(t), in one streaming
pass over data.hdf.  Output is a single-panel figure (analyze_cpm.png).

This is the Tully-tree sibling of MASH-RPMD_System-Bath/analyze_p2.py.  It borrows that script's
machinery wholesale -- the jump-weight expansion, the contiguous-block error bars and the streaming
slab loop -- but the observable is different and far simpler.

--- the observable ---------------------------------------------------------------------------------

    c_+(t) = < W_PP^(m)  h( S_z(t) ) >
    c_-(t) = < W_PP^(m)  h(-S_z(t) ) >

with h the Heaviside step and <...> an average over trajectories.  In MASH, sgn(S_z) selects the
active adiabatic surface -- mash_rpmd.get_PE weights E_bo[:,1] by +1 and E_bo[:,0] by -1 -- so

    c_+  is the population of the UPPER adiabat,
    c_-  is the population of the LOWER adiabat.

traj.py calls init_map_spin(0), which draws cos(theta) in (-1,0), so every trajectory starts on the
lower adiabat and therefore c_+(0) = 0, c_-(0) = <W_PP>.

Only ONE of the four quantum-jump weights appears, and -- unlike analyze_p2.py -- no potential
parameter enters anywhere.  There is no P2eq, no K12 and no rate fit; those are specific to the
ET_with_bath model.

--- the weights ------------------------------------------------------------------------------------

W_PP^(m) is taken at m(t) = #{ jumps with t_jump <= t }, i.e. every jump up to AND INCLUDING t.
That matches mash_rpmd, which writes the post-jump state at t_jump, so the weight and the spin refer
to the same instant.  The weights live in data.hdf as a ragged per-jump table (/weights,
/n_weights) and are expanded onto the time grid by workflow.expand_weights.

A run consolidated WITHOUT the weights datasets is refused rather than silently analysed, since the
result would look plausible and be wrong.  --w0-only forces the no-jump baseline W^(0)_PP = 2|Sz(0)|
instead, which is the right comparison curve when you want to see what the jumps are doing.

--- bead reduction and an exact identity -----------------------------------------------------------

analyze_p2.py reduces the bead axis with np.sign(Sz).mean(axis=2).  The Heaviside analogue is
np.heaviside(Sz, 0.5).mean(axis=2), and the two conventions agree exactly because

    h(x) = (1 + sgn(x))/2        including at x = 0, where h = 0.5 and sgn = 0.

Since h(x) + h(-x) = 1 bead-by-bead, the bead means satisfy hp + hm = 1 identically, hence

    c_+(t) + c_-(t) = <W_PP^(m)>(t)      exactly, at every t.

That is a free end-to-end check on the weighting and the accumulators; it is verified numerically
and reported on the console.  For nbds == 1 (what this tree runs) every bead convention coincides.

--- block errors and memory ------------------------------------------------------------------------

The trajectory axis is split into `nblocks` contiguous blocks (np.array_split, the same split
analyze_rate.py and analyze_p2.py use), and the SE is the standard deviation of the block means.
Trajectories are read in slabs and only the running totals and per-block running sums are kept, so
peak memory is O(nblocks*T + slab*T) at any n_traj.  Block membership comes from the same contiguous
split, so the result is IDENTICAL to the whole-array form, not an approximation.

Usage (from the run dir, PYTHONPATH=<RP_MASH root>):
    python analyze_cpm.py
    python analyze_cpm.py --nblocks 20 --tmax 3000
    python analyze_cpm.py --w0-only --out analyze_cpm_nojump.png
"""

import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from workflow import open_run, expand_weights   # h5py.File(path,'r'); per-jump -> dense weights


def jump_label(cfg):
    """'jump_index=[...]' for current runs, 'Tjump=...' for old ones: titles and console lines."""
    if 'jump_index' in cfg:
        ji = list(cfg['jump_index'])
        if len(ji) > 6:
            return f'jump_index=[{ji[0]}, {ji[1]}, ..., {ji[-1]}] ({len(ji)} jumps)'
        return f'jump_index={ji}'
    return f"Tjump={cfg.get('Tjump')}"


def block_sem_from_means(block_means, n_blocks):
    """SE over trajectories from ALREADY-FORMED block means (same estimator as analyze_p2.py)."""
    if n_blocks < 2:
        return np.zeros(block_means.shape[1:])
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def compute(path, nblocks=None, slab=5000, w0_only=False):
    with open_run(path) as f:
        time   = f['time'][:]                    # (T,)
        cfg    = json.loads(f.attrs['config_json'])
        nbds   = int(f.attrs['nbds'])
        n_traj = f['mapSz'].shape[0]

        has_w = 'weights' in f
        if not has_w and not w0_only:
            raise KeyError(
                f"{path} has no /weights dataset. Either it is an old run made without Tjump, or it was "
                f"consolidated with a version of workflow.consolidate that predates the quantum-jump "
                f"weights. Re-run `python -m workflow.consolidate --config <config.py>` to pick them "
                f"up, or pass --w0-only to deliberately analyse the no-jump baseline W^(0).")

        # contiguous blocks, identical to np.array_split(..., axis=0) over the trajectory axis
        nb = n_traj if nblocks is None else max(1, min(int(nblocks), n_traj))
        block_of = np.concatenate([np.full(len(b), i, dtype=int)
                                   for i, b in enumerate(np.array_split(np.arange(n_traj), nb))])
        block_sizes = np.bincount(block_of, minlength=nb).astype(float)

        T = time.size
        tot_p = np.zeros(T);       tot_m = np.zeros(T)        # running sums over all trajectories
        blk_p = np.zeros((nb, T)); blk_m = np.zeros((nb, T))  # per-block running sums
        tot_w = np.zeros(T)                                   # <W_PP>, for the identity check

        n_jumps = 0

        # ---- streaming pass: one slab of trajectories at a time ----
        for lo in range(0, n_traj, slab):
            hi  = min(lo + slab, n_traj)
            sel = slice(lo, hi)

            Sz = f['mapSz'][sel]                 # (s, T, nbds) -- the only dataset this observable needs

            # h(Sz) and h(-Sz), bead-reduced. heaviside(.,0.5) at Sz==0 matches sgn(0)=0, so this is
            # exactly (1 +/- np.sign(Sz).mean(axis=2))/2 -- analyze_p2.py's convention.
            hp = np.heaviside( Sz, 0.5).mean(axis=2)          # (s,T) upper adiabat
            hm = np.heaviside(-Sz, 0.5).mean(axis=2)          # (s,T) lower adiabat

            if w0_only:
                # the no-jump baseline W^(0)_PP = 2|Sz(0)|, constant in t. abs of the BEAD MEAN, to
                # match mash_rpmd.init_jump_weights, which sets Sz0 = np.mean(self.mapSz).
                W_PP = 2.0 * np.abs(Sz[:, :1].mean(axis=2))   # (s,1) -> broadcasts over t
            else:
                W_PP = expand_weights(f, time=time, sel=sel)['W_PP']     # (s,T)
                n_jumps += int(f['n_weights'][sel].sum() - (hi - lo))    # row 0 is t=0, not a jump

            cp = W_PP * hp                       # (s,T) per-trajectory integrands
            cm = W_PP * hm

            tot_p += cp.sum(axis=0)
            tot_m += cm.sum(axis=0)
            tot_w += (W_PP * np.ones_like(hp)).sum(axis=0)   # ones_like: W_PP may be (s,1)
            np.add.at(blk_p, block_of[lo:hi], cp)
            np.add.at(blk_m, block_of[lo:hi], cm)

    cpt = tot_p / n_traj
    cmt = tot_m / n_traj
    W_mean = tot_w / n_traj

    cpt_SE = block_sem_from_means(blk_p / block_sizes[:, None], nb)
    cmt_SE = block_sem_from_means(blk_m / block_sizes[:, None], nb)

    # exact identity: h(x) + h(-x) = 1, so the two curves must sum to <W_PP> at every t
    identity_err = float(np.max(np.abs(cpt + cmt - W_mean)))

    return dict(time=time, cpt=cpt, cmt=cmt, cpt_SE=cpt_SE, cmt_SE=cmt_SE,
                W_mean=W_mean, identity_err=identity_err,
                n_traj=n_traj, nbds=nbds, nblocks=nb, n_jumps=n_jumps,
                w0_only=w0_only, has_weights=has_w,
                Tjump=cfg.get('Tjump'), jump_label=jump_label(cfg), full_jump=cfg.get('full_jump'))


def _plot(out, r, tmax=None):
    t = r['time']
    cpt, cpt_SE = r['cpt'], r['cpt_SE']
    cmt, cmt_SE = r['cmt'], r['cmt_SE']

    if tmax is not None:                         # truncate for plotting only
        msk = t <= tmax
        t = t[msk]
        cpt, cpt_SE = cpt[msk], cpt_SE[msk]
        cmt, cmt_SE = cmt[msk], cmt_SE[msk]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))

    ax.axhline(0.0, color='0.7', lw=0.8, ls='--')
    ax.plot(t, cpt, color='C3', lw=1.6,
            label=r'$c_+(t)=\langle W_{PP}^{(m)}\,h(S_z(t))\rangle$')
    ax.fill_between(t, cpt - cpt_SE, cpt + cpt_SE, color='C3', alpha=0.3, lw=0)
    ax.plot(t, cmt, color='C0', lw=1.6,
            label=r'$c_-(t)=\langle W_{PP}^{(m)}\,h(-S_z(t))\rangle$')
    ax.fill_between(t, cmt - cmt_SE, cmt + cmt_SE, color='C0', alpha=0.3, lw=0)

    ax.set_xlabel('time (a.u.)')
    ax.set_ylabel(r'$c_\pm(t)$')
    ax.legend(loc='best', frameon=False)

    ax.set_title(f"n_traj={r['n_traj']}, nbds={r['nbds']}, nblocks={r['nblocks']}"
                 + ("   [W^(0) ONLY -- no-jump baseline]" if r['w0_only'] else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _main():
    ap = argparse.ArgumentParser(
        description='Jump-weighted adiabat populations c_+(t) and c_-(t) from an RP-MASH data.hdf.')
    ap.add_argument('--file',    default='data.hdf')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOT to t <= tmax')
    ap.add_argument('--slab',    type=int, default=5000,
                    help='trajectories read per streaming pass (default 5000; lower it if RAM is tight)')
    ap.add_argument('--w0-only', action='store_true',
                    help='ignore the jump weights and use W^(0)_PP = 2|Sz0|: the no-jump baseline')
    ap.add_argument('--out',     default='analyze_cpm.png', help='output png')
    a = ap.parse_args()

    r = compute(a.file, nblocks=a.nblocks, slab=a.slab, w0_only=a.w0_only)
    _plot(a.out, r, tmax=a.tmax)

    print(f"[cpm] n_traj={r['n_traj']} nbds={r['nbds']} nblocks={r['nblocks']}"
          + ("  [W^(0) ONLY]" if r['w0_only'] else ""))
    if not r['w0_only']:
        print(f"[cpm] {r['jump_label']} full_jump={r['full_jump']}  "
              f"total jumps across all trajectories: {r['n_jumps']}")
        if r['n_jumps'] == 0:
            print(f"[cpm] NOTE: no jumps fired, so W_PP is frozen at its t=0 value 2|Sz(0)| and "
                  f"this result is identical to --w0-only.")
    print(f"[cpm] c_+(0)={r['cpt'][0]:.6e} +/- {r['cpt_SE'][0]:.3e}   "
          f"c_-(0)={r['cmt'][0]:.6e} +/- {r['cmt_SE'][0]:.3e}")
    print(f"[cpm] c_+(T)={r['cpt'][-1]:.6e} +/- {r['cpt_SE'][-1]:.3e}   "
          f"c_-(T)={r['cmt'][-1]:.6e} +/- {r['cmt_SE'][-1]:.3e}")
    print(f"[cpm] <W_PP>: [{r['W_mean'].min():.6f}, {r['W_mean'].max():.6f}]   "
          f"identity max|c_+ + c_- - <W_PP>| = {r['identity_err']:.3e}")
    if r['identity_err'] > 1e-10:
        print(f"[cpm] WARNING: the c_+ + c_- = <W_PP> identity is violated by "
              f"{r['identity_err']:.3e}; that should be round-off. Something is wrong.")
    print(f"[cpm] wrote {a.out}")


if __name__ == '__main__':
    _main()
