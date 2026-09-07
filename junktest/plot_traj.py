"""Plot RP-MASH trajectory diagnostics: total energy, ring-polymer bead
positions, and spin-mapping variables, all vs time.

Usage:  python plot_traj.py [run_directory]

Reads output.dat (t, E_tot, KE, PE, <sign Sz>, R_com...), nucR.dat, and
mapSx/y/z.dat as written by mash_rpmd.print_data.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless: no GUI window, never blocks
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "."
nuc_dof = 0                      # which nuclear DOF to plot if nnuc > 1
save_to = f"{d}/trajectory.png"

# --- load -------------------------------------------------------------------
names = ("output", "nucR", "mapSx", "mapSy", "mapSz")
raw = {f: np.loadtxt(f"{d}/{f}.dat") for f in names}
rows = {f: a.shape[0] for f, a in raw.items()}
n = min(rows.values())

# A run still in progress leaves the files a row or two out of step with each
# other; trim to the common length and say so rather than plotting a mismatch.
if len(set(rows.values())) > 1:
    print("WARNING: .dat files disagree on row count - is a run still writing?")
    for f in names:
        print(f"    {f}.dat: {rows[f]} rows")
    print(f"    -> trimming all series to {n} rows")
raw = {f: a[:n] for f, a in raw.items()}

out = raw["output"]
t, etot, ke, pe, sgn = out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]
com = out[:, 5:]                                  # (nsteps, nnuc)
nnuc = com.shape[1]

nucR = raw["nucR"][:, 1:]                         # bead-major: b1n1 b1n2 ... b2n1 ...
nbds = nucR.shape[1] // nnuc
nucR = nucR.reshape(n, nbds, nnuc)

Sx = raw["mapSx"][:, 1:]                          # (nsteps, nbds)
Sy = raw["mapSy"][:, 1:]
Sz = raw["mapSz"][:, 1:]

# --- style ------------------------------------------------------------------
ink, muted = "#22201d", "#6b6660"
bead_cmap = plt.cm.Blues(np.linspace(0.35, 0.95, nbds))   # sequential: bead index is ordinal
spin_c = {"Sx": "#0072B2", "Sy": "#D55E00", "Sz": "#009E73"}  # categorical, CVD-validated

fig, ax = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True,
                       gridspec_kw={"hspace": 0.16})

# --- panel 1: total energy --------------------------------------------------
# plotted as a deviation from E(0) so conservation is readable without an axis offset
ax[0].plot(t, etot - etot[0], lw=2.0, color=ink)
ax[0].axhline(0.0, lw=0.8, color=muted, alpha=0.5, zorder=0)
drift = (etot[-1] - etot[0]) / abs(etot[0])
ax[0].set_ylabel("$E_{tot} - E_{tot}(0)$")
ax[0].set_title(f"Total energy   ($E_0$ = {etot[0]:.6f},  drift $\\Delta E/E_0$ = {drift:+.1e},"
                f"  spread = {np.ptp(etot):.1e})", loc="left", fontsize=11, color=ink)

# --- panel 2: nuclear ring-polymer positions --------------------------------
for b in range(nbds):
    ax[1].plot(t, nucR[:, b, nuc_dof], lw=1.2, color=bead_cmap[b])
ax[1].plot(t, com[:, nuc_dof], lw=2.2, color=ink, label="centroid")
ax[1].plot([], [], lw=1.2, color=bead_cmap[nbds // 2],
           label=f"beads 1$\\to${nbds} (light$\\to$dark)")
ax[1].set_ylabel("position  $R$")
ax[1].set_title(f"Ring-polymer bead positions (DOF {nuc_dof})",
                loc="left", fontsize=11, color=ink)
ax[1].margins(y=0.22)                       # headroom so the legend clears the data
ax[1].legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

# --- panel 3: spin-mapping variables ----------------------------------------
for name, S in (("Sx", Sx), ("Sy", Sy), ("Sz", Sz)):
    ax[2].plot(t, S, lw=0.9, color=spin_c[name], alpha=0.30)      # per bead
    ax[2].plot(t, S.mean(axis=1), lw=2.0, color=spin_c[name], label=f"$\\langle {name[0]}_{name[1]}\\rangle$")
ax[2].axhline(0.0, lw=0.8, color=muted, alpha=0.5, zorder=0)
ax[2].set_ylabel("spin-mapping variable  $S$")
ax[2].set_xlabel("time")
ax[2].set_title("Mapping spin variables (faint = per bead, bold = bead average)",
                loc="left", fontsize=11, color=ink)
ax[2].margins(y=0.28)
ax[2].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")

for a in ax:
    a.grid(True, lw=0.5, color=muted, alpha=0.18)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(muted)
    a.tick_params(colors=muted, labelsize=9)
    a.yaxis.label.set_color(ink)
    a.xaxis.label.set_color(ink)

# provenance stamp: makes it obvious from the image alone whether it is current
data_time = time.strftime("%Y-%m-%d %H:%M:%S",
                          time.localtime(os.path.getmtime(f"{d}/output.dat")))
stamp = (f"{os.path.abspath(d)}   |   data written {data_time}   |   {n} rows   |   "
         f"t = {t[0]:g} to {t[-1]:g}   |   plotted {time.strftime('%Y-%m-%d %H:%M:%S')}")
fig.text(0.5, 0.004, stamp, ha="center", fontsize=7, color=muted)

fig.savefig(save_to, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"{n} rows, {nbds} beads, {nnuc} nuclear DOF, t = {t[0]:g} to {t[-1]:g} -> {save_to}")
print(f"    data written {data_time}")
