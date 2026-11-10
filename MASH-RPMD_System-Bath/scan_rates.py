"""
scan_rates.py -- quick-and-dirty analysis of a whole scan: for every scan point (delta_1e*/ or
epsil_*/) of one or more method dirs, run analyze_rate.py (MASH) or analyze_bo_rate.py (BO) in-process,
write the point's PNG exactly as those scripts do, and collect the block-analysis rate m_bar +/- sigma_m.

Per method dir it prints a table and copy-pasteable arrays, and saves both to <method_dir>/scan_rates.txt:

    log10_delta = [-1.1, -1.0, ...]
    rate        = [1.234e-05, ...]
    rate_err    = [2.1e-07, ...]

BO vs MASH is taken from the method dir's config_method.py (dynamics == 'bo').  The per-unit partials
are cached as usual (<shards>_partials/), so a re-run only does the new shards.

Usage (PYTHONPATH=<RP_MASH root>):
    python scan_rates.py Low-Friction/Delta-Scan/*/ --ncores 16
    cd Low-Friction/Delta-Scan/MASH-Classical && python ../../../scan_rates.py --ncores 16
"""

import os
import re
import sys
import glob
import argparse
import traceback
import importlib.util

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import analyze_rate
import analyze_bo_rate


_POINT_RE = re.compile(r'^(delta_1e|epsil_)(.+)$')
_KEYNAME = {'delta_1e': 'log10_delta', 'epsil_': 'epsil'}


def method_dynamics(method_dir):
    """'bo' or 'mash', from <method_dir>/config_method.py (default 'mash')."""
    p = os.path.join(method_dir, 'config_method.py')
    if not os.path.isfile(p):
        return 'mash'
    try:
        spec = importlib.util.spec_from_file_location('_scan_cfg_method', p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(mod.CONFIG.get('dynamics', 'mash')).lower()
    except Exception as e:
        print(f'[scan] WARNING: could not load {p} ({e}); assuming MASH')
        return 'mash'


def scan_points(method_dir):
    """[(value, key name, point dir)] sorted by value."""
    pts = []
    for d in glob.glob(os.path.join(method_dir, '*/')):
        name = os.path.basename(os.path.normpath(d))
        m = _POINT_RE.match(name)
        if not m:
            continue
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        pts.append((v, _KEYNAME[m.group(1)], os.path.normpath(d)))
    pts.sort()
    keys = {k for _, k, _ in pts}
    if len(keys) > 1:
        print(f'[scan] WARNING: {method_dir} mixes {sorted(keys)} scan points')
    return pts


def analyze_point(point_dir, is_bo, a):
    """Run the right analysis on one point; return the result dict (PNG written into point_dir)."""
    path = os.path.join(point_dir, 'shards')
    if not os.path.isdir(path):
        path = os.path.join(point_dir, 'data.hdf')
        if not os.path.isfile(path):
            raise RuntimeError(f'no shards/ or data.hdf in {point_dir}')
    common = dict(nblocks=a.nblocks, fitmin=a.fitmin, fitmax=a.fitmax, ncores=a.ncores,
                  refresh=a.refresh)
    if is_bo:
        r = analyze_bo_rate.compute(path, **common)
        analyze_bo_rate._plot(os.path.join(point_dir, 'analyze_bo_rate.png'), r, tmax=a.tmax)
    else:
        r = analyze_rate.compute(path, w0_only=a.w0_only, **common)
        out = 'analyze_rate_nojump.png' if a.w0_only else 'analyze_rate.png'
        analyze_rate._plot(os.path.join(point_dir, out), r, tmax=a.tmax)
    return r


def _fmt_list(xs, fmt):
    return '[' + ', '.join('nan' if not np.isfinite(x) else format(x, fmt) for x in xs) + ']'


def run_method(method_dir, a):
    method_dir = os.path.normpath(method_dir)
    label = os.path.basename(os.path.abspath(method_dir))
    is_bo = method_dynamics(method_dir) == 'bo'
    pts = scan_points(method_dir)
    if not pts:
        print(f'[scan] {method_dir}: no delta_1e*/ or epsil_*/ scan points, skipping')
        return
    print(f'\n[scan] ===== {label} ({"BO" if is_bo else "MASH"}): {len(pts)} point(s) =====', flush=True)

    rows = []
    for v, key, d in pts:
        print(f'[scan] --- {os.path.basename(d)}', flush=True)
        try:
            r = analyze_point(d, is_bo, a)
            warn = []
            if r.get('n_nonphys'):
                warn.append(f'nonphys={r["n_nonphys"]}')
            if r.get('n_skipped'):
                warn.append(f'skipped={r["n_skipped"]}')
            rows.append((v, int(r['n_traj']), float(r['m_bar']), float(r['sigma_m']),
                         float(r['m_full']), ','.join(warn)))
            print(f'[scan]     m = {r["m_bar"]:.6g} +/- {r["sigma_m"]:.6g}  (n_traj={r["n_traj"]})'
                  + (f'  WARN {",".join(warn)}' if warn else ''), flush=True)
        except Exception as e:
            print(f'[scan]     FAILED: {e}', flush=True)
            if a.verbose:
                traceback.print_exc()
            rows.append((v, 0, np.nan, np.nan, np.nan, 'FAILED'))

    key = pts[0][1]
    lines = [f'# {label}  ({"BO" if is_bo else "MASH"}{", w0-only" if a.w0_only and not is_bo else ""}, '
             f'nblocks={a.nblocks}, fit [{a.fitmin:g},{a.fitmax:g}])',
             f'# {key:>11s} {"n_traj":>9s} {"m_bar":>14s} {"sigma_m":>14s} {"m_full":>14s}  warn']
    for v, n, m, s, mf, w in rows:
        lines.append(f'{v:13g} {n:9d} {m:14.6e} {s:14.6e} {mf:14.6e}  {w}')
    lines += ['',
              f'{key} = ' + _fmt_list([r[0] for r in rows], 'g'),
              'rate = ' + _fmt_list([r[2] for r in rows], '.6e'),
              'rate_err = ' + _fmt_list([r[3] for r in rows], '.6e'),
              'rate_full = ' + _fmt_list([r[4] for r in rows], '.6e')]
    text = '\n'.join(lines) + '\n'
    print('\n' + text, flush=True)
    out = os.path.join(method_dir, a.out)
    with open(out, 'w') as f:
        f.write(text)
    print(f'[scan] wrote {out}')


def _main():
    ap = argparse.ArgumentParser(description='Run analyze_rate / analyze_bo_rate over every scan '
                                             'point of the given method dirs and collect the rates.')
    ap.add_argument('dirs', nargs='*', default=['.'], help='method dirs (default: .)')
    ap.add_argument('--ncores',  type=int, default=None, help='worker processes (default: SLURM CPUs, else 1)')
    ap.add_argument('--nblocks', type=int, default=100, help='trajectory blocks (default 100)')
    ap.add_argument('--fitmin',  type=float, default=5.0, help='fit window start (default 5)')
    ap.add_argument('--fitmax',  type=float, default=15.0, help='fit window end (default 15)')
    ap.add_argument('--tmax',    type=float, default=None, help='truncate the PLOTS to t <= tmax')
    ap.add_argument('--w0-only', action='store_true', help='MASH only: no-jump baseline (W^(0))')
    ap.add_argument('--refresh', action='store_true', help='recompute every cached partial')
    ap.add_argument('--out',     default='scan_rates.txt', help='summary file name in each method dir')
    ap.add_argument('-v', '--verbose', action='store_true', help='print tracebacks of failed points')
    a = ap.parse_args()
    if a.w0_only and a.out == 'scan_rates.txt':
        a.out = 'scan_rates_nojump.txt'
    for d in a.dirs:
        if os.path.basename(os.path.normpath(d)).startswith('__'):    # __pycache__ from a glob
            continue
        run_method(d, a)


if __name__ == '__main__':
    _main()
