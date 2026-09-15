"""
snip_data.py -- truncate a consolidated data.hdf so that it stops at an earlier final time.

Production runs for Nprod = round(cfg['traj_time'] / delt) steps -- the run length is used
DIRECTLY (the thermalization time is not subtracted). Runs configured for 40 a.u. therefore
produced 40 a.u. of production when only 15 a.u. was wanted. Rather than regenerate the raw
trajectories, this script cuts an already-consolidated data.hdf down along the time axis.
(Older files carry this as `total_time`; both spellings are handled.)

The result is meant to be INDISTINGUISHABLE from a run that genuinely stopped at the shorter time:

  * every observable is truncated along its time axis (axis 1) to the kept rows,
  * /time is truncated to the same rows,
  * chunks become (1,) + <new per-trajectory shape>, exactly what workflow.consolidate would have
    written for a run of that length,
  * the source's compression/shuffle/fletcher32 filters and dtype are mirrored,
  * config_json's run-length key (traj_time, or total_time in older files) is set to the new
    final time,
  * every other attribute -- root and per-dataset -- is copied verbatim, and NO provenance
    attributes are added (they would make the file identifiable as snipped).

/traj_index and n_traj are untouched: they describe which trajectories are present, not how long
they ran.

The source file is never modified; the snipped copy is written to a new path.

Layout handled (see workflow/consolidate.py):
    /time                     (T,)
    /traj_index               (n,)
    /nucR, /nucP              (n, T, nbds, nnuc)
    /mapSx, /mapSy, /mapSz    (n, T, nbds)
    /output                   (n, T, ncol)   + attrs['column_labels']
    root attrs: config_json, nbds, nnuc, nstates, beta, delt, gamma, n_traj

Usage:
    python snip_data.py data.hdf                       # -> data_snipped.hdf, stops at t=15.000
    python snip_data.py data.hdf --tmax 15.0 --out short.hdf
"""

import os
import json
import argparse

import numpy as np
import h5py


def _new_config_json(fin, t_end, n_keep):
    """Copy config_json with the run-length key set to the new final time.

    Returns (json_str, key, old_value, cfg) -- `key` is whichever run-length key the file uses.
    """
    cfg = json.loads(fin.attrs['config_json'])
    # The run-length key was renamed total_time -> traj_time when the configs were split into a
    # hierarchy; data.hdf files written before that still carry total_time. Update whichever key
    # this file actually has, and do not introduce the other one.
    key = 'traj_time' if 'traj_time' in cfg else 'total_time'
    old_total = cfg.get(key)
    cfg[key] = t_end

    # A genuine run of this length would satisfy round(traj_time/delt)//Nprint + 1 == rows.
    delt   = cfg.get('delt')
    nprint = cfg.get('Nprint')
    if delt and nprint:
        predicted = round(t_end / float(delt)) // int(nprint) + 1
        if predicted != n_keep:
            print(f'[snip] WARNING: a real run with {key}={t_end:g}, delt={delt:g}, '
                  f'Nprint={nprint} would have {predicted} rows, but the cut keeps {n_keep}. '
                  f'tmax is not a clean stopping point for this grid -- the output will not look '
                  f'like a genuine run.')
    return json.dumps(cfg, default=str), key, old_total, cfg


def snip(path, tmax=15.0, out=None, force=False, batch=256):
    """Write a copy of `path` truncated to t <= tmax. Returns the output path."""
    src = os.path.abspath(path)
    if out is None:
        stem, ext = os.path.splitext(src)
        out = stem + '_snipped' + ext
    out = os.path.abspath(out)

    if out == src:
        raise ValueError('output path is the same as the input; the source is never modified')
    if os.path.exists(out) and not force:
        raise FileExistsError(f'{out} exists (pass --force to overwrite)')

    with h5py.File(src, 'r') as fin:
        time = fin['time'][:]
        T    = time.shape[0]

        n_keep = int(np.searchsorted(time, tmax + 1e-9, side='right'))
        if n_keep < 1:
            raise ValueError(f'tmax={tmax:g} is before the first time point ({time[0]:g}); '
                             f'nothing would be kept')
        if n_keep >= T:
            raise ValueError(f'tmax={tmax:g} is at or beyond the last time point '
                             f'({time[-1]:g}); there is nothing to cut')
        t_end = float(time[n_keep - 1])

        cfg_json, key, old_total, cfg = _new_config_json(fin, t_end, n_keep)

        print(f'[snip] {src}')
        print(f'[snip] keeping {n_keep} of {T} rows  (t = {time[0]:g} .. {t_end:g}, '
              f'dropping {T - n_keep} rows out to t = {time[-1]:g})')
        print(f'[snip] {key} {old_total} -> {t_end:g}')

        with h5py.File(out, 'w') as fout:
            # ---- root attrs verbatim, in source order, config_json swapped ----
            for k, v in fin.attrs.items():
                fout.attrs[k] = cfg_json if k == 'config_json' else v

            fout.create_dataset('time', data=time[:n_keep])
            if 'traj_index' in fin:
                fout.create_dataset('traj_index', data=fin['traj_index'][:])

            # ---- every other dataset: truncate axis 1, stream in trajectory batches ----
            for name in fin:
                if name in ('time', 'traj_index'):
                    continue
                d = fin[name]
                if d.ndim < 2 or d.shape[1] != T:
                    raise ValueError(f'dataset "{name}" has shape {d.shape}; expected its time '
                                     f'axis (axis 1) to be {T}')
                n     = d.shape[0]
                shape = (n, n_keep) + d.shape[2:]
                dst = fout.create_dataset(
                    name, shape=shape, dtype=d.dtype, chunks=(1,) + shape[1:],
                    compression=d.compression, compression_opts=d.compression_opts,
                    shuffle=d.shuffle, fletcher32=d.fletcher32)
                for k, v in d.attrs.items():          # carries output's column_labels
                    dst.attrs[k] = v

                step = max(1, int(batch))
                for i in range(0, n, step):
                    j = min(i + step, n)
                    dst[i:j] = d[i:j, :n_keep]
                    print(f'[snip]   {name}: {j}/{n} trajectories', flush=True)

    print(f'[snip] wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB; '
          f'source left untouched at {os.path.getsize(src) / 1e6:.1f} MB)')
    return out


def _main():
    ap = argparse.ArgumentParser(
        description='Truncate a consolidated data.hdf so it stops at an earlier final time.')
    ap.add_argument('file', help='the data.hdf to snip (never modified)')
    ap.add_argument('--tmax',  type=float, default=15.0,
                    help='new final time, inclusive (default 15.0)')
    ap.add_argument('--out',   default=None,
                    help='output path (default: <stem>_snipped.hdf beside the source)')
    ap.add_argument('--force', action='store_true', help='overwrite an existing output file')
    ap.add_argument('--batch', type=int, default=256,
                    help='trajectories per read/write block (default 256)')
    a = ap.parse_args()
    snip(a.file, tmax=a.tmax, out=a.out, force=a.force, batch=a.batch)


if __name__ == '__main__':
    _main()
