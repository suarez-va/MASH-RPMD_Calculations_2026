"""
consolidate_therm.py -- merge the thermalization shards into ONE file indexed by trajectory.

therm.py appends each trajectory's restart state to a per-SLURM-task shard (shard_*.hdf), so the
thermalize stage writes ~495 files instead of n_traj.  This script merges them into

    <therm_dir>/therm_data.hdf

which traj.py reads row idx of.  Layout written (N = n_traj):

    /nucR    (N, nbds, nnuc)          /nucP  (N, nbds, nnuc)
    /memP    (N, nbds, nnuc, Nmem)    /Ffluci (N, nbds, 2*Nfluc)
    /seed    (N,)                     /present (N,) bool  -- False for a trajectory no shard holds
    root attrs: the shards' run-constant metadata (nbds, delt, Tmem, Nburn, ...) + n_traj, n_present

Datasets are chunked one trajectory per chunk and uncompressed (float64 noise does not compress), so
f['memP'][k] is a single read.  Shards are streamed in slabs of --slab rows, so peak memory is about
one slab of one shard; raise --slab on a big-memory node for fewer, larger reads.

Only a shard's COMMITTED rows (the first len(idx)) are read -- a row past that is a write cut off by
a kill.  A trajectory found in more than one shard is taken from the first and counted as a
duplicate.  Shards are left in place; delete them once the merged file checks out.  The output is
written to <out>.tmp and renamed, so an interrupted merge never leaves a half file under <out>.

CLI:
    python consolidate_therm.py --dir <therm_dir>                      # -> <therm_dir>/therm_data.hdf
    python consolidate_therm.py --dir <therm_dir> --n-traj 10000000 --slab 20000
"""

import os
import glob
import argparse

import numpy as np
import h5py


_ROWS = ('nucR', 'nucP', 'memP', 'Ffluci', 'seed')


def _attrs_equal(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _scan(paths):
    """Read each shard's committed idx and check the shards agree. Returns (attrs, row_shapes, shards)."""
    attrs0, shapes0, shards = None, None, []
    for p in paths:
        with h5py.File(p, 'r') as f:
            if 'idx' not in f or f['idx'].shape[0] == 0:
                print(f'[consolidate_therm] WARNING: {p} has no committed rows, skipped')
                continue
            idx    = f['idx'][:]
            attrs  = dict(f.attrs)
            shapes = {name: f[name].shape[1:] for name in _ROWS}
        if attrs0 is None:
            attrs0, shapes0 = attrs, shapes
        elif not _attrs_equal(attrs, attrs0) or shapes != shapes0:
            raise ValueError(f'{p} was made with different run metadata than {shards[0][0]}; '
                             f'the shards do not belong to the same thermalization run.')
        shards.append((p, idx))
    if not shards:
        raise FileNotFoundError('no shard has any committed rows')
    return attrs0, shapes0, shards


def _runs(tgt):
    """Split sorted target indices into maximal consecutive runs -> [(start, stop) positions]."""
    breaks = np.nonzero(np.diff(tgt) != 1)[0] + 1
    return zip(np.r_[0, breaks], np.r_[breaks, tgt.size])


def consolidate_therm(therm_dir, out=None, n_traj=None, slab=1000):
    therm_dir = os.path.abspath(therm_dir)
    paths = sorted(glob.glob(os.path.join(therm_dir, 'shard_*.hdf')))
    if not paths:
        raise FileNotFoundError(f'no shard_*.hdf in {therm_dir}')
    out = out or os.path.join(therm_dir, 'therm_data.hdf')

    attrs, shapes, shards = _scan(paths)
    N = int(n_traj) if n_traj is not None else int(max(idx.max() for _, idx in shards)) + 1
    print(f'[consolidate_therm] {len(shards)} shards, '
          f'{sum(idx.size for _, idx in shards)} committed rows -> N = {N}', flush=True)

    present = np.zeros(N, dtype=bool)
    n_dup = n_out_of_range = 0
    tmp = out + '.tmp'
    with h5py.File(tmp, 'w') as fo:
        fo.attrs.update(attrs)
        dsets = {}
        for name in _ROWS:
            sh = shapes[name]
            dsets[name] = fo.create_dataset(name, shape=(N,) + sh, dtype='f8' if sh else 'i8',
                                            chunks=(1,) + sh if sh else (min(N, 4096),))

        for k, (p, idx) in enumerate(shards):
            # first occurrence only: within this shard, and not already taken from an earlier shard
            in_range = idx < N
            n_out_of_range += int((~in_range).sum())
            keep = np.zeros(idx.size, dtype=bool)
            _, first = np.unique(idx, return_index=True)
            keep[first] = True
            keep &= in_range
            keep[keep] &= ~present[idx[keep]]
            n_dup += int(in_range.sum() - keep.sum())

            with h5py.File(p, 'r') as fi:
                for lo in range(0, idx.size, slab):
                    hi  = min(lo + slab, idx.size)
                    sel = np.nonzero(keep[lo:hi])[0]
                    if sel.size == 0:
                        continue
                    tgt   = idx[lo:hi][sel]
                    order = np.argsort(tgt, kind='stable')
                    sel, tgt = sel[order], tgt[order]
                    for name in _ROWS:
                        block = fi[name][lo:hi]
                        for s, e in _runs(tgt):
                            dsets[name][tgt[s]:tgt[s] + (e - s)] = block[sel[s:e]]
                    present[tgt] = True
            print(f'[consolidate_therm] shard {k + 1}/{len(shards)}  {os.path.basename(p)}  '
                  f'({int(present.sum())}/{N} present)', flush=True)

        fo.create_dataset('present', data=present)
        fo.attrs['n_traj']    = N
        fo.attrs['n_present'] = int(present.sum())
    os.replace(tmp, out)

    missing = np.nonzero(~present)[0]
    if missing.size:
        print(f'[consolidate_therm] WARNING: {missing.size}/{N} trajectories missing: '
              f'{missing[:20].tolist()}{" ..." if missing.size > 20 else ""}')
    if n_dup:
        print(f'[consolidate_therm] WARNING: {n_dup} duplicate rows ignored (first occurrence kept)')
    if n_out_of_range:
        print(f'[consolidate_therm] WARNING: {n_out_of_range} rows with idx >= {N} ignored')
    print(f'[consolidate_therm] {out}  ({int(present.sum())}/{N} trajectories)')
    return out


def _main():
    ap = argparse.ArgumentParser(description='Merge therm.py shards into one therm_data.hdf.')
    ap.add_argument('--dir', required=True, help='directory holding the shard_*.hdf files')
    ap.add_argument('--out', default=None, help='output file (default: <dir>/therm_data.hdf)')
    ap.add_argument('--n-traj', type=int, default=None,
                    help='number of trajectories N (default: largest idx found + 1). Set it to the '
                         'config n_traj so trailing missing trajectories are reported too.')
    ap.add_argument('--slab', type=int, default=1000,
                    help='shard rows read per batch (default 1000); larger = fewer reads, more RAM')
    a = ap.parse_args()
    consolidate_therm(a.dir, out=a.out, n_traj=a.n_traj, slab=a.slab)


if __name__ == '__main__':
    _main()
