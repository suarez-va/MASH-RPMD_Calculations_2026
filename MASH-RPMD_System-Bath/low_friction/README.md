# Multi-trajectory RP_MASH run

A worked example of running many independent RP_MASH trajectories and averaging them, using the
`workflow` package that ships inside RP_MASH. You only maintain two files here:

| file        | what it holds                                                              |
|-------------|----------------------------------------------------------------------------|
| `config.py` | one `CONFIG` dict: orchestration knobs **and** model parameters            |
| `system.py` | `run_trajectory(cfg, seed)` — builds one `mash_rpmd` object and runs it     |

Everything else (parallel scheduling, per-trajectory seeds, per-trajectory output dirs, averaging
with error bars, SLURM submission) is handled by `workflow`.

## Setup (once per shell)

`workflow` and `mash_rpmd` both live under the RP_MASH root, so put it on `PYTHONPATH`:

```bash
export PYTHONPATH=/home/victorwsl/repos/RP_MASH:$PYTHONPATH
cd /home/victorwsl/repos/mashrpmd_calcs/RPMD_GLE/nuconly/multi_traj
```

## Run locally

```bash
python -m workflow.runner  --config config.py            # fan n_traj out across ncores cores
```

Each trajectory `k` runs in `traj_grid/trajk/` with its own `output.dat`, `mapS[xyz].dat`, etc.
Seeds derive from `(base_seed, k)`, so the run is reproducible and every trajectory is independent.
If some trajectories crash, the runner prints which indices `FAILED`; redo just those:

```bash
python -m workflow.runner  --config config.py --rerun 3 17
```

## Average (quick, per-file)

```bash
python -m workflow.average --config config.py --file mapSz.dat      # -> mapSz.avg (means | SEs)
python -m workflow.average --config config.py --file output.dat --cols 1 2 3
```

In a script you can instead call the API directly:

```python
from workflow import average_column, mash_population
mean, sem = average_column('config.py', 'mapSz.dat')   # each (nrows, ncols)
spins     = mash_population('config.py')                # dict: time, Sx/Sy/Sz (+ *_sem)
```

## Consolidate to one HDF5 file (for custom analysis)

For anything beyond a column average -- normal-mode transforms, correlation functions, whatever --
gather every trajectory into a single HDF5 file once, then write plain analysis scripts against it:

```bash
python -m workflow.consolidate --config config.py       # -> data.hdf (gzip-compressed)
```

`data.hdf` holds `nucR (n_traj, T, nbds, nnuc)`, `nucP`, `mapS[xyz] (n_traj, T, nbds)`,
`output`, a shared `time` grid, and the full config in attrs. Datasets are chunked one trajectory
per chunk, so slicing reads only what you ask for -- the whole run never has to sit in RAM:

```python
from workflow import open_run
with open_run('data.hdf') as f:
    time = f['time'][:]
    R7   = f['nucR'][7]            # lazy: only trajectory 7  (T, nbds, nnuc)
    sz   = f['mapSz'][:, :, 0]     # lazy: bead-0 spin vs time, all trajectories
    print(dict(f.attrs))           # nbds, nnuc, beta, delt, gamma, config_json, ...
```

See `analyze_nm_acf.py` in this directory for a copyable template: it opens `data.hdf`, transforms
bead coordinates to normal modes, and builds a per-mode correlation function averaged over
trajectories with a standard-error band. Copy it and swap in whatever correlator your plot needs.

## Run on SLURM

Same worker, launched as a job array instead of local subprocesses:

```bash
python -m workflow.slurm   --config config.py           # writes submit_grid.sh
sbatch submit_grid.sh
python -m workflow.average --config config.py --file mapSz.dat   # once the array finishes
```

Trajectory `k` gets the same seed whether it ran locally or on the cluster, so results match.

## Scanning a parameter

Copy this directory, change the numbers in `config.py` (e.g. a different `gamma`), and re-run. Give
each scan point its own directory (or its own `grid_dir`) so their `traj_grid/` outputs don't mix.
