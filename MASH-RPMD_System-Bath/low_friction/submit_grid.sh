#!/bin/bash
#SBATCH --job-name=rpmash_grid
#SBATCH --array=0-449
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=/storage/project/r-jkretchmer3-0/vsuarez6/repos/MASH-RPMD_Calculations_2026/MASH-RPMD_System-Bath/low_friction/traj_grid/logs/task_%a.out
#SBATCH --error=/storage/project/r-jkretchmer3-0/vsuarez6/repos/MASH-RPMD_Calculations_2026/MASH-RPMD_System-Bath/low_friction/traj_grid/logs/task_%a.err
#SBATCH --account=gts-jkretchmer3-chemx
#SBATCH --mem=16GB
# one thread per task -> a task runs its chunk of trajectories sequentially on ONE core
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTHONPATH=/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/repos/RP_MASH:$PYTHONPATH
cd /storage/project/r-jkretchmer3-0/vsuarez6/repos/MASH-RPMD_Calculations_2026/MASH-RPMD_System-Bath/low_friction
eval "$(/storage/home/hcoda1/8/vsuarez6/r-jkretchmer3-0/MiniConda/bin/conda shell.bash hook)"
conda activate map-rpmd
# each task runs trajectories  (SLURM_ARRAY_TASK_ID * 1) .. +1-1  one at a time
python -m workflow.worker --config /storage/project/r-jkretchmer3-0/vsuarez6/repos/MASH-RPMD_Calculations_2026/MASH-RPMD_System-Bath/low_friction/config.py --task-id $SLURM_ARRAY_TASK_ID --chunk 1
