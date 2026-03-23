#!/bin/bash
#SBATCH --job-name=data-block-N-uchar4
#SBATCH --output=data-block-N-uchar4.%j.out
#SBATCH --error=data-block-N-uchar4.%j.err
#SBATCH --account=156171513713
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:h100:1

set -e
module load CUDA/12.0

BIN=/scratch/user/u.yg345014/my_research_project/cuSZp/build/examples/bin/cuSZp_dblk
DS=/scratch/user/u.yg345014/my_research_project/datasets

for f in $DS/EXASKY-HACC-data-medium/*.f32; do $BIN -i "$f" -eb rel 1e-3; done
for f in $DS/SDRBENCH-EXASKY-NYX-512x512x512/*.f32; do $BIN -i "$f" -eb rel 1e-3; done
for f in $DS/SDRBENCH-Hurricane-ISABEL-100x500x500/*.f32; do $BIN -i "$f" -eb rel 1e-3; done
for f in $DS/qwen3.5-2b/out/*.f32; do $BIN -i "$f" -eb rel 1e-3; done
