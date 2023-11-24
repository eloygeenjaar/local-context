#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=192g
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH -t 1200
#SBATCH -J comms
#SBATCH -e jobs/error-comms-%A.err
#SBATCH -o jobs/out-comms-%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dkim195@gsu.edu
#SBATCH --oversubscribe
#SBATCH --exclude=arctrdagn007

sleep 10s

export PATH=/data/users1/dkim195/miniconda3/bin:$PATH
source /data/users1/dkim195/miniconda3/etc/profile.d/conda.sh
conda activate /data/users1/dkim195/miniconda3/envs/pytorch
python main.py

sleep 30s
