#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mem 200G
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu

# Example run:
# sbatch -D ./slurm-logs/ --export="PYFILE=src/vit_train_cifar.py,ARGS=''" scripts/gpu_script.sh

cd /usr/xtmp/tqtran/chromatin-transformers

echo "batch: Starting job on $(date)"

. /usr/xtmp/tqtran/anaconda3/etc/profile.d/conda.sh

# activate environment
conda activate cadmium-py3

python $PYFILE $ARGS

echo $(date)
echo "batch: Completed job on $(date)"
