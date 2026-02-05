#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 200G
#SBATCH -p compsci

# Example run:
# sbatch -D ./slurm-logs/ --export="PYFILE=src/vit_train_cifar.py,ARGS=''" scripts/gpu_script.sh

cd # todo: ensure the script is running in the proper project directory

echo "batch: Starting job on $(date)"

# Note: ensure conda command is avaible, this may require sourcing the conda installation

# activate environment
conda activate chromatin-transformers

python $PYFILE $ARGS

echo $(date)
echo "batch: Completed job on $(date)"
