#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=Autoformer
#SBATCH --mem-per-cpu=10G
#SBATCH -n 8
#SBATCH -c 1
#SBATCH -o /scratch/itee/uqxxu16/Autoformer/tensor_out.txt
#SBATCH -e /scratch/itee/uqxxu16/Autoformer/tensor_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:4

module load anaconda/3.6
source activate /clusterdata/uqxxu16/.conda/env/autoformer

srun python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path ./data/imagenet --gp --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/subnet/AutoFormer-A.yaml --batch-size=64
