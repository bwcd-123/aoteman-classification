#!/bin/bash
#JSUB -q AI205009
#JSUB -n 10
#JSUB -gpgpu 1
#JSUB -e error/single_train.%J
#JSUB -o output/single_train.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh
module load cuda/11.6
conda activate torch39
python train_single_cuda.py  --data_path ../datasets/aoteman  --batch_size 110  --num_workers 10