#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --partition=gpu2 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=UBAIJOB 
#SBATCH -o ./logs/jupyter.%N.%j.out  # STDOUT 
#SBATCH -e ./logs/jupyter.%N.%j.err  # STDERR

echo "start at:" `date` 
echo "node: $HOSTNAME" 
echo "jobid: $SLURM_JOB_ID" 

module unload CUDA/11.2.2 
module load cuda/11.8.0

export PYTHONPATH=.

python src/data/process.py \
    --data data/black_vs_noise \
    --output data/black_vs_noise/data_light.jsonl \
    --filter black_vs_noise_filter_light \

python src/data/process.py \
    --data data/center_vs_random \
    --output data/center_vs_random/data_light.jsonl \
    --filter center_vs_random_filter_light \

python src/data/process.py \
    --data data/color_vs_wb \
    --output data/color_vs_wb/data_light.jsonl \
    --filter color_vs_wb_filter_light \

python src/data/process.py \
    --data data/direction \
    --output data/direction/data_light.jsonl \
    --filter direction_filter_light \