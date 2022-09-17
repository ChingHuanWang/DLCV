#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

source activate b08901192
cd $PBS_O_WORKDIR
module load cuda/cuda-10.0/x86_64
python hw1_1.py ###你要跑的code
source deactivate