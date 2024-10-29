#!/bin/bash
#SBATCH -A m4716
#SBATCH -J HydraGNN
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32
#SBATCH -N 2

## Remove write permission for others in terms of newly created files and dirs
umask 002

## Load Basic Envs
module reset
module load pytorch/2.0.1

module use -a /global/cfs/cdirs/m4133/jyc/perlmutter/sw/modulefiles
module load hydragnn/pytorch2.0.1-v2
module use -a /global/cfs/cdirs/m4133/c8l/sw/modulefiles
module load deepspeed

## MPI Envs
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=0

## HYDRAGNN Envs
HYDRAGNN_DIR=/global/cfs/cdirs/m4716/c8l/HydraGNN
export PYTHONPATH=$HYDRAGNN_DIR:$PYTHONPATH

export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_TRACE_LEVEL=0

## Dataset Envs
DATASET_PATH="/global/cfs/projectdirs/m4716/mlupopa/HydraGNN/examples/multidataset_hpo/dataset"
DATASET_LIST="MPTrj-v3"

## run scripts
set -x

srun -N2 -n8 -c32 --ntasks-per-node=4 --gpus-per-task=1 \
    python -u $HYDRAGNN_DIR/examples/multidataset_deepspeed/train.py \
        --inputfile=base.json \
        --dataset_path=$DATASET_PATH \
        --multi \
        --multi_model_list=$DATASET_LIST \
        --num_epoch=10 \
        --everyone --ddstore \
        --log=exp_base

set +x
