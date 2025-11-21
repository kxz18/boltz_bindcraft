#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem 16g
#SBATCH --time 24:00:00
#SBATCH --job-name=boltzgo
#SBATCH --output=/work/lpdi/users/xkong/codes/boltz_bindcraft/tmp/logs/boltzgo.%x_%j.out
#SBATCH --error=/work/lpdi/users/xkong/codes/boltz_bindcraft/tmp/logs/boltzgo.%x_%j.error

module load gcc/13.2.0
module load cuda/12.4.1
module load cudnn/9.2.1.18-12

echo "Setting up virtual python environment..."
source /work/lpdi/users/xkong/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /work/lpdi/users/xkong/codes/boltz_bindcraft/env
echo "BoltzGO environment succesfully loaded!"

SEED=$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID

CONFIG=$1
OUTDIR=$2

cd /work/lpdi/users/xkong/codes/boltz_bindcraft/src
mkdir $OUTDIR
python -m design.main --config $CONFIG --out_dir $OUTDIR --ckpt_dir ../.boltz/ 2>&1 | tee $OUTDIR/log.txt
