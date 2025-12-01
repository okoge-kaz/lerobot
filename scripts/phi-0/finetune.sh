#!/bin/sh
#PBS -q rt_HF
#PBS -N pi0
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m n
#PBS -koed
#PBS -V
#PBS -o outputs/pi0/

cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.9.1
module load cudnn/9.10.2
module load nccl/2.27.5-cuda12.9
module load hpcx/2.20

source .venv/bin/activate

# distributed settings
JOB_ID=$(echo $PBS_JOBID | cut -d. -f1)
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=8
NODE_TYPE="h200"

NODEFILE=$PBS_NODEFILE
NODE_COUNT=$(sort -u $NODEFILE | wc -l)
NUM_NODES=$NODE_COUNT
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
sort -u "$PBS_NODEFILE" | while read -r line; do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done >"$HOSTFILE_NAME"

# training settings
TRAIN_ITERATIONS=100000
BATCH_SIZE=2

EVAL_INTERVAL=2000
SAVE_INTERVAL=2000


DATASET_HF_REPO_ID="lerobot/pusht"

export WANDB_ENTITY="okoge"
export WANDB_PROJECT_NAME="lerobot"

python src/lerobot/scripts/lerobot_train.py \
  --wandb.project=${WANDB_PROJECT_NAME} \
  --wandb.enable=true \
  --policy.type='pi0' \
  --policy.repo_id=lerobot/pi0 \
  --policy.device=cuda \
  --dataset.repo_id=${DATASET_HF_REPO_ID} \
  --env.type=pusht \
  --batch_size=${BATCH_SIZE} \
  --steps=${TRAIN_ITERATIONS} \
  --eval_freq=${EVAL_INTERVAL} \
  --log_freq=1 \
  --save_checkpoint=True \
  --save_freq=${SAVE_INTERVAL} \
  --use_policy_training_preset=True
