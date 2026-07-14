#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_80g|gpu_80g+
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/scratch/user/hdejean/bergen/expPISCO/slurm-%j.log
#SBATCH --error=/beegfs/scratch/user/hdejean/bergen/expPISCO/slurm-%j.err
# args: RUNPREFIX RETRIEVER RERANK(-|name) PROMPT "DATASETS" "GENSPEC" "EXTRA"
source ~/.bashrc
cd /beegfs/scratch/user/hdejean/bergen
PY=/beegfs/scratch/user/hdejean/pisco/.pixi/envs/default/bin/python
RUNPREFIX=$1; RETR=$2; RERANK=$3; PROMPT=$4; DATASETS=$5; GENSPEC=$6; EXTRA=$7; RUNSF=${8:-/beegfs/scratch/project/calmar/rag-benchmark/V1/runs}
RERANK_ARG=""; [ "$RERANK" != "-" ] && RERANK_ARG="reranker=$RERANK"
EXP=/beegfs/scratch/user/hdejean/bergen/expPISCO
for DS in $DATASETS; do
  echo "===== $RUNPREFIX $DS ====="
  CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CPATH=/usr/local/cuda/include \
  HF_HOME=/beegfs/scratch/user/hdejean/cache \
  PYTHONPATH=/beegfs/scratch/user/hdejean/bergen:/beegfs/scratch/user/hdejean/pisco \
  $PY bergen.py --config-name=rag --config-path=config \
    run_name=${RUNPREFIX}_${DS} experiments_folder=$EXP \
    index_folder=/beegfs/scratch/project/calmar/rag-benchmark/V0/indexes256/ \
    runs_folder=$RUNSF \
    dataset_folder=/beegfs/scratch/user/hdejean/calmar/bergen/datasets \
    dataset=$DS prompt=$PROMPT generation_top_k=5 +gpu_memory_utilization=0.5 \
    retriever=$RETR $RERANK_ARG $GENSPEC $EXTRA
  echo "$DS exit=$?"; cat $EXP/${RUNPREFIX}_${DS}/eval_dev_metrics.json 2>/dev/null
done
echo "===== $RUNPREFIX ALL DONE ====="
