#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#S BATCH --requeue
#SBATCH --job-name=bopt1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1
##SBATCH --constraint="gpu_40g+"
#SBATCH --constraint="gpu_32g+"
#  S BATCH --constraint="cuda12"
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc

echo $PWD
#cd /home/sclincha/RAG
export JAVA_HOME="/nfs/core/java/jdk-11"
export PYTHONPATH=$PWD
#export HF_HOME=/beegfs/scratch/user/sclincha/huggingface_cache_reml
export HF_HOME=/beegfs/scratch/user/hdejean/cache/
conda activate reml


CONFIG=rag-nle python3 main.py  retriever=retromae reranker=debertav3  generator=llama-2-7b-chat dataset='kilt_wow'