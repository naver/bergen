#!/bin/bash


BASE_TEMPLATE='#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --constraint="gpu_40g+"
#SBATCH --output=slurm_logs/%x_%j.log

source ~/.bashrc
export HYDRA_FULL_ERROR=1

conda activate bergen

'

LANGUAGES=("bn" "es" "fa" "fi" "hi" "id" "ja" "ko" "sw" "te" "th" "zh" "ar" "fr" "ru")
declare -a commands=()
for lang in "${LANGUAGES[@]}"; do
    commands+=("python -u gen_silver_labeling_provence.py \
--model Qwen/Qwen3-8B \
--datastore '../../../datasets/wiki-100w-${lang}_train' \
--queries '../../../datasets/miracl_${lang}_train' \
--trec '../../../runs/run.rerank.retriever.top_50.BAAI_bge-m3.rerank.top_50.miracl_${lang}.wiki-100w-${lang}.dev.BAAI_bge-reranker-v2-m3.trec' \
--outdir '../../../scripts/provence/qwen_miracl/miracl_data_${lang}_wiki' \
--overwrite \
--batch_size 256")
done

mkdir -p sbatch_scripts

# Generate individual sbatch scripts for each command
for i in "${!commands[@]}"; do
    # Extract run name for use in filename (simplified)
    run_name=$(echo "${commands[$i]}" | grep -o "miracl_[^ ]*_wiki" | sed "s/miracl_//;s/_wiki//")

    job_name="${run_name}"
    
    # Create script file
    script_file="sbatch_scripts/${run_name}.sh"
    
    # Write the sbatch script
    echo "${BASE_TEMPLATE}" > "${script_file}"
    echo "#SBATCH --job-name=${job_name}" >> "${script_file}"
    echo "${commands[$i]}" >> "${script_file}"
    
    # Make script executable
    chmod +x "${script_file}"
    
    # Submit the job
    echo "Submitting job for: ${run_name}"
    sbatch "${script_file}"
done

echo "All jobs submitted."