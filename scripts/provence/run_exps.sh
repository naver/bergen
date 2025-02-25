#!/bin/bash

# this script shows which configurations to use for the main figure of the Provence paper (Figure 2)
# for your first runs, we recommend selecting a subset of datasets and methods below, to test that everything works

cd ../..

# step 1: change exp folder to your location
OUTFOLDER=experiments_provence
mkdir -p $OUTFOLDER

# step 2: you can add your sbatch params here
SBATCH_PARAMS=""

# step 3: run this script

# step 4: after all the runs are finished, run LLMeval :
# python3 eval.py --experiments_folder $OUTFOLDER --llm_batch_size 16 --split 'dev' --vllm

for dataset in wiki_cntx_granularities/nq_castorini_6-3; do
#for dataset in wiki_cntx_granularities/nq_castorini_6-3 wiki_cntx_granularities/hotpotqa_castorini_6-3 wiki_cntx_granularities/tydiqa_castorini_6-3 wiki_cntx_granularities/popqa_castorini_6-3 multidomain/syllabusQA multidomain/pubmed_bioasq11b_ragged multidomain/rgb; do
# full version with all datasets from Provence paper

    for method in full provence/provence_rerank_0.5; do
    #for method in full provence/provence_rerank_0.5 provence/provence_rerank_0.1 provence/provence_standalone_0.5 provence/provence_standalone_0.1 recomp/recomp_ext_top1 recomp/recomp_ext_top2 recomp/recomp_ext_top3 llmlingua2/llmlingua2_0.25 llmlingua2/llmlingua2_0.5 llmlingua2/llmlingua2_0.7 longllmlingua/longllmlingua_0.25 longllmlingua/longllmlingua_0.5 longllmlingua/longllmlingua_0.75 dslr/dslr_ce_bge_t01 dslr/dslr_ce_bge_t02 dslr/dslr_ce_bge_t05; do
    # full version with all the methods from the main plot Figure 2

        methodlabel=$(basename $method)
        datasetlabel=$(basename $dataset)
        # full = full context baseline, regular RAG pipeline without context compression
        if [ "$method" = "full" ]; then
            addons=""
        else
            addons="+context_processor=${method}"
        fi
        if [ "$dataset" = "multidomain/rgb" ]; then
            retline="retriever='oracle_provenance'"
        else
            # to use Provence as reranker, skip `reranker` in command line args and use "rerank" config of Provence,
            # e.g. +context_processor=provence/provence_rerank_0.1 (already set above)
            if [[ "$method" == *"provence_rerank"* ]]; then
                retline="retriever='splade-v3' ++generation_top_k=50"
            else
                retline="retriever='splade-v3' reranker='debertav3' ++generation_top_k=5"
            fi
        fi
        if [ "$method" = "multidomain/syllabusQA" ]; then
            promptline="prompt='multidomain/syllabusQA'"
        else
            promptline="prompt='basic'"
        fi

        sbatch $SBATCH_PARAMS --wrap="python3 bergen.py dataset='$dataset' $retline $addons $promptline +run_name=${datasetlabel}_${methodlabel} generator='vllm_llama-2-7b-chat' ++experiments_folder=$OUTFOLDER"
    done
done

