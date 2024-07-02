exp_folder=YOUR FOLDER

# write here your code for launching one exp, e.g. sbatch parameters 
runexp() {
    LABEL=$1
    shift
    echo sbatch --output="$exp_folder/out_${LABEL}.txt" --error="$exp_folder/${LABEL}.txt" --wrap="$@"
}

for lang in ar zh fi fr de ja it ko pt ru es th; do 
    # no retrieval
    label=mkqa_cmdr_noret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' dataset='mkqa/mkqa_${lang}.retrieve_en' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in English
    label=mkqa_cmdr_enret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_${lang}.retrieve_en' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in user language
    label=mkqa_cmdr_langret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_${lang}.retrieve_${lang}' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in English and user language
    label=mkqa_cmdr_langenret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_${lang}.retrieve_en_${lang}' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in all languages
    label=mkqa_cmdr_allret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_${lang}.retrieve_all' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
done

lang=en
# no retrieval
label=mkqa_cmdr_noret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' dataset='mkqa/mkqa_${lang}.retrieve_en' prompt='basic' ++experiments_folder=$exp_folder +run_name=$label
# retrieval in English
label=mkqa_cmdr_enret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_${lang}.retrieve_en' prompt='basic' ++experiments_folder=$exp_folder +run_name=$label
label=mkqa_cmdr_allret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_${lang}.retrieve_all' prompt='basic' ++experiments_folder=$exp_folder +run_name=$label
    

for lang in fi ko ar ru ja; do 
    # no retrieval
    label=xorqa_cmdr_noret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' dataset='xor_tydiqa/xor_tydiqa_${lang}.retrieve_en' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in English
    label=xorqa_cmdr_enret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='xor_tydiqa/xor_tydiqa_${lang}.retrieve_en' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in user language
    label=xorqa_cmdr_langret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='xor_tydiqa/xor_tydiqa_${lang}.retrieve_${lang}' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in English and user language
    label=xorqa_cmdr_langenret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='xor_tydiqa/xor_tydiqa_${lang}.retrieve_en_${lang}' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
    # retrieval in all languages
    label=xorqa_cmdr_allret_${lang}; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='xor_tydiqa/xor_tydiqa_${lang}.retrieve_all' prompt='basic_translated_langspec/$lang' ++experiments_folder=$exp_folder +run_name=$label
done

lang=en
label=tydiqa_cmdr_noret_${lang}; runexp $label python3 bergen.py generator='command-r-35b'  dataset='tydiqa' prompt='basic' ++experiments_folder=$exp_folder +run_name=$label
label=tydiqa_cmdr_enret_en; runexp $label python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='tydiqa' prompt='basic' ++experiments_folder=$exp_folder +run_name=$label
label=tydiqa_cmdr_allret_${lang}; runexp $label python3 main.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='tydiqa.retrieve_all' prompt='basic' ++experiments_folder=$exp_folder +run_name=$label
