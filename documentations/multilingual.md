# Multilingual RAG in BERGEN

[BERGEN](https://github.com/naver/bergen) is a library to benchmark retrieval-augmented generation (RAG) systems.
BERGEN supports experimenting with multilingual user queries and/or multlingual datastore. 
This guide explains how to launch such multilingual evaluation.

## Quick start

Example of launching evaluation on the MKQA dataset in French, with retrieval from English Wikipedia:

```bash
python3 bergen.py generator='command-r-35b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_fr.retrieve_en' prompt='basic_translated_langspec/fr'
```

You can add `++experiments_folder=$exp_folder` to specify a custom folder to save results and `+run_name=$label"` to specify a custom experiment name. 
For more details on installing BEGREN and running experiments, please refer to the [main readme](https://github.com/naver/bergen/tree/main). 

## Datasets

For experimenting with multilingual user queries, BERGEN includes two datasets: [MKQA](https://github.com/apple/ml-mkqa) and [XOR TyDi QA](https://github.com/AkariAsai/XORQA). MKQA consists of the translations of questions from the [Natural Questions](https://ai.google.com/research/NaturalQuestions) dataset into various languages; all questions are therefore grounded in English Wikipedia by design. The XOR TyDi QA dataset consists of questions from the [TyDI QA](https://ai.google.com/research/tydiqa) dataset, and by design questions are grounded either in English Wikipedia or Wikipedia in the user lanuage.

For experimenting with retrieval from a multilingual datastore, BERGEN provides support for multiligual Wikipedia, particularly [Wikimedia HF dump](https://huggingface.co/datasets/wikimedia/wikipedia). Wikipedia articles are split into chunks of 100 words (or 100 Unicode charachetrs for Chinese, Japanese, and Thai) which are preneded with article titles.

Currently supported languages:
* for MKQA: `ar`, `zh`, `fi`, `fr`, `de`, `ja`, `it`, `ko`, `pt`, `ru`, `es`, `th`, `en`
* for XOR TyDi QA: `fi`, `ko`, `ar`, `ru`, `ja` (+`en` in TyDI QA)

Usage:
* MKQA: `dataset="mkqa/mkqa_${lang}.${retrieval_option}"`
* XOR TyDi QA: `dataset="xor_tydiqa/xor_tydiqa_${lang}.${retrieval_option}"`
* `${lang}` denotes language of user queries (see above)
* `${retrieval_option}` denotes the language setting for retrieval:
    * `retrieve_en`: retrieve from English wikipedia
    * `retrieve_${lang}`: retrieve from Wikipedia in user language
    * `retrieve_en_${lang}`: retrieve from concatenation of English wikipedia and Wikipedia in user language
    * `retrieve_all`: retrieve from concatenation of Wikipedia in all supported languages (same as for MKQA).
* Example: `dataset="mkqa/mkqa_fr.retrieve_en_fr"` sets testing on MKQA in French with retrieval from Wikipedia in English and French.
* Since XOR TyDI QA does not include English, we use the initial TyDI QA English set: `dataset="tydiqa"` or  `dataset="tydiqa.retrieve_all"`. For MKQA, use English MKQA subset: `dataset="mkqa/mkqa_en.retrieve_en"` or `dataset="mkqa/mkqa_en.retrieve_all"`.

## Prompts

We provide various prompts:
* `prompt="basic"`: simplest English prompt
* `prompt="basic_matchlang"`: English prompt with instruction to answer in the same language as the language of the query.
* `prompt="basic_langspec/{lang}"`: English prompt with an instruction to answer in an explicitely given language
* `prompt="basic_translated/{lang}"`: simple prompt translated into multiple languages
* `prompt="basic_translated_langspec/{lang}"`: prompt with an instruction to answer in an explicitely given language, translated into this language
* `prompt="basic_translated_langspec_namedentities/{lang}"`: same as the previous one, with added instruction to generate named entities in the given language

Based on our experiments, we recommend using `prompt="basic_translated_langspec/{lang}"`, as it ensures the highest correct language rate.

## Retrieval and reranking

For multilingual retrieval, BERGEN uses models from the [BGE-m3 project](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3):
* first-stage retrieval (queries and documents are encoded independently): `retriever="bge-m3"`
* second-stage reranker (queries and documents are encoded together using cross-encoder): `reranker="bge-m3"`

You can also experiment with prompt translation with the [NLLB-3.3B model](https://ai.meta.com/research/no-language-left-behind/): `+query_generator="translate/${lang}"`

## Generation

Multilingual generation model, [Command-R-35B](https://huggingface.co/CohereForAI/c4ai-command-r-v01): `generator="command-r-35b"`

With the use of advanced prompts, such as `prompt="basic_translated_langspec/{lang}"`, you can also try English-centric models such as `generator="SOLAR-107B"` (or its vllm version "vllm_SOLAR-107B") or `generator="mixtral-moe-7b-chat"` (or its vllm version "vllm_mixtral-moe-7b-chat").
Their performance depends on the language, e.g. Solar replies well in French but has troubles in Korean. VLLM version is much faster but we recommend checking that it performs well on the considered language.

## Adding a new language

To add a new language supported in MKQA or XOR TyDi QA, you will need to add corresponding config files in `config/dataset` (modify field `lang` in `query` according to [MKQA language codes](https://huggingface.co/datasets/apple/mkqa) and in `doc` according to [Wikimedia language codes](https://huggingface.co/datasets/wikimedia/wikipedia)) and 
in `config/prompt` (if you use language-dependent prompt, which usually leads to better results, see section "Prompts" above). 
In case the language uses the non-alphabetic writing system, you can also specify it in `modules/dataset_processor/Wiki_monolingual_100w/process/map_100w` function so that articles are splitted by characters and not words.


In case you use query translation, also add a new config in `config/query_generator/translate` (change `src_lang` field according to [Flores language codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)).

In case you need to add a new dataset, retriever, or generator, please refer to the [BERGEN extensions guide](https://github.com/naver/bergen/blob/main/documentations/extensions.md).