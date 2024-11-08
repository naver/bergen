# Multidomain RAG in BERGEN

[BERGEN](https://github.com/naver/bergen) is a library to benchmark retrieval-augmented generation (RAG) systems.
BERGEN supports a range of multi-domain query datasets along with each specific datastores from the literature, that we present below.

## Quick start

Example of launching evaluation on the bioasq12b dataset with retrieval from pubmed abstracts.

```bash
python3 bergen.py reranker='debertav3' retriever='splade-v3' generator='qwen-25-3b-instruct' dataset='multidomain/bioasq11b' prompt='multidomain/bioasq11b'
```

You can add `++experiments_folder=$exp_folder` to specify a custom folder to save results and `+run_name=$label"` to specify a custom experiment name.

For more details on installing BEGREN and running experiments, please refer to the [main readme](https://github.com/naver/bergen/tree/main). 

## Datasets

| Dataset | Description | Paper | Source Used | Datastore | Datastore Source | Train Set | Config and prompt at ```config/dataset/``` and ```config/prompt/``` | Preferred Metric(s) | Notes |
|-|-|-|-|-|-|-|-|-|-|
| Bioasq11b | Biomedical questions of types factoid/yesno/list| [Link](https://arxiv.org/abs/2307.05131) | Task B, 2023 [Link](https://huggingface.co/datasets/jenhsia/ragged) | Pubmed abstracts | [Link](https://huggingface.co/datasets/jenhsia/ragged) | No | ```multidomain/bioasq11b``` | Recall, LLMeval | Dropped 'summary' question types. Bioasq11b is directly accessible through the given HuggingFace source, while the bioasq12b data needs manual downloading from [bioasq challenge website](http://participants-area.bioasq.org/datasets/) before running it in bergen out-of-the-box. |
| Bioasq12b | Biomedical questions of types factoid/yesno/list| [Link](https://www.iit.demokritos.gr/wp-content/uploads/2024/10/paper-01-1.pdf) | Task B, 2024 [Link](http://participants-area.bioasq.org/datasets/) | Pubmed abstracts | [Link](https://huggingface.co/datasets/jenhsia/ragged) | Yes | ```multidomain/bioasq12b``` | Recall, LLMeval | Modified train-test split to increase test size, and dropped 'summary' question types. |
| CovidQA | Short and long-form QA related to covid-19 | [Link](https://aclanthology.org/2020.nlpcovid19-acl.18/) | [Link](https://huggingface.co/datasets/deepset/covid_qa_deepset) | CORD-19 | [Link](https://aclanthology.org/2020.nlpcovid19-acl.1.pdf) | No | ```multidomain/covidQA``` | LLMeval | Some queries are not "RAG-friendly" and depend on a fixed context which introduces *unanswerable* queries. |
| FiQA | Fact-based and Opinion-based QA for financial domain | Task 2 of [FiQA challenge](https://sites.google.com/view/fiqa/) | [Link](https://huggingface.co/datasets/LLukas22/fiqa) | StackExchange posts, *Investment* topic (2009-2017) | Corpus from [BeIR](https://huggingface.co/datasets/BeIR/fiqa) | No | ```multidomain/FiQA``` | LLMeval |  Documents are NOT parsed into chunks. |
| ParaphraseRC | Short, fact-based QA on movies. | [Link](https://arxiv.org/pdf/1804.07927) | [Link](https://huggingface.co/datasets/ibm/duorc/viewer/ParaphraseRC) | Movie plots | [Link](https://huggingface.co/datasets/ibm/duorc/viewer/ParaphraseRC) | Yes | ```multidomain/paraphraseRC``` | Recall, LLMeval | We take ParaphraseRC only and not SelfRC as it is reportedly a more challenging task. We parse each movie plot into chunks, and each chunk is pre-pended with the movie title. Each query is also pre-pended with the movie title. |
| RobustQA-Lifestyle | Various lifestyle questions (e.g. cooking, nutrition, everyday tasks, etc.) | [Link](https://aclanthology.org/2023.findings-acl.263/) | [Link](https://github.com/awslabs/rag-qa-arena/tree/main/data) | LoTTE (StackExchange) | [Link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) | No | ```multidomain/RobustQA_Lifestyle``` | LLMeval | LoTTE documents are parsed into chunks. |
| RobustQA-Recreation | QA on various video games | [Link](https://aclanthology.org/2023.findings-acl.263/) | [Link](https://github.com/awslabs/rag-qa-arena/tree/main/data) | LoTTE (StackExchange) | [Link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) | No | ```multidomain/RobustQA_Recreation``` | LLMeval | LoTTE documents are parsed into chunks. |
| RobustQA-Science | Scientific QA, e.g. math, physics, biology, etc. | [Link](https://aclanthology.org/2023.findings-acl.263/) | [Link](https://github.com/awslabs/rag-qa-arena/tree/main/data) | LoTTE (StackExchange) | [Link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) | No | ```multidomain/RobustQA_Science``` | LLMeval | LoTTE documents are parsed into chunks. |
| RobustQA-Technology | E.g. questions about security, hardware, software, etc. | [Link](https://aclanthology.org/2023.findings-acl.263/) | [Link](https://github.com/awslabs/rag-qa-arena/tree/main/data) | LoTTE (StackExchange) | [Link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) | No | ```multidomain/RobustQA_Technology``` | LLMeval | LoTTE documents are parsed into chunks. |
| RobustQA-Writing | E.g. questions about syntax, grammar, vocabulary, etc. | [Link](https://aclanthology.org/2023.findings-acl.263/) | [Link](https://github.com/awslabs/rag-qa-arena/tree/main/data) | LoTTE (StackExchange) | [Link](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) | No | ```multidomain/RobustQA_Writing``` | LLMeval | LoTTE documents are parsed into chunks. |
| SearchQA | General QA with context from search engine | [Link](https://arxiv.org/abs/1704.05179) | [Link](https://huggingface.co/datasets/kyunghyuncho/search_qa) | Search Engine Results | [Link](https://huggingface.co/datasets/kyunghyuncho/search_qa) | No | ```multidomain/SearchQA``` | Recall, LLMeval | The original dataset contains ~50 search results for each query; we merge all search results into one RAG datastore. |
| SyllabusQA | Short questions about course logistics | [Link](https://arxiv.org/abs/2403.14666) | [Link](https://github.com/umass-ml4ed/SyllabusQA/tree/main/data/dataset_split) | Courses syllabi | [Link](https://github.com/umass-ml4ed/SyllabusQA/tree/main/syllabi/syllabi_redacted/text) | Yes | ```multidomain/syllabusQA``` | Recall, LLMeval | We pre-pend the course title to each query, and chunk each course syllabus and each chunk is pre-pended with the course title. We merge all courses' chunks into a final RAG datastore. |
| TechQA | Technical support queries, *highly long-tail*, obtained by 'crawling the IBM Developer and IBM DeveloperWorks forums' | [Link](https://aclanthology.org/2020.acl-main.117.pdf) | [Link](https://huggingface.co/datasets/rojagtap/tech-qa) | Documents obtained from *'Technotes'* | [Link](https://huggingface.co/datasets/rojagtap/tech-qa) | No | ```multidomain/techQA``` | LLMeval | We concatenate the train/validation/test sets to get one bigger evaluation set. We chunk each document and all chunks are merged into a single datastore. |

The datastores for ```ParaphraseRC```, ```CORD-19```, ```LoTTE``` by default contain chunks of 100 words with 20 words overlap between consecutive chunks. The datastores for ```SyllabusQA``` and ```TechQA``` by default contain chunks of 1000 characters with 200 characters overlap between consecutive chunks.

## Benchmarks

We consistently use simple domain-specific system prompts as our experiments show increased metrics most of the time. It is always something simple, e.g. for bioasq: ```You are a biomedical expert. Your task is to answer questions using the given documents.```. We set the temperature to $0$ for reproducibility. When RAG is used we retrieve with ```splade-v3``` and rerank with ```debertav3```. For ```gemma-2b-it```, ```qwen-2.5-3b-instruct```, and ```llama3-8b-instruct``` we use 4-bit quantization. For ```SOLAR-10.7B-instruct```, we use [vllm](https://github.com/vllm-project/vllm). 

### Bioasq11b
| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.341 | 0.357 |
| gemma-2b-it | Yes | 0.470 | 0.589 |
| qwen-2.5-3b-instruct | No | 0.382 | 0.491 |
| qwen-2.5-3b-instruct | Yes | 0.649 | 0.769 |
| llama3-8b-instruct | No | 0.446 | 0.595 |
| llama3-8b-instruct | Yes | 0.615 | 0.762 |
| SOLAR-10.7b-instruct | No | 0.445 | 0.622 |
| SOLAR-10.7b-instruct | Yes | 0.668 | 0.791 |

### Bioasq12b
| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.247 | 0.367 |
| gemma-2b-it | Yes | 0.438 | 0.584 |
| qwen-2.5-3b-instruct | No | 0.362 | 0.494 |
| qwen-2.5-3b-instruct | Yes | 0.634 | 0.769 |
| llama3-8b-instruct | No | 0.462 | 0.600 |
| llama3-8b-instruct | Yes | 0.617 | 0.763 |
| SOLAR-10.7b-instruct | No | 0.431 | 0.609 |
| SOLAR-10.7b-instruct | Yes | 0.674 | 0.782 |

### CovidQA

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.229 | 0.213 |
| gemma-2b-it | Yes | 0.357 | 0.453 |
| qwen-2.5-3b-instruct | No | 0.305 | 0.336 |
| qwen-2.5-3b-instruct | Yes | 0.480 | 0.559 |
| llama3-8b-instruct | No | 0.321 | 0.328 |
| llama3-8b-instruct | Yes | 0.501 | 0.551 |
| SOLAR-10.7b-instruct | No | 0.311 | 0.405 |
| SOLAR-10.7b-instruct | Yes | 0.503 | 0.605 |

### FiQA

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.170 | 0.308 |
| gemma-2b-it | Yes | 0.169 | 0.317 |
| qwen-2.5-3b-instruct | No | 0.200 | 0.406 |
| qwen-2.5-3b-instruct | Yes | 0.201 | 0.391 |
| llama3-8b-instruct | No | 0.198 | 0.432 |
| llama3-8b-instruct | Yes | 0.230 | 0.438 |
| SOLAR-10.7b-instruct | No | 0.196 | 0.492 |
| SOLAR-10.7b-instruct | Yes | 0.218 | 0.499 |

### ParaphraseRC

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it  | No | 0.104 | 0.104 |
| gemma-2b-it  | Yes | 0.211 | 0.322 |
| qwen-2.5-3b-instruct  | No | 0.190 | 0.141 |
| qwen-2.5-3b-instruct  | Yes | 0.553 | 0.572 |
| llama3-8b-instruct  | No | 0.274 | 0.279 |
| llama3-8b-instruct  | Yes | 0.607 | 0.624 |
| SOLAR-10.7b-instruct  | No | 0.377 | 0.356 |
| SOLAR-10.7b-instruct  | Yes | 0.635 | 0.648 |

### RobustQA-Lifestyle

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.234 | 0.241 |
| gemma-2b-it | Yes | 0.201 | 0.320 |
| qwen-2.5-3b-instruct | No | 0.304 | 0.390 |
| qwen-2.5-3b-instruct | Yes | 0.337 | 0.565 |
| llama3-8b-instruct | No | 0.299 | 0.442 |
| llama3-8b-instruct | Yes | 0.358 | 0.592 |
| SOLAR-10.7b-instruct | No | 0.301 | 0.549 |
| SOLAR-10.7b-instruct | Yes | 0.360 | 0.688 |

### RobustQA-Recreation
| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.192 | 0.140 |
| gemma-2b-it | Yes | 0.185 | 0.261 |
| qwen-2.5-3b-instruct | No | 0.299 | 0.238 |
| qwen-2.5-3b-instruct | Yes | 0.381 | 0.498 |
| llama3-8b-instruct | No | 0.282 | 0.283 |
| llama3-8b-instruct | Yes | 0.367 | 0.532 |
| SOLAR-10.7b-instruct | No | 0.308 | 0.361 |
| SOLAR-10.7b-instruct | Yes | 0.412 | 0.603 |

### RobustQA-Science
| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.280 | 0.288 |
| gemma-2b-it | Yes | 0.242 | 0.376 |
| qwen-2.5-3b-instruct | No | 0.325 | 0.454 |
| qwen-2.5-3b-instruct | Yes | 0.339 | 0.457 |
| llama3-8b-instruct | No | 0.320 | 0.464 |
| llama3-8b-instruct | Yes | 0.364 | 0.568 |
| SOLAR-10.7b-instruct | No | 0.325 | 0.426 |
| SOLAR-10.7b-instruct | Yes | 0.377 | 0.634 |

*science prompt includes "think step by step"

### RobustQA-Technology
| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.239 | 0.241 |
| gemma-2b-it | Yes | 0.208 | 0.342 |
| qwen-2.5-3b-instruct | No | 0.289 | 0.380 |
| qwen-2.5-3b-instruct | Yes | 0.313 | 0.508 |
| llama3-8b-instruct | No | 0.290 | 0.440 |
| llama3-8b-instruct | Yes | 0.332 | 0.586 |
| SOLAR-10.7b-instruct | No | 0.283 | 0.422 |
| SOLAR-10.7b-instruct | Yes | 0.341 | 0.637 |

*technology prompt includes "think step by step"

### RobustQA-Writing
| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.231 | 0.257 |
| gemma-2b-it | Yes | 0.193 | 0.313 |
| qwen-2.5-3b-instruct | No | 0.335 | 0.470 |
| qwen-2.5-3b-instruct | Yes | 0.353 | 0.549 |
| llama3-8b-instruct | No | 0.287 | 0.521 |
| llama3-8b-instruct | Yes | 0.363 | 0.629 |
| SOLAR-10.7b-instruct | No | 0.318 | 0.567 |
| SOLAR-10.7b-instruct | Yes | 0.386 | 0.740 |

*writing prompt includes "think step by step"

### SearchQA

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.149 | 0.183 |
| gemma-2b-it | Yes | 0.315 | 0.350 |
| qwen-2.5-3b-instruct | No | 0.091 | 0.140 |
| qwen-2.5-3b-instruct | Yes | 0.671 | 0.724 |
| llama3-8b-instruct | No | 0.662 | 0.668 |
| llama3-8b-instruct | Yes | 0.722 | 0.780 |
| SOLAR-10.7b-instruct | No | 0.397 | 0.552 |
| SOLAR-10.7b-instruct | Yes | 0.687 | 0.754 |

### SyllabusQA

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.256 | 0.300 |
| gemma-2b-it | Yes | 0.195 | 0.231 |
| qwen-2.5-3b-instruct | No | 0.339 | 0.312 |
| qwen-2.5-3b-instruct | Yes | 0.391 | 0.278 |
| llama3-8b-instruct | No | 0.382 | 0.304 |
| llama3-8b-instruct | Yes | 0.311 | 0.313 |
| SOLAR-10.7b-instruct | No | 0.309 | 0.295 |
| SOLAR-10.7b-instruct | Yes | 0.305 | 0.284 |

### TechQA

| Generator | RAG | Recall | LLMeval (Llama-3.1-70b) |
|-|-|-|-|
| gemma-2b-it | No | 0.237 | 0.184 |
| gemma-2b-it | Yes | 0.268 | 0.253 |
| qwen-2.5-3b-instruct | No | 0.274 | 0.256 |
| qwen-2.5-3b-instruct | Yes | 0.412 | 0.528 |
| llama3-8b-instruct | No | 0.288 | 0.266 |
| llama3-8b-instruct | Yes | 0.469 | 0.576 |
| SOLAR-10.7b-instruct | No | 0.273 | 0.300 |
| SOLAR-10.7b-instruct | Yes | 0.461 | 0.597 |
