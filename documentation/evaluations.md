
## Evaluation


### Output files
Example files generated for split `dev` using `naver_splade-cocondenser-selfdistil` as a retriever.
- `config.yaml` The parameters of the experiment in yaml format.
- `eval_dev_generation_time.json` The generation time in json format.
- `eval_dev_metrics.json` Generation evaluation metrics in json format.
- `eval_dev_out.json` Output of the generation, contains `q_id` (str), `response` `(str)` the generated response, `label` `(list (str))` the answer reference (multiple possible), `instruction` `(str)` the instruction given to the generator, `ranking_label` `(list(list(str)), optional)` ids of reference paragraph (again multiple references possible).
- `run.retrieve.top_5.kilt_nq.dev.naver_splade-cocondenser-selfdistil.trec` The retrieval run in `trec` format.
- `eval_dev_ranking_metrics.json` Retrieval evaluation metrics in json format.


Non-neural metrics will be calculated automatically. Neural metrics such as `BEM` and `LLM` need to be evoked seperately.

By default `evaluate.py` will scan all folders in `experiments/` and evaluate them sequentially. To evaluate a single folder pass the folder using `--folder`. To avoid running out of memory either run `BEM` using `--bem` or run `LLM` using `--llm` . A csv file will automatically be saved to `results/` containing the table in `csv` format.

When using `--llm` you have a choice on how you transform LLM predictions in the final score:
- directly check in the generated answer for the expepected label occurence (default Yes/No), and assign corresponding score (default 1/0), when no expected label is found, or more than one expected label is matched, we assign score -100 to the corresponding sample, such samples are excluded from the mean score computation
- rely on the logits assigned to the first token (not available with vllm generators): get values corresponding to the expected labels, normalize them to 1 (get probability distribution across possible labels `p(label)`); final score would correspond to Inline equation: $\sum_{label} score(label)*p(label)$ 
The choice of score interpretation is done via `use_logits` parameter specified at evaluation config file. Default value is set to `True` (corresponding to the second option)


```bash
python3 evaluate.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm
```
Similarly to  `--generator` you can specify which LLM you are willing as first options of `--llm`, as well as short name at metrics naming (use the name of the configuration file as the name of the llm). 
 

```bash
# use llama2-7b-chat to run evaluation, output metric will be named VLLMeval_l2_7b
python3 evaluate.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm  "vllm_llama-2-7b-chat" "l2_7b"

# use tinyllama to run evaluation, output metric will be named LLMeval_tinyllama
python3 evaluate.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm  "tinyllama-chat" "tinyllama"

# in default settings (with no arguments specified) we use SOLAR-107B for evaluation and output metric is named LLMeval
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm  

```

You can specify prompt and other parameters in the evaluation config file for `--llm` at `config/evaluator` directory. By default they rely on `default_qa.yaml` configuration which assigns binary (Yes/No) value to each triple of <em>Question/Response/Gold Response</em>. You can specify finer granularity options and prompt (aka <em>rubrik section</em>). See example of more fine-grained configuration at `config/evaluator/default_multi_qa.yaml`. 

```bash
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm  --llm_prompt default_multi_qa
```


If you have local ollama server running, you can call models installed on this server as following:

```bash
python3 eval.py --experiments_folder experiments/ --llm_ollama "phi3:latest" --ollama_url "http://localhost:11434"   --llm_prompt default_multi_qa
```

### Pairwise comparisons

Instead of computing an LLM eval score for a given run, you can compare two outputs using the same script and some additional arguments e.g.
````
python3 evaluate.py --llm --folder mistral_preds --opponent_folder llama_preds  --opponent_name llama
```
where both `mistral_preds` and `llama_preds` are output folders of bergen inferences.
This scripts uses an LLM (can be any LLM supported in bergen or gpt-4o) to compare the two sets of predictions and compute win/tie/lose rates against the opponent. Results are stored in the metrics file of the folder. The prompt used is the pairwise prompt in `config/default_qa.yaml`.

This approach does not use logits but rather the raw prediction of the LLMs (win, tie or lose).

In this setup note that:
    - A single experiment folder must be specified for `--folder` and `--opponent_folder`
    - the `opponent_name` is required 