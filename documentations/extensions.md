
# Extend Library
 
## Add Retriever
Retrievers inherit from the abstract `Retriever` class and thus needs to follow this structure:

```python
from models.retrievers.retriever import Retriever

class NewRetriever(Retriever):
  def __init__(self, model_name=None):
    self.model_name = 'new_retriever'

  @abstractmethod
  def __call__(self, kwargs):
    # model inference e.g. return model(**kwargs)
    pass

  @abstractmethod
  def collate_fn(self, batch, query_or_doc=None):
    # implement collate_fn here
    pass 

  @abstractmethod
  def similarity_fn(self, q_embds, doc_embs):
    # similarity fn to use e.g. torch.mm.(q_embs, doc_embs.t())
    pass 
```
We save it under `models/retrievers/new_retriever.py`.

As the second step create a config for this model under `config/retrievers/new_retriever.yaml`. 

```yaml
init_args: 
  _target_: models.retrievers.new_retriever.NewRetriever
  model_name: "new_retriever"
batch_size: 1024
batch_size_sim: 256
```

To use the model add the argument `retriever='new_retriever'`:

```python
python3 main.py retriever='new_retriever'
```


## Add Reranker
Rerankers inherit from the abstract `Reranker` class and thus needs to follow this structure:

```python
from models.rerankers.reranker import Reranker

class NewReranker(Reranker):
  def __init__(self, model_name=None):
    self.model_name = 'new_reranker'

  @abstractmethod
  def __call__(self, kwargs):
    # model inference e.g. self.model(**kwargs)
    pass

  @abstractmethod
  def collate_fn(self, batch, query_or_doc=None):
    # implement collate function 
    pass

```

We save it under `models/rerankers/new_reranker.py`.

As the second step create a config for this model under `config/rerankers/new_reranker.yaml`. 

```yaml
init_args: 
  _target_: models.rerankers.new_reranker.NewReranker
  model_name: "new_reranker"
batch_size: 2048
```

To use the model add the argument `reranker='new_reranker'`:

```python
python3 main.py reranker='new_reranker'
```

### Add Generator
The Generator inherits from the abstract `Generator` class and thus needs to follow this structure:

```python
from models.generators.generator import Generator

class NewGenerator(Generator):
  def __init__(self, model_name=None):
    self.model_name = 'new_generator'

  @abstractmethod
  def generate(self, inp):
    # generation e.g. self.model(**inp)
    pass
  @abstractmethod
  def collate_fn(self, inp):
    pass

  # only required for training
  @abstractmethod
  def prediction_step(self, model, model_input, label_ids=None):
      # e.g.       
      # output = model(**model_input, labels=label_ids)
      # return output.logits, output.loss
      pass 

```

We save it under `models/generators/new_generator.py`.

As the second step create a config for this model under `config/generators/new_generator.yaml`.


```yaml
defaults:
  - prompt: basic
init_args: 
  _target_: models.generators.new_generator.NewGenerator
  model_name: "new_generator"
  max_new_tokens: 128
batch_size: 32
max_inp_length: null
```


To use the model add the argument `generator='new_generator'`:

```python
python3 main.py generator='new_generator'
```


## Add Dataset
A dataset config contains two entries: `doc` for the collection and `query` for the queries.

A query dataset **must** contain the fields **`id`**, `wikipedia_id` (optional), **`content`**, **`label`** after the processing. 

A document dataset **must** contain the fields **`id`**, and **`content`** after the processing.

Define a new dataset class in `modules/dataset_processor.py`

```python
class NewDataset(Processor):

  def __init__(self, *args, **kwargs):
    # name under which the dataset will be saved 'datasets/new_dataset_{split}' (default)
    dataset_name = 'new_dataset'
    super().__init__(*args, **kwargs, dataset_name = dataset_name)

  def process(self):
    # load model 
    # e.g. for hf hub 
    #dataset = datasets.load_dataset('hf_dataset_name')
    def map_fn(example):
      # do some mapping
      return example

    dataset = dataset.map(map_fn, num_proc=self.num_proc)
    return dataset
```

To use the dataset add a new dataset config e.g. `config/dataset/new_config.yaml` using the new class `NewDataset` for the collection (`doc` field). As a query we are using an already existing Dataset `KILTNQProcessor`. Additinally, add the field `split` which defines which split within the dataset should be used. 

```yaml
test:
    doc: null
    query: null
dev:
  doc: 
    init_args:
    _target_: modules.dataset_processor.NewDataset
    split: "full"
query:
  init_args:
    _target_: modules.dataset_processor.KILTNQProcessor
    split: "validation"
train:
    doc: null
    query: null
```

All datasets can be overwritten by adding `+overwrite_datasets=True` as an argument (`Caution`: This might overwrite collections that take long long to encode). In case the indexing is interrupted you can continue encoding a collection from batch 1000 by additionally using the argument `+continue_batch=1000`.



## Add Prompt
Prompts are stored in `config/prompt/` via the argument `prompt`.

Create a new prompt `new_prompt` under  `config/prompt/new_prompt.yaml`
An exmaple prompt could look like this. THE local variables (e.g. `query`) will insterted into the formatted string within the respective models' `format_instruction()` function.
`Important`: empty spaces after a colon within the formatted string need to be escaped like to `Question:\ `.

```yaml
system: "You are a helpful assistant. Your task is to extract relevant information from the provided documents and to answer questions accordingly."
user: f"Background:\ {docs}\n\nQuestion:\ {question}\nAnswer:"
system_without_docs: "You are a helpful assistant."
user_without_docs: f"Question:\ {question}\nAnswer:"
```

To use the prompt pass it as an argument: 

```bash
python3 main.py generator='tinyllama-chat' prompt='new_prompt'

```


# Oracle
## Oracle Answer

Using the oracle answers instead of generating using a LLM.

For running the generation simply use the generator `oracle_answer`. For example: 

```python
python3 main.py dataset='kilt_nq' generator='oracle_answer'
```

## Oracle Provenances

To generate all oracle runs (trec runs) and save them in `runs` execute the script `scripts/kilt_generate_oracle.py` once.


### Oracle Provenances as Input to LLM

Generating answers using Llama with the oracle provenances as documents. 

For running the generation with e.g. `llama-2-7b-chat` simply select `orcale_provenance` as a retriever. For example: 

```python
python3 main.py dataset='kilt_nq' retriever='oracle_provenance' generator='llama-2-7b-chat'
```

## Testing
If you want to develop new functionalities, or want to test basic BERGEN configurations:

Run all tests:

```bash
pytest tests/
```
