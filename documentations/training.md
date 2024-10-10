

## Training
For training a model add a training config e.g. `train=lora` as an argument, e.g.

```bash
python3 main.py retriever='bm25' generator='llama-2-7b-chat' dataset='kilt_nq' train='lora'
```

For training the `train` dataset split that is defined in the config is split in `train` and `test` splits ( default test size: `0.01`). The best model (according to the newly generated `test` split) is loaded after the training and evaluated on the `dev`  dataset split.

To monitor training using wandb:
- ```pip install wandb```
- in the execution environment: set variable ```WANDB_API_KEY``` (run ```export WANDB_API_KEY="<your api key>"```)
- in training config: set ```trainer.report_to: "wandb"``` and ```wandb_project_name: <your project name>```
- finally, make sure to activate wandb: ```wandb enabled``` and sync to the wandb cloud: ```wandb online```