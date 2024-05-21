## Requirements

(see vllm install for CUDA <12)
```
conda create -n "bergen" python=3.10
pip install pytorch 
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
pip install vllm
pip install -r requirements.txt
```

## Installing Pyserini

(Make sure you have the right jdk version used by pyserini)
```
conda install -c conda-forge openjdk=21 maven -y
pip install torch faiss-cpu
pip install pyserini

```
