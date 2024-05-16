## Requirements

```
conda create -n "bergen" python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install -r requirements.txt
```

