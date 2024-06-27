## Requirements

For CUDA>=12
```
conda create -n "bergen" python=3.10
conda activate bergen
pip install torch 
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation   #skip it for V100
pip install vllm 
git clone https://github.com/naver/bergen.git
cd bergen
pip install -r requirements.txt
```
## Installing Pyserini
(Make sure you have the right jdk version used by pyserini)
```
conda install -c conda-forge openjdk=21 maven -y
pip install torch faiss-cpu
pip install pyserini
```


For CUDA 11.8

```
# Install vLLM with CUDA 11.8.
conda create -n "bergen" python=3.10
conda activate bergen
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=39
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation   #skip it for V100
pip install vllm 
git clone https://github.com/naver/bergen.git
cd bergen
pip install -r requirements.txt
```