<div align="center">
      <h1>SimpleGeneration</h1>
</div>

<div align="center">
    <p align="center">
          <em> Open-source / Simple / Lightweight / Easy-to-use / Extensible </em>
    </p>
</div>

<hr>

# Introduction

**This repository provides pytorch training examples for generation model.**

# Training GPU server

# Environments

**1、Python and Pytorch Supported Version: Python>=3.12, Pytorch>=2.5.1.**

**2、(optional)Add HF_HOME dir HF_ENDPOINT dir in .bashrc and .zshrc:**
```
# Add HF_HOME dir and HF_ENDPOINT dir in .bashrc and .zshrc files:
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
```
```
source .bashrc
source .zshrc
```

**3、Create a conda environment:**
```
conda create -n SimpleGeneration python=3.12
```

**4、Install PyTorch:**
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```
To install a different PyTorch version, find command from here:

https://pytorch.org/get-started/previous-versions/

**5、Install other Packages:**
```
pip install -r requirements.txt
```

# Download my pretrained models and experiments records


# Prepare datasets


# How to train or test a model


# Reference

```
https://github.com/FoundationVision/LlamaGen
https://github.com/duchenzhuang/FSQ-pytorch
https://github.com/CompVis/taming-transformers
https://github.com/CompVis/latent-diffusion
https://github.com/bytedance-seed/BAGEL
https://github.com/boogu-project/Boogu-Image
https://github.com/ByteVisionLab/DreamLite
https://github.com/baidu/ERNIE-Image
https://github.com/FireRedTeam/FireRed-Image-Edit
https://github.com/CostaliyA/Flow-OPD
https://github.com/black-forest-labs/flux
https://github.com/black-forest-labs/flux2
https://github.com/zlab-princeton/i1
https://github.com/ideogram-oss/ideogram4
https://github.com/jd-opensource/JoyAI-Image
https://github.com/krea-ai/krea-2
https://github.com/bytedance/Lance
https://github.com/microsoft/Mage
https://github.com/Hope7Happiness/minit2i-torch
https://github.com/jmliu206/SeFi-Image
https://github.com/Tongyi-MAI/Z-Image
```

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zgcr,
 title={SimpleGeneration-pytorch-training-examples},
 author={zgcr},
 year={2020-2030}
}
```