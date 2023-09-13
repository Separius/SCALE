# Installation Guide
In this short guide we assume that you have CUDA, python, and conda installed.
We used cuda=11.6 and python=3.8, but other versions might work as well.

```bash
# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
```bash
# install torchvision (v0.13.0)
mkdir opt && cd opt
conda remove --force torchvision
git clone https://github.com/pytorch/vision.git
cd vision
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
conda install libpng jpeg ninja ipython
conda install -c conda-forge ffmpeg accimage
git checkout tags/v0.13.0
python setup.py install
cd ..
```
```bash
# install other packages
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"

git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
BUILD_CUDA_EXTENSIONS=1 TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation . # this is compute capability
cd ..

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
python setup.py install
cd ..

pip install 'git+https://github.com/facebookresearch/fvcore' simplejson psutil opencv-python tensorboard moviepy matplotlib
conda install av -c conda-forge
conda install -c iopath iopath
conda install scipy pandas
# we have to change SlowFast a bit too, so we need a local copy of that
export PYTHONPATH=LOCAL_PATH/SlowFast:$PYTHONPATH

pip install hydra-core functorch timm
# change lines 13 and 30 of ~/.conda/envs/ENV_NAME/lib/python3.8/site-packages/pytorchvideo/layers/attention.py
```

# prepare kinetics:
download it from [here](https://github.com/cvdfoundation/kinetics-dataset)
then replace the corrupted files with the right ones and finally [resize them](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics)
