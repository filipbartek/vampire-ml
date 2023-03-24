# Vampire symbol weight recommender

Machine learning extension for [Vampire](https://vprover.github.io/)

## Setup

```sh
# Download Git submodules
git submodule update --init --recursive

# Build Vampire
pushd vampire
mkdir -p build
cd build
cmake ..
make
ln -s bin/vampire_* vampire
popd

# Set up the Python interpreter
# If preferred, set up a conda environment first.
conda install tensorflow=*=gpu_
# DGL installation instructions: https://www.dgl.ai/pages/start.html
conda install -c dglteam/label/cu113 dgl
pip install -r requirements.txt
```

## Usage

To train or use the symbol weight recommender,
follow the instructions in `weight/README.md`.
