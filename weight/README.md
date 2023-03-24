# Symbol weight recommender

## Basic information

### Configuration

All the Python scripts use the configuration file `config.yaml`.
The configuration is managed by [Hydra](https://hydra.cc/) and the option values can be overriden via command line.
See [Hydra Docs](https://hydra.cc/docs/intro/).

Besides that, some configuration is done by environment variables.
See section Environment variables.

### GPU support

GPUs are currently not supported.
Set `CUDA_VISIBLE_DEVICES=-1` to ensure correct behavior.

## Environment variables

- `DGLBACKEND=tensorflow`: Recognized by Deep Graph Library (a dependency of the Python scripts).
  TensorFlow is the only supported backend, so setting `DGLBACKEND=tensorflow` is required for the Python scripts.
  See https://docs.dgl.ai/install/index.html#working-with-different-backends
- `CUDA_VISIBLE_DEVICES`: Recognized by CUDA (a dependency of the Python scripts).
  Set to -1 to disable GPUs.
  See https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
- `VAMPIRE`: Recognized by the Python scripts. Path to a Vampire executable.
- `TPTP`: Recognized by the Python and Bash scripts. Path to TPTP. Optional.
- `WORKSPACE`: Recognized by the Bash scripts.
  Path to a directory used to store experiment-related data, namely the training data.
- `N_JOBS`: Recognized by the Bash scripts. Number of jobs to run in parallel.

## Bash scripts

Inspect the Bash scripts to see detail about their interface (environment variables, arguments).

- `collect_tptp_fof_cnf_problems.sh` finds all FOF and CNF problems in TPTP, stores their paths in a file, and splits the problems into test and train_val sets.
- `collect.sh` collects training and validation proof search traces by running Vampire on the training and validation problems. Wraps `collect.py`.
- `train.sh` trains a graph neural network on the training data. Wraps `train.py`.
- `solve.sh` solves one or more problems using a trained graph neural network. Wraps `solve.py`.

## Python scripts

- `collect.py`
- `train.py`
- `solve.py`
