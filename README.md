# Vampire+ML

Machine learning extensions for [Vampire](https://vprover.github.io/)

Experiments in Neptune.ai: [filipbartek/vampire-ml](https://ui.neptune.ai/filipbartek/vampire-ml/)

## Quick start

Before running the experiments for the first time,
it is necessary to build the Vampire prover located in the Git submodule `vampire`.

Before running an experiment, configure the following environment variables:

- `DGLBACKEND=tensorflow`
- `XDG_CACHE_HOME`: Path to a cache directory. System default: `$HOME/.cache`
- `VAMPIRE`: Path to a Vampire prover binary. Example: `vampire/build/release/bin/vampire`
- `TPTP`: Path to a [TPTP](http://www.tptp.org/) library directory. Example: `$HOME/TPTP-v7.4.0`

Call the module `qustions` to run an experiment.
Example calls:

```sh
# Train a predicate precedence recommender using a dataset of 1000000 examples
python -m questions questions.max_count=1000000 questions.randomize=[predicate] symbol_type=predicate

# Evaluate a predicate precedence recommender stored in the checkpoint "outputs/2021-02-16/12-28-14/tf_ckpts/epoch/weights.00289.tf" on all validation problems
python -m questions questions.max_count=1000000 questions.randomize=[predicate] symbol_type=predicate restore_checkpoint=outputs/2021-02-16/12-28-14/tf_ckpts/epoch/weights.00289.tf epochs=0 solver_eval.start=-1 solver_eval.iterations=5 solver_eval.problems.train=0 solver_eval.problems.val=null

# Print the supported parameters
python -m questions --help
```
