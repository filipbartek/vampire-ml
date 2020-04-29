# Problem lists

## cnf_fof.txt

Command:

```bash
. env.sh
find "$TPTP_PROBLEMS" -regextype sed -regex ".*/[A-Z]\{3\}[0-9]\{3\}[-+][1-9][0-9]*\(\.[0-9]\{3\}\)*\.p" -exec realpath --relative-to "$TPTP_PROBLEMS" {} + | sort > cnf_fof.txt
```

TPTP version: v7.3.0

## problems_selected_aggregated.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-both/batches/404135/aggregate/problems.pkl
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-predicate/batches/404131/aggregate/problems.pkl
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-function/batches/404133/aggregate/problems.pkl
```

Command:

```
join_problems.py data/sp-random-both/problems.pkl data/sp-random-predicate/problems.pkl data/sp-random-function/problems.pkl --output data/sp-random-aggregated --columns_common n_total predicates_count functions_count --columns_individual n_exit_0 "[saturation_iterations, variation]" --filter n_exit_0 >= 12 --filter predicates_count <= 1024 --filter functions_count <= 1024 --count 100 --sort_column "[saturation_iterations, variation]"
```

Filters:

- n_exit_0 >= 12
- predicates_count <= 1024
- functions_count <= 1024

From each dataset,
we fetch top 100 problems
according to `(saturation_iterations, variation)`
that satisfy the filters.

## predicate-1.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-predicate/batches/404131/aggregate/problems.pkl
```

Command:

```bash
join_problems.py data/sp-random-predicate/problems.pkl --output data/sp-random-predicate-train --columns_common n_total predicates_count functions_count --columns_individual n_exit_0 n_exit_1 "[saturation_iterations, variation]" --filter n_exit_0 >= 1 --filter n_exit_1 >= 1
```

Filters:

- n_exit_0 >= 1
- n_exit_1 >= 1

## predicate-7.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-predicate/batches/404131/aggregate/problems.pkl
```

Command:

```bash
join_problems.py data/sp-random-predicate/problems.pkl --output data/sp-random-predicate-train --columns_common n_total predicates_count functions_count --columns_individual n_exit_0 n_exit_1 "[saturation_iterations, variation]" --filter n_exit_0 >= 7 --filter n_exit_1 >= 7
```

Filters:

- n_exit_0 >= 7
- n_exit_1 >= 7

## predicate-easy-variation.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-predicate/batches/404131/aggregate/problems.pkl
```

Command:

```bash
join_problems.py data/sp-random-predicate/problems.pkl --output data/sp-random-predicate-train --columns_common n_total predicates_count functions_count clauses_count --columns_individual n_exit_0 n_exit_1 "[saturation_iterations, variation]" --filter n_exit_0 >= 12 --filter "[saturation_iterations, variation]" >= 1 --sort_columns "[saturation_iterations, variation]" --sort_order d
```

## function-1.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-function/batches/404133/aggregate/problems.pkl
```

Command:

```bash
join_problems.py data/sp-random-function/problems.pkl --output data/sp-random-function-train --columns_common n_total predicates_count functions_count --columns_individual n_exit_0 n_exit_1 "[saturation_iterations, variation]" --filter n_exit_0 >= 1 --filter n_exit_1 >= 1
```

Filters:

- n_exit_0 >= 1
- n_exit_1 >= 1

## function-easy-variation.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-function/batches/404133/aggregate/problems.pkl
```

Command:

```bash
join_problems.py data/sp-random-function/problems.pkl --output data/sp-random-function-train --columns_common n_total predicates_count functions_count clauses_count --columns_individual n_exit_0 n_exit_1 "[saturation_iterations, variation]" --filter n_exit_0 >= 12 --sort_columns "[saturation_iterations, variation]" --sort_order d --count 100
```

## both-easy-variation.txt

Source data:

```
sftp://bartefil@cluster.ciirc.cvut.cz/home/bartefil/git/vampire-ml/out/sp-random-both/batches/404135/aggregate/problems.pkl
```

Command:

```bash
join_problems.py data/sp-random-both/problems.pkl --output data/sp-random-both-train --columns_common n_total predicates_count functions_count clauses_count --columns_individual n_exit_0 n_exit_1 "[saturation_iterations, variation]" --filter n_exit_0 >= 12 --sort_columns "[saturation_iterations, variation]" --sort_order d --count 1000
```
