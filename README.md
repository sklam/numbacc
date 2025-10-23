# README



## SETUP

Prerequiste

```
conda install conda-project
```

Setup environment:

```
conda project prepare -n dev
make setup-workspace
conda project run build
```

Activation

```
conda activate ./envs/dev
```

Testing

```
pytest ./nbcc
```
