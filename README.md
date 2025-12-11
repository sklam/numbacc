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

Note: 
- subsequent update to `dev` environment can run `make build` in the environment


Activation

```
conda activate ./envs/dev
```

Testing

```
pytest ./nbcc
```



## E2E compiler demo

```
python -m nbcc.compile <source> <dest-binary>
```
