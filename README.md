# CG Seminar Summer Term 2024

Repository for the CG seminar (S2024).


## Anaconda instructions

### Create Environment
```
conda env create --prefix ./gs_venv -f gs_venv.yml
```

### Activate Environment
```
conda activate ./gs_venv
```

### Export Environment
```
conda env export --from-history
```

### Update Environment
```
conda env update --prefix ./gs_venv --file gs_venv.yml  --prune
```