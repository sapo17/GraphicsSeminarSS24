# Computer Graphics Seminar Summer Term 2024 at TU Wien

Repository for the CG seminar (S2024).

## Report: Seminar submission
You can find the seminar submission in this repository ([click here](docs/scg_24_submission_edition/Hasbay_DrDelaunay2024.pdf)). The differences from the intermediary report can also be found [here](docs/scg_24_submission_edition/notes_on_scg24_submission.md).

## Report: Extended edition
The extended edition is still in the works and planned to be released at the end of June 2024.

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

### Git submodules

- To retrieve submodules that contain experiments:
  - `git submodule update --init --recursive`