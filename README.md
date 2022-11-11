# Porto Seguro
Build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year.

## Setup
Download data from the [kaggle competiton](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data) and save in `/data`.

## Criteria
- Understanding of the business problem to be solved
- Explanation for decisions made in formulating your solution
- Code quality
- Packaging of solution
- Evaluation of solution vs. baseline

### Installation
Need some version of `Python3.9` in `$PythonPath` then use pipenv to install dependencies:

```
    $PythonPath\python.exe -m pip install pipenv
    set PIPENV_VENV_IN_PROJECT="enabled"
    pipenv install -d
    pipenv shell
    cd src
```


### Conda environment
To use a conda environment and make it discoverable by Jupyter (EDA notebook or if conda preferred over pipenv) run the following:
```
    conda env update -f env.yml
    conda activate ami
    python -m ipykernel install --user --name ami --display-name "AMI (python 3.9)"
```


### Unit\Integration Tests
Run unit tests locally
```
    python -m pytest tests/unit
```


## Model Pipeline
Scripts to be ran in sequence:
1. `process.py` - process data for model development
1. `baseline.py` - generate baseline model from example kernel
1. `score.py` - score model vs. baseline and naive estimator

```
    pipenv run src/process.py
    pipenv run src/baseline.py
    pipenv run src/score.py
```