
# Sentiment Analysis
![Test Suite](https://github.com/paluchasz/sentiment_analysis/actions/workflows/check_pr.yaml/badge.svg?branch=main)

Using transformers to predict the sentiment of movie reviews from the Stanford dataset available [here](https://ai.stanford.edu/~amaas/data/sentiment/).


## Setup

We use poetry to manage the virtual env and dependency management.

- Install poetry from [here](https://python-poetry.org/docs/)
- From project root run
```
poetry install
```
this will create a virtual env in a `.venv` folder and install all dependencies into it.
- To activate a shell in the venv run
```
poetry shell
```

## Training scripts
To run the scripts the sample_data and sample env file can be used. For the full dataset download from
[here](https://ai.stanford.edu/~amaas/data/sentiment/).

The 3 scripts are defined as executables in the `pyproject.toml` file and if this repository is installed
as a package with either `poetry install` or `pip install .` the scripts can be called with simply: `get_train_test_data`,
`train` and `predict_and_evaluate`.
