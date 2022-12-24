
# Sentiment Analysis
![Test Suite](https://github.com/paluchasz/sentiment_analysis/actions/workflows/check_pr.yaml/badge.svg?branch=main)
![Interrogate](./docs/_static/interrogate_badge.svg)

Using transformers to predict the sentiment of movie reviews from the Stanford datataset available here: https://ai.stanford.edu/~amaas/data/sentiment/.
Create a `data` directory and put the `aclImdb` downloaded folder inside it to run.


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
