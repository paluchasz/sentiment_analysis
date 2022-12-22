# Sentiment Analysis

Using transformers to predict the sentiment of movie reviews from the Stanford datataset available here: https://ai.stanford.edu/~amaas/data/sentiment/.
Create a `data` directory and put the `aclImdb` downloaded folder inside it to run.


## Setup

WIP needs filling in once we have decided on how we manage dependencies and run code

We are usng poerty to manage virtual envs and dependency management.

- Install poetry from [here](https://python-poetry.org/docs/)
- From project root run
```
poetry install
```
this will create a virtual env in a `.venv` folder and install all dependencies into it.
- To run command in the venv either open a shell with
```
poetry shell
```
or run from commands from outside the shell with
```
poetry run <command>
```
