conditioning-of-musical-generative-models
==============================

Required to initialize project:

- `python3`(3.9;3.10), `pipx`, `just`, `poetry`
- filling .env file

Clone repo:

```shell
git clone --recurse-submodules -j8 git@github.com:MikolajSzawerda/musical-generative-models-conditioning.git
```

Available tools:

```shell
just -l
```

### Repository structure

---------------------------------

    ├── data
    │   ├── input         <- data to digest by models
    │   ├── output        <- data retrieved from models
    │   └── raw           <- data to be processed
    ├── docs                    <- text files with knowledge handy to have in repo
    │   └── autoresearch  <- papers autmaticaly scraped
    ├── logs                    <- logs from trainings
    ├── configs                 <- yaml configs for training
    ├── models                  <- saved artefacts after succesfull training
    ├── visualization           <- aggregated results from experiments in the form of notebooks
    ├── scripts                 <- handy shell scripts
    ├── src                     <- experiments/tooling
    │   ├── app           <- gradio app to have fun with textual inversion
    │   ├── audiocraft    <- experiments with music gen
    │   ├── paper-query   <- auto arxiv scraper
    │   └── tools         <- tools for easier life in the project

---------------------------------

###  






