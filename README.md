conditioning-of-musical-generative-models
==============================

Required to initialize project:

- `python3`(3.9;3.12), `pipx`, `just`
- filling .env file

Available tools:
```shell
just -l
```

### Resources

As for now I keep heavy files in nextcloud(from mikrus vps :D)

```shell
just pull data/ #with filled .env this will fetch all data
```

---------------------------------

    ├── data
    │   ├── input         <- data to digest by models
    │   ├── output        <- data retrieved from models
    │   └── raw           <- data to be processed
    ├── docs                    <- text files with knowledge handy to have in repo
    │   └── autoresearch  <- papers autmaticaly scraped
    ├── justfile                <- source of handy automation of project
    ├── logs                    <- logs from trainings
    ├── models                  <- saved artefacts after succesfull training
    ├── notebooks               <- aggregated results from experiments in the form of notebooks
    ├── scripts                 <- handy shell scripts
    ├── src                     <- experiments/tooling
    │   ├── audiocraft    <- experiments with music gen
    │   ├── paper-query   <- auto arxiv scraper
    │   └── tools         <- tools for easier life
---------------------------------






