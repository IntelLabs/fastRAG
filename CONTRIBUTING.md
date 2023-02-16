# Contributing to fastRAG

The following document describes the process of contributing and developing extensions to fastRAG.

## Setting up your development environment

Preliminary requirements:

- fastRAG installed in a developement enviroment (via `pip install -e`)
- Python 3.8+
- Pytorch
- Any 3rd party store engine package

## Installing development packages

To install extra packages required for library development:

```sh
pip install -e .[dev]
```

We use [pre-commit](https://pre-commit.com/) to automatically format the code in a consistent way. Installing the dev packages will install `pre-commit`. To manually install it:

```sh
pip install pre-commit
pre-commit install        # in project's dir
```

It uses the `black` and `isort` utilities. No need to download them, they are handled by `pre-commit`. Have a look at [pre-commit-config](./.pre-commit-config.yaml) to see what it does.
To explicitly run the hooks:

```sh
pre-commit run --all-files
```

That's it!

## New component contribution process

1. Fork the repository to your own github space
2. Add your component as an extension or new addition following Haystack components hierarchy and according to the structure of the library (e.g., a new reader to `fastrag/readers/`).
3. Provide a pipeline or script to validate the flow of the new component (a `yaml` file or modifications in `scripts/generate_pipeline.py`).
4. Open a pull request and describe your contribution.
