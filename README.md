# `expy`

`expy` is a minimal, lightweight framework for running AI experiments. An `expy` pipeline consists of a data repository from which data is retrieved, together with pre-processing, inference, and post-processing steps. These are specified in a json file which deserializes to an `Experiment` object which is exposed to the user. Alternatively, a CLI is provided for running experiments on the command line.

## Installation

`expy` relies on `llama-cpp-python` to perform LLM inference. Depending on the type
of hardware acceleration desired, installation of this backend requires some
care. See the [readme](https://github.com/abetlen/llama-cpp-python) for
clarification.

`expy` is a pure python package and unpublished on e.g. pyPL. The most
straightforward way to use this package is to install a local copy with `pip`.

First obtain a local copy,
```bash
git clone https://www.github.com/KerryCerqueira/expy expy/
```
and then after switching to whatever flavour of python virtual environment you prefer, install the local package with pip.
```bash
pip install ./expy 
```

## Usage

Say you have a git repository containing some textual data you want to perform
inference on, whose `HEAD`/`INDEX` looks like:

```text
experiment/
└── /data
    ├── .git/
    ├── foo.txt
    └── bar.txt
```

We'll need a lightweight model to test the inference. For example, we could snag one by installing `huggingface-hub` and running the following:

```bash
huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q2_K.gguf --local-dir ./models --local-dir-use-symlinks False
```

Now we can specify a simple experiment in `experiment.json`:

```json
{
    "dataset_commit": "HEAD",
    "data_paths": [
        "*.txt"
    ],
    "data_repo": "data",
    "pre_inference_pipeline": { "lib_fn": "id" },
    "inference_pipeline": { "model_path": "models/phi-2.Q2_K.gguf" }
}
```

Now we can run the experiment. We could either use the CLI,

```bash
expy run experiment.json
```

or within a python session within the same directory,

```python
from expy import Experiment

exp = Experiment.from_json("experiment.json")
exp.run()
```

In either case the results will be dumped into the working directory.
