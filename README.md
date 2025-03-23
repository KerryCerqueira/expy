# `expy`

`expy` is a minimal, lightweight framework for running AI experiments. An
`expy` pipeline consists of a data repository from which data is retrieved,
together with pre-processing, inference, and post-processing steps. These are
specified in a json file which deserializes to an `Experiment` object which is
exposed to the user. Alternatively, a CLI is provided for running experiments
on the command line.

## Installation

`expy` relies on `llama-cpp-python` to perform LLM inference. Depending on the
type of hardware acceleration desired, installation of this backend requires
some care. See the [readme](https://github.com/abetlen/llama-cpp-python) for
clarification.

`expy` is a pure python package and unpublished on e.g. pyPL. The most
straightforward way to use this package is to install a local copy with `pip`.

First obtain a local copy,

```bash git clone https://www.github.com/KerryCerqueira/expy expy/ ```

and then after switching to whatever flavour of python virtual environment you
prefer, install the local package with pip.

```bash pip install ./expy ```

## Usage

### A minimal example

Say you have directory containing some textual data you want to perform
inference on.

```text
experiment/
└── /data
    ├── foo.txt
    └── bar.txt
```

We'll need a lightweight model to test the inference. For example, we could
snag one by installing `huggingface-hub` and running the following:

```bash
huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q2_K.gguf \
--local-dir ./models --local-dir-use-symlinks False
```

Now we can specify a simple experiment in `experiment.json`,

```json
{
	"data_paths": ["data/*.txt"],
	"inference_pipeline": { "model_path": "models/phi-2.Q2_K.gguf" }
}
```

and then run the experiment. We could either use the CLI,

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

### A less minimal example

Let's say you've decided to version control your data in a git repository, and
you have a jupyter notebook to post-process and present the results:

```text
experiment/
├── /data
│   ├── .git/
│   ├── foo.txt
│   └── bar.txt
├────── notebook.ipynb
└────── experiment.json
```

We can write an experiment specification that retrieves the text from the git
repository, applies a pre-processing function, and then runs the notebook in
the output directory.

```json
{
    "data_repo_path": "data",
    "data_commit": "HEAD",
    "data_paths": [ "*.txt" ],
    "pre_inference_pipeline": { "lib_fn": "id" },
    "inference_pipeline": { "model_path": "models/phi-2.Q2_K.gguf" },
    "post_inference_pipeline": [
        { "lib_fn": "hello_world" },
        { "notebook_paths": ["notebook.ipynb"], "kernel": "python3" }
    ]
}
```

By specifying a commit for the inputs, a degree of reproducibility can be
guaranteed for the experiment. `expy` takes care of retrieving the relevant
files matching any passed patterns from the underlying git store without having
to check anything out.
