"""A module defining an Experiment abstraction for expy."""
import abc
import fnmatch
import glob
import importlib
import importlib.util
import json
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Self

import nbclient
import nbformat
from git import Blob, Commit, Repo
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class DataSpec(BaseModel):
    """A specification of input data for an experiment.

    :param data_commit: The commit containing the input data.
    :param data_paths: Glob patterns specifying paths to the input data.
    If ``data_repo_path`` is specified the paths are expanded relative
    to the repository tree; Otherwise, they are understood to be
    relative to the working directory.
    :param data_repo_path: A path to the data repository.
    """

    data_commit: str | None = Field(kw_only=True, default=None)
    data_paths: list[str] = Field(kw_only=True)
    data_repo_path: Path | None = Field(kw_only=True, default=None)


    @model_validator(mode= "after")
    def foo(self) -> Self:
        """Finish initializing this ``DataSpec``.

        :raises ValueError: When a data repository is not specified but
        a commit is.
        """
        # TODO: Add check to guarantee if data_paths begin with / (i.e. are
        # absolute) the repo can't be spec'd
        if self.data_repo_path is not None and self.data_commit is None:
            self.data_commit = "HEAD"
        if self.data_repo_path is None and self.data_commit is not None:
            raise ValueError(
                "Must specify a data repository when specifying a commit"
            )
        return self

    def get_repo(self) -> Repo | None:
        """Get the repository specified by this ``DataSpec``.

        :return: A ``GitPython`` ``Repo`` object, or ``None`` if none is
        found.
        """
        if self.data_repo_path is None:
            return None
        return Repo(self.data_repo_path)

    def has_repo(self) -> bool:
        """Whether a valid repository exists for this ``DataSpec``.

        :return: ``True`` if a git repository is found at
        ``self.data_repo_path``.
        """
        return self.get_repo() is not None

    def get_commit(self) -> Commit | None:
        """Get the commit specified by this ``DataSpec``.

        :return: A ``GitPython`` ``commit`` object, or ``None`` if one
        is not found.
        """
        repo = self.get_repo()
        if self.data_commit is None or repo is None:
            return None
        return repo.commit(self.data_commit)

    def has_commit(self) -> bool:
        """Whether a valid commit exists for this ``DataSpec``.

        :return: ``True`` if a repository is found with a commit
        matching ``self.data_commit``.
        """
        return self.get_commit is not None

    def get_matching_paths(self) -> list[str]:
        """Expand globs in ``self.data_paths``.

        Patterns are expanded to matching paths in the input data.
        :return: A list of paths matching patterns in
        ``self.data_paths``. If these are not absolute paths, they are
        resolved relative to the working directory if
        ``self.has_repo()`` is ``False``, or otherwise relative to the
        repository tree.
        """
        # use of extend is readable and not meant to be performant
        commit = self.get_commit()
        if commit is None:
            expanded_paths = []
            for ptrn in self.data_paths:
                expanded_paths.extend(glob.glob(ptrn))
            return expanded_paths
        blob_paths = [
            str(obj.path)
            for obj in commit.tree.traverse()
            if isinstance(obj, Blob)
        ]
        expanded_paths = []
        for ptrn in self.data_paths:
            expanded_paths.extend(fnmatch.filter(blob_paths, ptrn))
        return expanded_paths

    def get_data_iter(self) -> Iterator[tuple[str, str]]:
        """Get an iterator over the data specified by this ``DataSpec``.

        :return: An iterator yielding tuples ``(path, data)``, where
        ``path`` is expanded path the text ``data`` was found at. The
        path is understood to be either absolute, relative to the data
        repository, or relative to the current working directory
        depending on whether a data repository was specified or not.
        """
        commit = self.get_commit()
        if commit is None:
            for path in self.get_matching_paths():
                with open(path, encoding="utf-8") as file:
                    data = file.read()
                yield (path, data)
        else:
            for path in self.get_matching_paths():
                data = commit.tree[path].data_stream.read().decode("utf-8")
                yield (path, data)


class InputPipeSpec(abc.ABC, BaseModel):
    """An abstract specification for a pre-inference data pipeline.

    :param kwargs: Keyword arguments passed to the pipeline function.
    :param _pipeline_fn: The function to be called.
    """

    kwargs: dict[str, Any] | None = Field(default=None, kw_only=True)
    _pipeline_fn: Callable[..., MutableMapping[str, Any]] = PrivateAttr()

    def __call__(self, data: str) -> MutableMapping[str, Any]:
        """Evaluate the data pipeline on a single piece of data.

        :param data: Text input for the pipeline.
        :return: The output of the pipeline.
        """
        return self._pipeline_fn(data, **(self.kwargs or {}))


class LibInputPipeSpec(InputPipeSpec):
    """Spec for a pre-inference pipeline based on a library function.

    When called, evaluates a library function in
    ``expy.pipelines.input`` given by ``self.lib_fn``.

    :param lib_fn: The name of the library function.
    :param _input_lib: The function resolved from the library from
    ``self.lib_fn`` after initialization.
    """

    lib_fn: str = Field(kw_only=True)
    _input_lib: ClassVar[ModuleType] = importlib.import_module(
        ".pipelines.input", package=__package__
    )

    @model_validator(mode="after")
    def bind_pipeline_fn(self) -> Self:
        self._pipeline_fn = getattr(LibInputPipeSpec._input_lib, self.lib_fn)
        return self


class CustomInputPipeSpec(InputPipeSpec):
    """Spec for a pre-inference pipeline based on a given python module.

    :param module: A path to a python module.
    :param fn_name: The name of the function which will be called when
    this input pipeline is evaluated.
    """

    module: Path = Field(kw_only=True)
    fn_name: str = Field(kw_only=True, default="input_pipeline")

    @model_validator(mode="after")
    def bind_pipeline_fn(self) -> Self:
        """Load `self.module`, bind `self.fn_name`, and return `self`."""
        try:
            spec = importlib.util.spec_from_file_location(
                self.module.stem, self.module
            )
            if spec is None or spec.loader is None:
                raise FileNotFoundError
            module: ModuleType = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.module}"
            ) from None
        except SyntaxError:
            raise ImportError(
                f"{self.module} contains invalid Python"
            ) from None
        try:
            self._pipeline_fn = getattr(module, self.fn_name)
        except AttributeError:
            raise AttributeError(
                f"`{self.fn_name}` not found in module {self.module}"
            ) from None
        return self


class CustomGraphSpec(BaseModel):
    module: Path = Field(kw_only=True, default=Path("exp.py"))
    fn_name: str = Field(kw_only=True, default="inference_pipeline")
    kwargs: dict[str, Any] | None = Field(kw_only=True, default=None)
    _state_graph: CompiledStateGraph = PrivateAttr()

    @model_validator(mode="after")
    def compile_graph(self) -> Self:
        """Load `self.module`, bind `self.fn_name`, and return `self`."""
        spec = importlib.util.spec_from_file_location(
            self.module.stem, self.module
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._state_graph = getattr(module, self.fn_name)(
            **(self.kwargs or {}),
        ).compile()
        return self

    def __call__(
        self, state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        return self._state_graph.invoke(state)

class LibOutputPipeSpec(BaseModel):
    """Spec for a post-inference pipeline based on a library function.

    :param lib_fn: The name of the library function.
    :param kwargs: keyword arguments to be passed.
    :param _pipeline_fn: The function to be called.
    """

    lib_fn: str = Field(kw_only=True)
    kwargs: dict[str, Any] | None = Field(kw_only=True, default=None)
    _pipeline_fn: Callable[..., None] = PrivateAttr()
    # TODO: Couldn't we import this with a standard non-dynamic mechanism?
    _output_lib: ClassVar[ModuleType] = importlib.import_module(
        ".pipelines.output", package=__package__
    )

    @model_validator(mode="after")
    def bind_pipeline_fn(self) -> Self:
        self._pipeline_fn = getattr(
            LibOutputPipeSpec._output_lib,
            self.lib_fn
        )
        return self

    def run(self) -> None:
        """Run the pipeline by evaluating the specified function."""
        return self._pipeline_fn(**(self.kwargs or {}))


class NotebookPipeSpec(BaseModel):
    """Spec for a post-inference pipeline based on a jupyter notebook.

    :param notebook_paths: Paths to jupyter notebooks.
    :param kernel: The name of a jupyter kernel to run the notebooks
    with.
    """

    #TODO: This should return notebook objects, not write them to disk.
    notebook_paths: list[Path] = Field(kw_only=True)
    kernel: str | None = Field(kw_only=True, default=None)

    def run(self) -> None:
        """Evaluate this notebook pipeline.

        Runs each notebook in the order it was given and writes the
        results to the working directory.
        """
        # TODO: Detect path collisions.
        for nb_path in self.notebook_paths:
            nb = nbformat.read(nb_path, as_version=4)
            nbclient.execute(nb, kernel_name=self.kernel)
            nbformat.write(nb, nb_path.name)


class Experiment(BaseModel):
    """A specification for an AI experiment.

    At a high level, an ``Experiment`` consists of specifications for
    data together with pipelines for pre-inference, inference, and
    post-inference computations. Each of these can be optionally
    specified to guarantee reproducibility of the experiment via hashes
    of the inputs. These are encoded by a json of a prescribed schema
    which an experiment can be deserialized from.
    """

    dataset: DataSpec = Field(kw_only=True)
    pre_inference_pipeline: LibInputPipeSpec | CustomInputPipeSpec = Field(
        kw_only=True,
        default_factory=lambda: CustomInputPipeSpec(
            module=Path("exp.py"),
            fn_name="pre_inference_pipeline",
        ),
    )
    inference_pipeline: CustomGraphSpec = Field(
        kw_only=True,
        default_factory=lambda: CustomGraphSpec(
            module=Path("exp.py"),
            fn_name="inference_pipeline",
        ),
    )
    post_inference_pipeline: list[LibOutputPipeSpec | NotebookPipeSpec] = (
        Field(kw_only=True, default_factory=list)
    )

    def run(self) -> None:
        """Run the experiment."""
        # TODO: Add exception when output dir files exist
        # TODO: Add logging
        # TODO: Add input to run in given directory
        output = {}
        for path, data in self.dataset.get_data_iter():
            output[path] = self.inference_pipeline(
                self.pre_inference_pipeline(data)
            )
            with open("inferences.json", "w") as inferences_file:
                inferences_file.write(json.dumps(output))
                for pipe in self.post_inference_pipeline:
                    pipe.run()
