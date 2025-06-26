"""A module defining an Experiment abstraction for expy."""
import abc
import fnmatch
import glob
import importlib
import importlib.util
from collections.abc import Callable, Iterator

# from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import nbclient
import nbformat
from git import Blob, Commit, Repo
from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionResponse,
    CreateCompletionResponse,
    Llama,
)
from pydantic import BaseModel, Field, PrivateAttr


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

    def __post_init__(self):
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

    def _get_repo(self) -> Repo | None:
        """Get the repository specified by this ``DataSpec``.

        :return: A ``GitPython`` ``Repo`` object, or ``None`` if none is
        found.
        """
        if self.data_repo_path is None:
            return None
        else:
            return Repo(self.data_repo_path)

    def has_repo(self) -> bool:
        """Whether a valid repository exists for this ``DataSpec``.

        :return: ``True`` if a git repository is found at
        ``self.data_repo_path``.
        """
        return self._get_repo() is not None

    def _get_commit(self) -> Commit | None:
        """Get the commit specified by this ``DataSpec``.

        :return: A ``GitPython`` ``commit`` object, or ``None`` if one
        is not found.
        """
        repo = self._get_repo()
        if self.data_commit is None or repo is None:
            return None
        else:
            return repo.commit(self.data_commit)

    def _has_commit(self) -> bool:
        """Whether a valid commit exists for this ``DataSpec``.

        :return: ``True`` if a repository is found with a commit
        matching ``self.data_commit``.
        """
        return self._get_commit is not None

    def _expand_paths(self) -> list[str]:
        """Expand globs in ``self.data_paths``.

        Patterns are expanded to matching paths in the input data.
        :return: A list of paths matching patterns in
        ``self.data_paths``. If these are not absolute paths, they are
        resolved relative to the working directory if
        ``self.has_repo()`` is ``False``, or otherwise relative to the
        repository tree.
        """
		# use of extend is readable and not meant to be performant
        commit = self._get_commit()
        if commit is None:
            expanded_paths = []
            for ptrn in self.data_paths:
                expanded_paths.extend(glob.glob(ptrn))
            return expanded_paths
        else:
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
        commit = self._get_commit()
        if commit is None:
            for path in self._expand_paths():
                with open(path, encoding="utf-8") as file:
                    data = file.read()
                yield (path, data)
        else:
            for path in self._expand_paths():
                data = commit.tree[path].data_stream.read().decode("utf-8")
                yield (path, data)


class InputPipeSpec(abc.ABC, BaseModel):
	"""An abstract specification for a pre-inference data pipeline.

	:param kwargs: Keyword arguments passed to the pipeline function.
	:param _pipeline_fn: The function to be called.
	"""

	kwargs: dict[str, Any] | None = Field(default=None, kw_only=True)
	_pipeline_fn: (
		Callable[..., str] | Callable[..., list[ChatCompletionRequestMessage]]
	) = PrivateAttr()

	@abc.abstractmethod
	def __post_init__(self) -> None:
		"""Finishes initializing this input pipeline."""
		...

	def __call__(self, data: str) -> str | list[ChatCompletionRequestMessage]:
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

    def __post_init__(self) -> None:
        """Finishes initializing this input pipeline."""
        # TODO: Add exception handling
        self._pipeline_fn = getattr(LibInputPipeSpec._input_lib, self.lib_fn)


class CustomInputPipeSpec(InputPipeSpec):
    """Spec for a pre-inference pipeline based on a given python module.

    :param module: A path to a python module.
    :param fn_name: The name of the function which will be called when
    this input pipeline is evaluated.
    """

    module: Path = Field(kw_only=True)
    fn_name: str = Field(kw_only=True, default="input_pipeline")

    def __post_init__(self) -> None:
        """Finishes initializing this input pipeline."""
        # TODO: Add exception handling
        spec = importlib.util.spec_from_file_location(
            self.module.stem, self.module
        )
        assert (
            spec is not None and spec.loader is not None
        ), f"{self.module} not found"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._pipeline_fn = getattr(module, self.fn_name)

class ModelSpec(abc.ABC, BaseModel):
    """An abstract specification for a model inference pipeline.

    :param kwargs: keyword arguments passed along to the pipeline.
    :param _pipeline_fn: A ``llama_cpp_python`` ``Llama`` object
    representing the model, or ``None`` if it hasn't been
    initialized.
    """

    kwargs: dict[str, Any] | None = Field(kw_only=True, default=None)
    _model: Llama | None = PrivateAttr()

    @abc.abstractmethod
    def _init_model(self) -> None: ...

    def __call__(
        self,
        data: str | list[ChatCompletionRequestMessage],
    ) -> CreateCompletionResponse | CreateChatCompletionResponse:
        """Evaluate the inference pipeline on a single input.

        :param data: The input text to provide inference on.
        :raises ValueError: If keyword arguments are passed that produce
        a streaming response.
        :return: The result of the model inference.
        """
        # TODO: Redirect llama output to log file
        # TODO: Intercept keyword arguments early that produce a
        # streaming response, either fail or change/warn
        if self._model is None:
            self._init_model()
        assert self._model is not None
        if isinstance(data, str):
            response = self._model(data)
        else:
            response = self._model.create_chat_completion(data)
        if isinstance(response, Iterator):
            raise ValueError("Streaming responses not supported")
        else:
            return response


class LocalModelSpec(ModelSpec):
    """A spec for a pipeline based on a local model.

    :param model_path: A path to a model to be passed loaded with
    ``llama_cpp_python``.
    """

    model_path: Path = Field(kw_only=True)

    def _init_model(self) -> None:
        """Initialize inference pipeline.

        This should generally be called lazily after initialization.

        :raises ValueError: If the model has already been initialized.
        """
        # TODO: Add exception handling, precheck model spec
        if self._model is None:
            self._model = Llama(
                str(self.model_path), **(self.kwargs or {})
            )
        else:
            raise ValueError("Model already initialized")


class HFModelSpec(ModelSpec):
    """A spec for a pipeline based on a huggingface model.

    :param repo_id: A HFHub repository.
    :param filename: The filename of the model as stored in the
    repository.
    """

    repo_id: str = Field(kw_only=True)
    filename: str = Field(kw_only=True)

    def _init_model(self) -> None:
        # TODO: Add exception handling, precheck model spec
        if self._model is not None:
            self._model = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                **(self.kwargs or {}),
            )
        else:
            raise ValueError("Model already initialized")


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

    def __post_init__(self) -> None:
        """Finish initializing this ``LibOutputPipeSpec``."""
        self._pipeline_fn = getattr(LibOutputPipeSpec._output_lib, self.lib_fn)

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
        kw_only=True, default_factory=lambda: LibInputPipeSpec(lib_fn="id")
    )
    inference_pipeline: LocalModelSpec | HFModelSpec = Field(kw_only=True)
    post_inference_pipeline: list[
        LibOutputPipeSpec | NotebookPipeSpec
    ] = Field(kw_only=True, default_factory=list)

    # @staticmethod
    # def from_json(json_str: str) -> "Experiment":
    #     """Initialize an ``Experiment`` from a json string.
    #
    #     :param json_str: A json string to be deserialized.
    #     :return: The ``Experiment`` encoded by the input json.
    #     """
    #     return from_json(Experiment, json_str)

    # def run(self) -> None:
    #     """Run the experiment."""
    #     # TODO: Add exception when output dir files exist
    #     # TODO: Add logging
    #     # TODO: Add post inference computation
    #     # TODO: Add input to run in given directory
    #     data_iter = self.dataset.get_data_iter()
    #     output = {}
    #     for path, data in data_iter:
    #         output[path] = self.inference_pipeline(
    #             self.pre_inference_pipeline(data)
    #         )
    #     with open("inferences.json", "w") as inferences_file:
    #         inferences_file.write(to_json(output))
    #     for pipe in self.post_inference_pipeline:
    #         pipe.run()
