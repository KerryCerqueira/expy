import abc
from types import ModuleType
from serde import serde, field, Untagged
from serde.json import to_json, from_json
from pathlib import Path
from typing import Any, Tuple, ClassVar, Optional, Iterator, Union
from collections.abc import Callable
import importlib
import importlib.util
from llama_cpp import CreateCompletionResponse, Llama
from git import Repo, Commit, Blob
import fnmatch
import glob
from hashlib import sha256
import nbclient
import nbformat


@serde
class DataSpec:
    data_commit: Optional[str] = field(kw_only=True, default=None)
    data_paths: list[str] = field(kw_only=True)
    data_repo_path: Optional[Path] = field(kw_only=True, default=None)

    def __post_init__(self):
        if self.data_repo_path is not None and self.data_commit is None:
            self.data_commit = "HEAD"
        if self.data_repo_path is None and self.data_commit is not None:
            raise ValueError(
                "Must specify a data repository when specifying a commit"
            )

    def get_repo(self) -> Optional[Repo]:
        if self.data_repo_path is None:
            return None
        else:
            return Repo(self.data_repo_path)

    def has_repo(self) -> bool:
        return self.get_repo() is not None

    def get_commit(self) -> Optional[Commit]:
        repo = self.get_repo()
        if self.data_commit is None or repo is None:
            return None
        else:
            return repo.commit(self.data_commit)

    def has_commit(self) -> bool:
        return self.get_commit is not None

    def expand_paths(self) -> list[str]:
        # use of extend is readable and not meant to be performant
        commit = self.get_commit()
        if commit is None:
            expanded_paths = []
            for ptrn in self.data_paths:
                expanded_paths.extend(glob.glob(ptrn))
            return expanded_paths
        else:
            blob_paths = [
                obj.path
                for obj in commit.tree.traverse()
                if isinstance(obj, Blob)
            ]
            expanded_paths = []
            for ptrn in self.data_paths:
                expanded_paths.extend(fnmatch.filter(str(blob_paths), ptrn))
            return expanded_paths

    def get_data_iter(self) -> Iterator[Tuple[str, str]]:
        commit = self.get_commit()
        if commit is None:
            for path in self.expand_paths():
                with open(path, "r", encoding="utf-8") as file:
                    data = file.read()
                yield (path, data)
        else:
            for path in self.expand_paths():
                data = (
                    commit
                    .tree[path]
                    .read()
                    .decode("utf-8")
                )
                yield (path, data)


@serde
class InputPipeSpec(abc.ABC):
    kwargs: Optional[dict[str, Any]] = field(default=None, kw_only=True)
    _pipeline_fn: Callable[..., str] = field(
        skip=True,
        init=False,
        repr=False
    )

    def __call__(self, data: str) -> str:
        return self._pipeline_fn(data, **(self.kwargs or {}))


@serde
class LibInputPipeSpec(InputPipeSpec):
    lib_fn: str = field(kw_only=True)
    _input_lib: ClassVar[ModuleType] = importlib.import_module(
        ".pipelines.input",
        package=__package__
    )

    def __post_init__(self) -> None:
        # TODO: Add exception handling
        self._pipeline_fn = getattr(
            LibInputPipeSpec._input_lib,
            self.lib_fn
        )


@serde
class CustomInputPipeSpec(InputPipeSpec):
    module: Path = field(kw_only=True)
    fn_name: str = field(kw_only=True, default="input_pipeline")

    def __post_init__(self) -> None:
        # TODO: Add exception handling
        spec = importlib.util.spec_from_file_location(
            self.module.stem, self.module
        )
        assert spec is not None and spec.loader is not None, \
            f"{self.module} not found"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._pipeline_fn = getattr(module, self.fn_name)


@serde
class ModelSpec(abc.ABC):
    kwargs: Optional[dict[str, Any]] = field(kw_only=True, default=None)
    _pipeline_fn: Optional[Llama] = field(
        skip=True,
        init=False,
        repr=False,
        default=None,
    )

    @abc.abstractmethod
    def _init_model(self) -> None:
        pass

    def __call__(self, data: str, **kwargs) -> CreateCompletionResponse:
        # TODO: Redirect llama output to log file
        if self._pipeline_fn is None:
            self._init_model()
        assert self._pipeline_fn is not None
        response = self._pipeline_fn(data, **kwargs)
        if isinstance(response, Iterator):
            raise ValueError("Streaming responses not supported")
        else:
            return response


@serde
class LocalModelSpec(ModelSpec):
    model_path: Path = field(kw_only=True)

    def _init_model(self) -> None:
        # TODO: Add exception handling, precheck model spec
        if self._pipeline_fn is None:
            self._pipeline_fn = Llama(
                str(self.model_path),
                **(self.kwargs or {})
            )
        else:
            raise ValueError("Model already initialized")


@serde
class HFModelSpec(ModelSpec):
    repo_id: str = field(kw_only=True)
    filename: str = field(kw_only=True)

    def _init_model(self) -> None:
        # TODO: Add exception handling, precheck model spec
        if self._pipeline_fn is not None:
            self._pipeline_fn = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                **(self.kwargs or {})
            )
        else:
            raise ValueError("Model already initialized")


@serde
class LibOutputPipeSpec:
    lib_fn: str = field(kw_only=True)
    kwargs: Optional[dict[str, Any]] = field(kw_only=True, default=None)
    _pipeline_fn: Callable[..., str] = field(
        skip=True,
        init=False,
        repr=False
    )
    # TODO: Couldn't we import this with a standard non-dynamic mechanism?
    _output_lib: ClassVar[ModuleType] = importlib.import_module(
        ".pipelines.output",
        package=__package__
    )

    def __post_init__(self) -> None:
        self._pipeline_fn = getattr(
            LibInputPipeSpec._input_lib,
            self.lib_fn
        )

    def __call__(self, data: dict[str, Any]) -> str:
        return self._pipeline_fn(data, **(self.kwargs or {}))


@serde
class NotebookPipeSpec:
    notebook_paths: list[Path] = field(kw_only=True)
    kernel: Optional[str] = field(kw_only=True, default=None)

    def __call__(self) -> None:
        for nb_path in self.notebook_paths:
            nb = nbformat.read(nb_path, as_version=4)
            nbclient.execute(
                nb,
                cwd="./",
                kernel_name=self.kernel or None,
            )
            nbformat.write(nb, nb_path.name)


@serde(tagging=Untagged)
class Experiment:
    dataset: DataSpec = field(kw_only=True, flatten=True)
    pre_inference_pipeline: (
       LibInputPipeSpec | CustomInputPipeSpec
    ) = field(
        kw_only=True,
        default_factory=lambda: LibInputPipeSpec(lib_fn="id")
    )
    inference_pipeline: LocalModelSpec | HFModelSpec = field(kw_only=True)
    post_inference_pipeline: (
        list[Union[LibOutputPipeSpec, NotebookPipeSpec]]
    ) = field(kw_only=True, default_factory=list)

    @staticmethod
    def from_json(json_str: str) -> "Experiment":
        return from_json(Experiment, json_str)

    def get_hash(self) -> str:
        # TODO: Add input/output module to hash input
        return sha256(to_json(self).encode("utf-8")).hexdigest()

    # def get_output_paths(self) -> list[Path]:
    #     output_paths = [
    #         Path("inferences.json"),
    #         Path("log.txt"),
    #         Path("experiment.json")
    #     ]
    #     if (
    #         self.post_inference_pipeline is not None
    #         and self.post_inference_pipeline.notebook_paths is not None
    #     ):
    #         output_paths.extend([
    #             Path(path.name)
    #             for path in self.post_inference_pipeline.notebook_paths
    #         ])
    #     return output_paths

    def _run_post_inference(self):
        pass

    def run(self) -> None:
        # TODO: Add exception when output dir files exist
        # TODO: Add logging
        data_iter = self.dataset.get_data_iter()
        output = {}
        for path, data in data_iter:
            output[path] = self.inference_pipeline(
                self.pre_inference_pipeline(data)
            )
        with open("inferences.json", "w") as inferences_file:
            inferences_file.write(to_json(output))
