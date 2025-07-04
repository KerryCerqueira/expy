from typing import Annotated
import typer
from .experiment import Experiment
from pathlib import Path

expy = typer.Typer()


@expy.command()
def run(
	path: Annotated[Path, typer.Argument(default_factory=Path.cwd)],
) -> None:
    path = path / "exp.json" if path.is_dir() else path
    with open(path) as f:
        exp = Experiment.model_validate_json(f.read())
    exp.run()


if __name__ == "__main__":
    expy()
