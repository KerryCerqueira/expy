import os
from pathlib import Path
from typing import Annotated

import typer

from .experiment import Experiment

expy = typer.Typer()


@expy.command()
def run(
	path: Annotated[Path, typer.Argument(default_factory=Path.cwd)],
) -> None:
    exp_path = path.parent if path.is_file() else path
    os.chdir(exp_path)
    with Path.open(Path("exp.json")) as f:
        exp = Experiment.model_validate_json(f.read())
    exp.run()


if __name__ == "__main__":
    expy()
