import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .experiment import Experiment

expy = typer.Typer()


@expy.command()
def run(
    path: Annotated[Path, typer.Argument(default_factory=Path.cwd)],
    *,
    skip_inference: Annotated[
        bool, typer.Option(help="Skip inference step")
    ] = False,
) -> None:
    exp_path = path.parent if path.is_file() else path
    os.chdir(exp_path)
    with Path.open(Path("exp.json")) as f:
        exp = Experiment.model_validate_json(f.read())
    if not skip_inference:
        exp.run_inference()
    exp.run_post_inference()

@expy.command()
def schema(
    path: Annotated[Path, typer.Argument(
        default_factory=lambda:Path.cwd() / "schema.json"
    )],
    *,
    overwrite: Annotated[bool, typer.Option(
        help="overwrites file if true"
    )] = False,
) -> None:
    mode = "w" if overwrite else "x"
    with Path.open(path, mode=mode) as schema_file:
        schema = Experiment.model_json_schema()
        schema_file.write(json.dumps(schema, indent=2))

if __name__ == "__main__":
    expy()
