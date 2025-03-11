import typer
from .experiment import Experiment
from serde.json import from_json

expy = typer.Typer()


@expy.command()
def run(path: str) -> None:
    with open(path) as f:
        exp = from_json(Experiment, f.read())
    exp.run()


if __name__ == "__main__":
    expy()
