import typer
from .experiment import Experiment

expy = typer.Typer()


@expy.command()
def run(path: str) -> None:
    with open(path) as f:
        exp = Experiment.from_json(f.read())
    exp.run()


if __name__ == "__main__":
    expy()
