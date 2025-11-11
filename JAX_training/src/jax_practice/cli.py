"""Command-line interface for the JAX practice playground."""

from __future__ import annotations

import random
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Optional

import typer

from jax_practice.problems import PROBLEMS, Problem

app = typer.Typer(help="Solve JAX problems locally with a LeetCode-style workflow.")

WORKSPACE_DIR = Path("workspace")


def _pick_problem(slug: Optional[str]) -> Problem:
    if not PROBLEMS:
        typer.echo("No problems registered yet.")
        raise typer.Exit(code=1)
    if slug is None:
        return random.choice(list(PROBLEMS.values()))
    try:
        return PROBLEMS[slug]
    except KeyError as exc:
        known = ", ".join(sorted(PROBLEMS))
        typer.echo(f"Unknown problem '{slug}'. Known slugs: {known}")
        raise typer.Exit(code=1) from exc


def _load_user_module(path: Path) -> ModuleType:
    spec = importlib_util.spec_from_file_location("user_solution", path)
    if spec is None or spec.loader is None:
        typer.echo(f"Unable to import solution from {path}")
        raise typer.Exit(code=1)
    module = importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"Error importing {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    return module


@app.command()
def list() -> None:
    """List all available problems."""
    if not PROBLEMS:
        typer.echo("No problems registered yet.")
        raise typer.Exit()
    for problem in sorted(PROBLEMS.values(), key=lambda p: p.slug):
        typer.echo(f"- {problem.slug}: {problem.title}")


@app.command()
def prompt(slug: Optional[str] = typer.Argument(None, help="Problem slug.")) -> None:
    """Print the prompt for a problem."""
    problem = _pick_problem(slug)
    typer.echo(f"# {problem.title}\n")
    typer.echo(problem.prompt.strip())


@app.command()
def solution(slug: Optional[str] = typer.Argument(None, help="Problem slug.")) -> None:
    """Show the sample solution."""
    problem = _pick_problem(slug)
    if not problem.example_solution:
        typer.echo("No stored reference solution for this problem.")
        raise typer.Exit(code=1)
    typer.echo(problem.example_solution.rstrip())


@app.command()
def start(
    slug: Optional[str] = typer.Argument(None, help="Problem slug; random if omitted."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing workspace stub."
    ),
) -> None:
    """Create or overwrite a starter file in workspace/."""
    problem = _pick_problem(slug)
    WORKSPACE_DIR.mkdir(exist_ok=True)
    stub_path = WORKSPACE_DIR / f"{problem.slug}.py"
    if stub_path.exists() and not force:
        typer.echo(f"{stub_path} already exists. Use --force to overwrite.")
        raise typer.Exit(code=1)

    stub_path.write_text(problem.render_starter(), encoding="utf-8")
    typer.echo(f"Wrote starter for '{problem.slug}' to {stub_path}")
    typer.echo("Open the file, implement the function, then run `jax-practice test`.")


@app.command()
def test(
    slug: Optional[str] = typer.Argument(None, help="Problem slug to test."),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to the solution file (defaults to workspace/<slug>.py).",
    ),
) -> None:
    """Run the hidden tests for a problem."""
    problem = _pick_problem(slug)
    candidate_file = file or (WORKSPACE_DIR / f"{problem.slug}.py")
    if not candidate_file.exists():
        typer.echo(f"Solution file not found: {candidate_file}")
        raise typer.Exit(code=1)

    module = _load_user_module(candidate_file)
    try:
        problem.run_hidden_tests(module)
    except AssertionError as exc:
        typer.echo("❌ Tests failed.")
        if str(exc):
            typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover
        typer.echo(f"❌ Unexpected error: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo("✅ All tests passed!")


def main(argv: Optional[list[str]] = None) -> None:
    app(argv or sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
