"""Install a built wheel into a clean environment and smoke the installed console script.

Exists as a script rather than as shell in workflow YAML because the thing being proved is
*platform independence*. A wheel that imports on Linux and resolves its console script
through ``bin/`` may still fail on Windows, where the same environment puts it in
``Scripts/`` under a different name. Expressing the check in Python means both platforms
run byte-identical logic, so a failure is a property of the wheel and not of the shell that
happened to be interpreting the path.

The checks are deliberately shallow but not cosmetic:

* the wheel installs into an environment that has nothing else in it;
* the installed ``psd`` console script runs, rather than ``python -m psd``, so a broken
  entry point is caught here instead of by an operator;
* the version, the canonical schema registry, and *every* ``psd openpowerlifting``
  subcommand answer ``--help``. A command group whose subcommands are registered
  conditionally imports cleanly and then fails only for whoever runs it.

Usage::

    uv run python scripts/wheel_smoke.py
    uv run python scripts/wheel_smoke.py dist/powerlifting_strength_dynamics-0.1.0-py3-none-any.whl
    uv run python scripts/wheel_smoke.py --python 3.12 --keep

Takes no arguments in the normal case and finds the wheel itself, because a shell glob
crossed the boundary between two operating systems in exactly the way this script exists to
prevent: ``dist/*.whl`` is expanded by bash and handed to the program literally by PowerShell.
Making the script glob its own argument removes that difference instead of documenting it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ("main", "psd_executable", "venv_python")

#: Every ``psd openpowerlifting`` subcommand, checked individually. The group importing is
#: not evidence that its subcommands are registered.
CORPUS_SUBCOMMANDS: tuple[str, ...] = ("acquire", "inspect", "build", "audit", "verify")

#: The other command groups a fresh install must be able to answer for.
TOP_LEVEL_GROUPS: tuple[str, ...] = ("canonical", "inspect", "ontology", "paths", "schema")


class WheelSmokeError(RuntimeError):
    """Raised when the installed wheel does not behave as an installed package."""


def venv_python(environment: Path) -> Path:
    """Return the interpreter inside *environment*.

    Raises:
        WheelSmokeError: The environment has no interpreter where a venv always puts one.
    """
    directory = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    candidate = environment / directory / name
    if not candidate.is_file():
        msg = f"No interpreter at {candidate}; uv did not create the environment as expected."
        raise WheelSmokeError(msg)
    return candidate


def psd_executable(environment: Path) -> Path:
    """Return the installed ``psd`` console script inside *environment*.

    Raises:
        WheelSmokeError: The wheel installed no console script where one must be.
    """
    directory = "Scripts" if os.name == "nt" else "bin"
    name = "psd.exe" if os.name == "nt" else "psd"
    candidate = environment / directory / name
    if not candidate.is_file():
        msg = (
            f"No psd console script at {candidate}. The wheel declares one, so either the "
            "entry point is broken or this environment is not the one it was installed into."
        )
        raise WheelSmokeError(msg)
    return candidate


def _run(command: list[str], *, label: str) -> str:
    """Run *command*, returning its output.

    Raises:
        WheelSmokeError: The command failed. Its output is in the message, because a smoke
            failure nobody can read is a smoke failure nobody can act on.
    """
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        msg = (
            f"{label} failed with exit code {completed.returncode}\n"
            f"  command: {' '.join(command)}\n"
            f"  stdout: {completed.stdout.strip()}\n"
            f"  stderr: {completed.stderr.strip()}"
        )
        raise WheelSmokeError(msg)
    return completed.stdout


def smoke(wheel: Path, *, python: str, environment: Path) -> list[str]:
    """Install *wheel* into a clean *environment* and exercise it. Return the check names.

    Args:
        wheel: The built wheel to install.
        python: Interpreter version request passed to ``uv venv``.
        environment: Directory to create the clean environment in. Must not exist.

    Returns:
        One line per check that ran, in order.

    Raises:
        WheelSmokeError: uv is unavailable, or any check failed.
    """
    if shutil.which("uv") is None:
        msg = "uv is not on PATH; the wheel smoke installs through uv and needs it."
        raise WheelSmokeError(msg)
    checks: list[str] = []
    _run(["uv", "venv", "--python", python, str(environment)], label="uv venv")
    checks.append(f"clean environment created ({python})")
    _run(
        ["uv", "pip", "install", "--python", str(venv_python(environment)), str(wheel)],
        label="uv pip install",
    )
    checks.append(f"installed {wheel.name}")
    interpreter = venv_python(environment)
    _run([str(interpreter), "-c", "import psd"], label="import psd")
    checks.append("import psd")
    psd = psd_executable(environment)
    _run([str(psd), "--help"], label="psd --help")
    checks.append("psd --help")
    reported = _run([str(psd), "version"], label="psd version").strip()
    checks.append(f"psd version -> {reported}")
    for group in TOP_LEVEL_GROUPS:
        _run([str(psd), group, "--help"], label=f"psd {group} --help")
    checks.append(f"psd {', '.join(TOP_LEVEL_GROUPS)} --help")
    for command in CORPUS_SUBCOMMANDS:
        _run(
            [str(psd), "openpowerlifting", command, "--help"],
            label=f"psd openpowerlifting {command}",
        )
    checks.append(f"psd openpowerlifting {{{','.join(CORPUS_SUBCOMMANDS)}}} --help")
    return checks


def resolve_wheels(given: list[Path]) -> list[Path]:
    """Return the wheels to install: the given paths, or everything under ``dist/``.

    Raises:
        WheelSmokeError: No wheel was named and none was built.
    """
    if given:
        return given
    found = sorted(path for path in Path("dist").glob("*.whl") if path.is_file())
    if not found:
        msg = "no wheel given and none under dist/; run `uv build` first."
        raise WheelSmokeError(msg)
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the smoke check. Return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheels",
        nargs="*",
        type=Path,
        help="Built wheel(s) to install; defaults to every wheel under dist/.",
    )
    parser.add_argument("--python", default="3.12", help="Interpreter version for the venv.")
    parser.add_argument(
        "--keep", action="store_true", help="Leave the environment behind for inspection."
    )
    arguments = parser.parse_args(argv)
    try:
        wheels = resolve_wheels(arguments.wheels)
        wheel = wheels[0]
    except WheelSmokeError as error:
        sys.stderr.write(f"wheel_smoke: {error}\n")
        return 1
    if not wheel.is_file():
        sys.stderr.write(f"wheel_smoke: no wheel at {wheel}\n")
        return 1
    scratch = Path(tempfile.mkdtemp(prefix="psd-wheel-smoke-"))
    environment = scratch / "env"
    try:
        for check in smoke(wheel, python=arguments.python, environment=environment):
            sys.stdout.write(f"  ok  {check}\n")
    except WheelSmokeError as error:
        sys.stderr.write(f"wheel_smoke: {error}\n")
        return 1
    finally:
        if arguments.keep:
            sys.stdout.write(f"  kept  {scratch}\n")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    sys.stdout.write("wheel smoke: OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
