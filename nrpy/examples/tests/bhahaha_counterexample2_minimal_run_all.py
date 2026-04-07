"""
Generate, build, run, and plot both minimal Counterexample 2 diagnostics.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    """Run one subprocess, printing the command before execution."""
    print(f"$ (cd {cwd} && {' '.join(cmd)})", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate, build, run, and plot both minimal Counterexample 2 diagnostics."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Path to the nrpy repository root. Default=auto-detected.",
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Skip project regeneration and only build/run/plot the existing generated projects.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"NRPy repo root not found: {repo_root}")

    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = "/tmp"
    env["HOME"] = "/tmp"
    env["MPLCONFIGDIR"] = "/tmp"

    python_exe = sys.executable
    generator_jobs = [
        (
            "naive",
            repo_root / "nrpy/examples/bhahaha_counterexample2_minimal.py",
            repo_root / "project/BHaHAHA-counterexample2-minimal",
        ),
        (
            "corrected",
            repo_root / "nrpy/examples/bhahaha_counterexample2_minimal_corrected.py",
            repo_root / "project/BHaHAHA-counterexample2-minimal-corrected",
        ),
    ]
    diagnostic_script = repo_root / "nrpy/examples/tests/bhahaha_counterexample2_minimal_diagnostic.py"

    if not args.no_regenerate:
        for _, generator_script, _ in generator_jobs:
            run_command([python_exe, str(generator_script)], cwd=repo_root, env=env)

    for _, _, project_dir in generator_jobs:
        run_command(
            [python_exe, str(diagnostic_script), "--project-dir", str(project_dir)],
            cwd=repo_root,
            env=env,
        )

    print("\nArtifacts written to:")
    for label, _, project_dir in generator_jobs:
        print(f"  {label}: {project_dir / 'counterexample2_output'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
