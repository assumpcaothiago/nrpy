"""
Build, run, and plot the Cartesian-only Counterexample 2 minimal diagnostic.
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def run_command(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess and raise on failure."""
    subprocess.run(cmd, cwd=cwd, check=True)


def load_csv(path: Path) -> np.ndarray:
    """Load one diagnostic CSV."""
    return np.genfromtxt(path, delimiter=",", names=True)


def summarize_errors(label: str, data: np.ndarray) -> None:
    """Print max and L2 error summaries for one dataset."""
    abs_error = np.asarray(data["abs_error"], dtype=np.float64)
    l2 = float(np.sqrt(np.mean(abs_error**2)))
    max_err = float(np.max(abs_error))
    print(f"{label}: max={max_err:.6e}  L2={l2:.6e}")


def plot_cut(
    data: np.ndarray,
    x_key: str,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot one 1D Cartesian cut."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data[x_key], data["dhdx"], label="Interpolated", linewidth=2.0)
    ax.plot(data[x_key], data["exact"], label="Exact", linestyle="--", linewidth=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\partial_x h_{\theta\theta}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_plane_value(
    data: np.ndarray,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    norm: TwoSlopeNorm,
) -> None:
    """Plot a 2D value heatmap from long-form CSV rows."""
    x_vals = np.unique(np.asarray(data[x_key], dtype=np.float64))
    y_vals = np.unique(np.asarray(data[y_key], dtype=np.float64))
    values = np.asarray(data["dhdx"], dtype=np.float64).reshape(len(x_vals), len(y_vals))

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(
        x_vals,
        y_vals,
        values.T,
        shading="auto",
        cmap="coolwarm",
        norm=norm,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\partial_x h_{\theta\theta}$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, run, and plot the Cartesian-only Counterexample 2 minimal diagnostic."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Path to the generated project directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where CSVs and PNGs will be written. Default=<project-dir>/counterexample2_output",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (project_dir / "counterexample2_output").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    executable = project_dir / project_dir.name

    run_command(["make", "-j"], cwd=project_dir)
    run_command([str(executable), str(output_dir)], cwd=project_dir)

    cart_x_line = load_csv(output_dir / "cart_x_line.csv")
    cart_y_line = load_csv(output_dir / "cart_y_line.csv")
    cart_z_line = load_csv(output_dir / "cart_z_line.csv")
    cart_xy_plane = load_csv(output_dir / "cart_xy_plane.csv")
    cart_xz_plane = load_csv(output_dir / "cart_xz_plane.csv")
    cart_yz_plane = load_csv(output_dir / "cart_yz_plane.csv")

    summarize_errors("cart_x_line", cart_x_line)
    summarize_errors("cart_y_line", cart_y_line)
    summarize_errors("cart_z_line", cart_z_line)
    summarize_errors("cart_xy_plane", cart_xy_plane)
    summarize_errors("cart_xz_plane", cart_xz_plane)
    summarize_errors("cart_yz_plane", cart_yz_plane)

    plot_cut(
        cart_x_line,
        "x_req",
        "x",
        r"Counterexample 2: Cartesian Line at $y=z=0$",
        output_dir / "cart_x_line.png",
    )
    plot_cut(
        cart_y_line,
        "y_req",
        "y",
        r"Counterexample 2: Cartesian Line at $x=z=0$",
        output_dir / "cart_y_line.png",
    )
    plot_cut(
        cart_z_line,
        "z_req",
        "z",
        r"Counterexample 2: Cartesian Line at $x=y=0$",
        output_dir / "cart_z_line.png",
    )

    cart_plane_datasets = (cart_xy_plane, cart_xz_plane, cart_yz_plane)
    max_delta = max(
        max(
            float(
                np.max(
                    np.abs(np.asarray(dataset["dhdx"], dtype=np.float64) - 1.0)
                )
            )
            for dataset in cart_plane_datasets
        ),
        1.0e-12,
    )
    cart_plane_norm = TwoSlopeNorm(
        vmin=1.0 - max_delta,
        vcenter=1.0,
        vmax=1.0 + max_delta,
    )

    plot_plane_value(
        cart_xy_plane,
        "x_req",
        "y_req",
        "x",
        "y",
        r"Counterexample 2: Cartesian $(x,y)$ Plane at $z=0$",
        output_dir / "cart_xy_plane.png",
        cart_plane_norm,
    )
    plot_plane_value(
        cart_xz_plane,
        "x_req",
        "z_req",
        "x",
        "z",
        r"Counterexample 2: Cartesian $(x,z)$ Plane at $y=0$",
        output_dir / "cart_xz_plane.png",
        cart_plane_norm,
    )
    plot_plane_value(
        cart_yz_plane,
        "y_req",
        "z_req",
        "y",
        "z",
        r"Counterexample 2: Cartesian $(y,z)$ Plane at $x=0$",
        output_dir / "cart_yz_plane.png",
        cart_plane_norm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
