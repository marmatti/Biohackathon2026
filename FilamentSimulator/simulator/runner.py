from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List
import random

import yaml

from simulator.config import ensure_output_dir, load_config
from simulator.movie import simulate_movie, write_simulation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple fluorescence movie simulations from a base YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML config, e.g. configs/base.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root output directory for all runs, e.g. outputs/batch_001",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of runs to generate if --seeds is not provided.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of seeds, e.g. --seeds 1 2 5 10",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override a config value using dotted paths, e.g. "
            "--override cells.n_cells=3 "
            "--override filament_event.probability_any_filament=1.0"
        ),
    )
    parser.add_argument(
        "--save-effective-config",
        action="store_true",
        help="Save the fully resolved config used for each run.",
    )
    return parser.parse_args()


def parse_override(expr: str) -> tuple[List[str], Any]:
    """
    Parse override expressions like:
        cells.n_cells=3
        noise.gaussian_read_sigma=8.0
        filament.translation_sigma_xyz_px=[0.4,0.4,0.15]
        output.noisy_png_dir='frames_noisy'
    """
    if "=" not in expr:
        raise ValueError(f"Invalid override '{expr}'. Expected dotted.path=value")

    key, value = expr.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        raise ValueError(f"Invalid override '{expr}'. Empty key.")

    # yaml.safe_load parses ints, floats, bools, lists, dicts, quoted strings, etc.
    parsed_value = yaml.safe_load(value)
    return key.split("."), parsed_value


def set_nested(cfg: Dict[str, Any], keys: Iterable[str], value: Any) -> None:
    """
    Set a nested config value given a list of keys.
    """
    keys = list(keys)
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Return a deep-copied config with dotted-path overrides applied.
    """
    cfg_new = copy.deepcopy(cfg)
    for expr in overrides:
        keys, value = parse_override(expr)
        set_nested(cfg_new, keys, value)
    return cfg_new

def generate_seeds(n_runs: int, cfg: Dict[str, Any]) -> List[int]:
    """
    Generate seeds for a batch of runs.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0")

    if n_runs == 1:
        if "seed" not in cfg:
            raise KeyError("Config must contain 'seed'")
        return [int(cfg["seed"])]

    rng = random.SystemRandom()
    seeds = set()

    while len(seeds) < n_runs:
        seeds.add(rng.randrange(0, 2**31 - 1))

    return sorted(seeds)

def save_effective_config(cfg: Dict[str, Any], out_dir: Path) -> None:
    config_path = out_dir / "effective_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    args = parse_args()

    base_cfg = load_config(args.config)
    seeds = generate_seeds(args.n_runs, base_cfg)

    output_root = ensure_output_dir(args.output_root)

    print(f"Base config: {Path(args.config).resolve()}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Number of runs: {len(seeds)}")
    print(f"Seeds: {seeds}")
    if args.override:
        print("Overrides:")
        for x in args.override:
            print(f"  - {x}")

    for run_idx, seed in enumerate(seeds):
        cfg = apply_overrides(base_cfg, args.override)
        cfg["seed"] = int(seed)

        run_dir = output_root / f"run_{run_idx:03d}_seed_{seed}"
        ensure_output_dir(run_dir)

        sim_data = simulate_movie(cfg)
        write_simulation_outputs(sim_data=sim_data, out_dir=run_dir, cfg=cfg)

        if args.save_effective_config:
            save_effective_config(cfg, run_dir)

        print(f"[{run_idx + 1}/{len(seeds)}] Saved run to {run_dir}")

    print("All runs completed.")


if __name__ == "__main__":
    main()