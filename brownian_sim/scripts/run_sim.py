"""CLI: rulare simulare dintr-un fișier YAML."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from brownian_sim.physics import make_wall_model
from brownian_sim.simulation import Simulation, SimulationConfig


def _build_assembly(geom_cfg: dict):
    preset = geom_cfg["preset"]
    params = geom_cfg.get("params") or {}

    if preset == "simple_box":
        from brownian_sim.geometry.presets.simple_box import SimpleBoxParams, build
        from brownian_sim.materials.wall_material import WallMaterial
        mat = params.get("material")
        kwargs = {k: tuple(v) if isinstance(v, list) else v
                  for k, v in params.items() if k != "material"}
        if mat is not None:
            kwargs["material"] = WallMaterial(**mat)
        return build(SimpleBoxParams(**kwargs)) if kwargs else build()

    if preset == "loop_chambers":
        from brownian_sim.geometry.presets.loop_chambers import LoopChambersParams, build
        if not params:
            return build()
        from brownian_sim.materials.wall_material import WallMaterial

        def _mat(v):
            return WallMaterial(**v) if isinstance(v, dict) else v

        def _mat_tuple(v):
            return tuple(_mat(x) for x in v) if isinstance(v, list) else v

        parsed = {}
        for k, v in params.items():
            if k in ("mat_cube_L", "mat_cube_R", "mat_ret"):
                parsed[k] = _mat(v)
            elif k == "mat_funnel":
                parsed[k] = _mat_tuple(v)
            else:
                parsed[k] = tuple(v) if isinstance(v, list) else v
        return build(LoopChambersParams(**parsed))

    raise ValueError(f"preset necunoscut: {preset}. Opțiuni: simple_box, loop_chambers")


def _build_writers(output_cfg: dict, assembly, track_ids: np.ndarray):
    out_dir = output_cfg["dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    writers = []
    if output_cfg.get("csv", True):
        from brownian_sim.io.csv_logger import CSVLoggers
        writers.append(CSVLoggers(out_dir, assembly.names))
    if output_cfg.get("gsd", True):
        try:
            from brownian_sim.io.gsd_writer import GSDWriter
            box = tuple(output_cfg.get("box", [520.0, 320.0, 260.0]))
            writers.append(GSDWriter(
                path=f"{out_dir}/run.gsd",
                box=box,
                subset=int(output_cfg.get("gsd_subset", 15000)),
                track_ids=track_ids,
            ))
        except RuntimeError as e:
            print(f"[WARN] {e}", file=sys.stderr)
    return writers


def main() -> int:
    ap = argparse.ArgumentParser(description="Run brownian simulation from YAML config")
    ap.add_argument("--config", "-c", required=True, type=Path, help="path to YAML config")
    ap.add_argument("--steps", type=int, default=None, help="override steps")
    ap.add_argument("--n", type=int, default=None, help="override n_particles")
    ap.add_argument("--device", type=int, default=None, help="GPU device index (0,1,...); -1 sau absent = CPU")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    assembly = _build_assembly(cfg["geometry"])

    sim_cfg_raw = cfg["simulation"]
    if args.steps is not None:
        sim_cfg_raw["steps"] = args.steps
    if args.n is not None:
        sim_cfg_raw["n_particles"] = args.n
    if args.device is not None:
        sim_cfg_raw["device"] = args.device
    if args.quiet:
        sim_cfg_raw["quiet"] = True

    sim_cfg = SimulationConfig(**sim_cfg_raw)
    wall = make_wall_model(cfg["wall_model"], kT_over_m=sim_cfg.kT / sim_cfg.mass)

    # track_ids = primele track_k particule (convenție stabilă)
    track_ids = np.arange(min(sim_cfg.track_k, sim_cfg.n_particles), dtype=np.int64)
    sim_cfg.writers = _build_writers(cfg.get("output", {}), assembly, track_ids)

    sim = Simulation(assembly, wall, sim_cfg)
    print(f"[run] config={args.config} assembly={assembly.names} wall={cfg['wall_model']}")
    sim.run()

    # închide writers
    for w in sim_cfg.writers:
        if hasattr(w, "close"):
            w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
