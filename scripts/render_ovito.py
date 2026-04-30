"""Render headless prin OVITO Python module — produce PNG-uri din GSD + OBJ.

Folosit când nu ai OVITO GUI instalat. Ai nevoie doar de `pip install ovito`
(deja disponibil). Produce:
  - overview.png     : frame final, vedere 3D, particule + geometrie translucidă
  - frames/*.png     : animație (opțional)

Usage:
    python scripts/render_ovito.py sim_out/loop_chambers_ou
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="directorul cu run.gsd și geometry.obj")
    ap.add_argument("--frames", action="store_true", help="render toate frame-urile")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    run_dir = args.run_dir
    gsd_path = run_dir / "run.gsd"
    obj_path = run_dir / "geometry.obj"
    if not gsd_path.exists():
        print(f"[err] lipsește {gsd_path}", file=sys.stderr)
        return 1

    from ovito.io import import_file
    from ovito.vis import Viewport, TachyonRenderer

    # pipeline particule
    pipeline = import_file(str(gsd_path))
    data = pipeline.compute()
    print(f"[ovito] {data.particles.count} particles, {pipeline.source.num_frames} frames")

    # ajustări vizuale
    pipeline.source.data.particles.vis.radius = 0.8
    pipeline.add_to_scene()

    # geometrie ca mesh static (opțional — OVITO suportă OBJ)
    if obj_path.exists():
        try:
            geom_pipeline = import_file(str(obj_path))
            geom_pipeline.add_to_scene()
            print(f"[ovito] loaded geometry: {obj_path}")
        except Exception as e:
            print(f"[warn] nu pot încărca OBJ: {e}")

    # viewport
    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all()
    renderer = TachyonRenderer()

    # frame final
    out_png = run_dir / "overview.png"
    pipeline.source.num_frames  # asigură decoding
    last = pipeline.source.num_frames - 1
    vp.render_image(
        size=(args.width, args.height),
        filename=str(out_png),
        renderer=renderer,
        frame=last,
    )
    print(f"[ok] written {out_png}")

    if args.frames:
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for k in range(pipeline.source.num_frames):
            vp.render_image(
                size=(args.width, args.height),
                filename=str(frames_dir / f"frame_{k:04d}.png"),
                renderer=renderer,
                frame=k,
            )
        print(f"[ok] rendered {pipeline.source.num_frames} frames -> {frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
