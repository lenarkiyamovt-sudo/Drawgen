"""
STL Symmetry Detector - DEPRECATED

This module is deprecated. Please use:
    from stl_drawing.features import CylinderDetector, SymmetryDetector

This file remains for backward compatibility.
"""

import warnings
import sys

warnings.warn(
    "stl_symmetry_detector is deprecated. "
    "Use 'from stl_drawing.features import CylinderDetector, SymmetryDetector' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from stl_drawing.features.cylinder_detector import (
    CylinderDetector,
    SymmetryDetector,
    load_vertices,
    load_mesh_data,
    compute_cylinder_projections,
    compute_all_projections,
    STANDARD_VIEWS,
)

__all__ = [
    "CylinderDetector",
    "SymmetryDetector",
    "load_vertices",
    "load_mesh_data",
    "compute_cylinder_projections",
    "compute_all_projections",
    "STANDARD_VIEWS",
]


def main():
    """CLI entry point - delegates to new module."""
    import argparse
    import time
    import json
    import numpy as np

    from stl_drawing.features.cylinder_detector import MAX_PTS

    ap = argparse.ArgumentParser(description="STL symmetry + cylinders (deprecated CLI)")
    ap.add_argument("stl_file")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-cn", type=int, default=8)
    ap.add_argument("--max-pts", type=int, default=MAX_PTS)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--no-cylinders", action="store_true")
    ap.add_argument("--min-dih", type=float, default=1.0)
    ap.add_argument("--max-dih", type=float, default=60.0)
    ap.add_argument("--min-reg", type=int, default=6)
    ap.add_argument("--min-arc", type=float, default=45.0)
    ap.add_argument("--sharp-angle", type=float, default=15.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--projections", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    # Configure logging for verbose mode
    import logging
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    det = SymmetryDetector(
        a.tol, a.threshold,
        36 if a.dense else 24, 18 if a.dense else 12,
        a.max_cn, a.max_pts,
        not a.no_cylinders, a.verbose,
        min_dih=a.min_dih, max_dih=a.max_dih,
        min_reg=a.min_reg, min_arc=a.min_arc,
        sharp_angle=a.sharp_angle
    )

    t0 = time.perf_counter()
    res = det.detect(a.stl_file)

    # Print results
    print("\n" + "=" * 60)
    print("  SYMMETRY ANALYSIS RESULTS")
    print("=" * 60)

    sym = res["symmetry_axes"]
    print(f"\nMirror symmetry axes: {len(sym)}")
    for i, (sc, ax) in enumerate(sym, 1):
        print(f"  #{i}  axis=[{ax[0]:+.4f}, {ax[1]:+.4f}, {ax[2]:+.4f}]  score={sc:.3f}")

    rot = res["rotation_axes"]
    if rot:
        print(f"\nRotation symmetry axes:")
        for n in sorted(rot):
            for i, (sc, ax) in enumerate(rot[n], 1):
                print(f"  C{n} #{i}  axis=[{ax[0]:+.4f}, {ax[1]:+.4f}, {ax[2]:+.4f}]  score={sc:.3f}")

    cyls = res.get("cylinders", [])
    if cyls:
        for ct, lb in [("hole", "HOLES"), ("boss", "BOSSES"), ("cylinder", "CYLINDERS")]:
            sub = [c for c in cyls if c["type"] == ct]
            if sub:
                print(f"\nCylindrical {lb}: {len(sub)}")
                for i, c in enumerate(sub, 1):
                    print(f"  #{i}  dia={c['radius']*2:.3f}  R={c['radius']:.3f}  "
                          f"L={c['length']:.3f}  quality={c['fit_quality']:.2f}")

    print(f"\nTolerance: {res['tolerance']:.5f}  |  Vertices: {res['n_vertices']:,}")
    print(f"Time: {time.perf_counter()-t0:.2f} sec")
    print("=" * 60)

    if a.json:
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        proj = compute_all_projections(res)
        export = {
            "source": a.stl_file,
            "cylinders": [{
                "id": i,
                "type": c["type"],
                "diameter": c["radius"] * 2,
                "radius": c["radius"],
                "length": c["length"],
                "axis": c["axis"].tolist(),
                "center": c["center"].tolist(),
                "fit_quality": c["fit_quality"],
            } for i, c in enumerate(cyls)],
            "symmetry_axes": [{
                "type": "mirror",
                "axis": ax.tolist(),
                "score": sc,
            } for sc, ax in res.get("symmetry_axes", [])],
            "rotation_axes": [{
                "type": f"C{n}",
                "axis": ax.tolist(),
                "score": sc,
            } for n, axl in res.get("rotation_axes", {}).items()
              for sc, ax in axl],
            "projections": proj,
        }
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2, default=_serialize)
        print(f"JSON saved: {a.json}")


if __name__ == "__main__":
    main()
