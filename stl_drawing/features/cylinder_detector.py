"""
STL Symmetry & Cylinder Detection Module (v4.2)

Topology-based cylinder detection using:
- KDTree vertex deduplication for correct face adjacency
- Dihedral angle analysis for curved face marking
- Region growing across curved face edges
- PCA normal analysis for axis fitting
- Least-squares circle fitting

Dependencies: numpy, scipy, numpy-stl
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree as KDTree

logger = logging.getLogger(__name__)

# Check scipy version for workers parameter
try:
    import scipy
    _SCIPY_VERSION = tuple(int(x) for x in scipy.__version__.split(".")[:2])
    _SCIPY_HAS_WORKERS = _SCIPY_VERSION >= (1, 6)
except Exception:
    _SCIPY_HAS_WORKERS = False

_KDT_KW: Dict[str, Any] = {"workers": -1} if _SCIPY_HAS_WORKERS else {}
MAX_PTS = 5000


# ============================================================================
# Mesh Data Loading
# ============================================================================

def load_vertices(path: str) -> np.ndarray:
    """Load unique vertices from STL file."""
    from stl import mesh
    m = mesh.Mesh.from_file(path)
    v = m.vectors.reshape(-1, 3)
    return np.unique(np.round(v, 5), axis=0).astype(np.float64)


def load_mesh_data(path: str) -> Dict[str, np.ndarray]:
    """Load mesh with computed normals, centroids, areas."""
    from stl import mesh
    m = mesh.Mesh.from_file(path)
    V = m.vectors.astype(np.float64)
    e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    cr = np.cross(e1, e2)
    nm = np.linalg.norm(cr, axis=1, keepdims=True)
    nm = np.where(nm < 1e-12, 1.0, nm)
    return {
        "vectors": V,
        "normals": cr / nm,
        "centroids": V.mean(axis=1),
        "areas": np.linalg.norm(cr, axis=1) * 0.5
    }


def center(pts: np.ndarray) -> np.ndarray:
    """Center points around origin."""
    return pts - pts.mean(axis=0)


def maybe_subsample(pts: np.ndarray, mx: int = MAX_PTS) -> np.ndarray:
    """Subsample points if exceeding limit."""
    return pts if len(pts) <= mx else pts[np.random.choice(len(pts), mx, False)]


def pca_axes(pts: np.ndarray) -> np.ndarray:
    """Compute PCA axes of point cloud."""
    _, vecs = eigh(np.cov(pts.T))
    return vecs.T[::-1]


def sphere_candidates(n_az: int = 18, n_pol: int = 9) -> np.ndarray:
    """Generate sphere candidate directions."""
    dirs = []
    for i in range(n_pol + 1):
        th = np.pi * i / (2 * n_pol)
        for j in range(n_az):
            ph = 2 * np.pi * j / n_az
            dirs.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return np.unique(np.round(dirs, 6), axis=0)


# ============================================================================
# Topology Analysis
# ============================================================================

def _build_adjacency(vectors: np.ndarray) -> Tuple[Dict[int, set], float]:
    """Build face adjacency through KDTree vertex deduplication."""
    n_f = len(vectors)
    all_v = vectors.reshape(-1, 3)

    # Median edge length for tolerance
    el = []
    for fi in range(n_f):
        tri = vectors[fi]
        for k in range(3):
            el.append(np.linalg.norm(tri[(k + 1) % 3] - tri[k]))
    med_e = np.median(el)
    tol = med_e * 0.01

    tree = KDTree(all_v)
    vid = np.full(len(all_v), -1, dtype=int)
    nxt = 0
    for i in range(len(all_v)):
        if vid[i] >= 0:
            continue
        for j in tree.query_ball_point(all_v[i], tol):
            if vid[j] < 0:
                vid[j] = nxt
        nxt += 1

    e2f: Dict[Tuple[int, int], set] = defaultdict(set)
    for fi in range(n_f):
        vi = [vid[fi * 3 + k] for k in range(3)]
        for k in range(3):
            ek = (min(vi[k], vi[(k + 1) % 3]), max(vi[k], vi[(k + 1) % 3]))
            e2f[ek].add(fi)

    adj: Dict[int, set] = defaultdict(set)
    for faces in e2f.values():
        fl = list(faces)
        for a in fl:
            for b in fl:
                if a != b:
                    adj[a].add(b)

    return adj, med_e


def _mark_curved_faces(adj: Dict[int, set], normals: np.ndarray, n_f: int,
                       min_deg: float = 3.0, max_deg: float = 60.0) -> np.ndarray:
    """Mark faces as curved if neighbor dihedral angle is in range."""
    mna, mxa = np.radians(min_deg), np.radians(max_deg)
    curved = np.zeros(n_f, dtype=bool)
    for fi in range(n_f):
        for fj in adj.get(fi, []):
            dot = np.clip(np.dot(normals[fi], normals[fj]), -1, 1)
            a = np.arccos(dot)
            if mna <= a <= mxa:
                curved[fi] = True
                break
    return curved


def _grow_regions(adj: Dict[int, set], curved: np.ndarray, n_f: int,
                  min_sz: int = 6) -> List[List[int]]:
    """BFS region growing across curved face edges."""
    vis = np.zeros(n_f, dtype=bool)
    regions = []
    for s in range(n_f):
        if vis[s] or not curved[s]:
            vis[s] = True
            continue
        reg = set()
        q = [s]
        vis[s] = True
        while q:
            fi = q.pop()
            reg.add(fi)
            for fj in adj.get(fi, []):
                if not vis[fj] and curved[fj]:
                    vis[fj] = True
                    q.append(fj)
        if len(reg) >= min_sz:
            regions.append(sorted(reg))
    return regions


def _segment_by_sharp_edges(adj: Dict[int, set], normals: np.ndarray, n_f: int,
                            sharp_angle_deg: float = 15.0) -> List[List[int]]:
    """Segment mesh by sharp edges using BFS."""
    sharp_rad = np.radians(sharp_angle_deg)
    vis = np.zeros(n_f, dtype=bool)
    segments = []
    for s in range(n_f):
        if vis[s]:
            continue
        seg = set()
        q = [s]
        vis[s] = True
        while q:
            fi = q.pop()
            seg.add(fi)
            for fj in adj.get(fi, []):
                if vis[fj]:
                    continue
                dot = np.clip(np.dot(normals[fi], normals[fj]), -1, 1)
                if np.arccos(dot) < sharp_rad:
                    vis[fj] = True
                    q.append(fj)
        segments.append(sorted(seg))
    return segments


def _segment_is_curved(face_ids: List[int], normals: np.ndarray,
                       min_spread: float = 0.05) -> bool:
    """Check if segment is curved (not planar)."""
    n = normals[np.array(face_ids)]
    if len(n) < 3:
        return False
    spread = np.std(n, axis=0).mean()
    return spread > min_spread


# ============================================================================
# Cylinder Fitting
# ============================================================================

def _fit_circle_lsq(pts: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Least squares circle fit. Returns (cx, cy, radius) or None."""
    n = len(pts)
    if n < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(n)])
    b = x ** 2 + y ** 2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = res[0], res[1]
    r2 = res[2] + cx ** 2 + cy ** 2
    return (cx, cy, np.sqrt(r2)) if r2 > 0 else None


def _angular_coverage(angles: np.ndarray) -> float:
    """Calculate angular coverage of points around circle."""
    if len(angles) < 2:
        return 0.0
    sa = np.sort(angles % (2 * np.pi))
    gaps = np.diff(sa)
    gaps = np.append(gaps, 2 * np.pi - sa[-1] + sa[0])
    return 2 * np.pi - gaps.max()


def _fit_cylinder(face_ids: List[int], normals: np.ndarray, centroids: np.ndarray,
                  areas: np.ndarray, min_fq: float = 0.75, min_arc: float = 90.0,
                  min_ld: float = 0.08, verbose: bool = False) -> Optional[Dict]:
    """Fit cylinder to face region. Returns cylinder dict or None."""
    idx = np.array(face_ids)
    n, c = normals[idx], centroids[idx]
    if len(idx) < 4:
        return None

    # PCA of normals -> axis
    cov = np.cov(n.T)
    eigv, eigvec = eigh(cov)
    axis = eigvec[:, 0].copy()
    axis /= np.linalg.norm(axis)
    if axis[2] < 0 or (axis[2] == 0 and axis[1] < 0):
        axis = -axis

    # Check normals perpendicular to axis
    dots = np.abs(n @ axis)
    if (dots < 0.35).mean() < 0.60:
        if verbose:
            logger.debug("Region %d faces: perpendicular ratio too low", len(idx))
        return None

    # Build orthonormal basis
    u = np.array([1., 0, 0])
    if abs(np.dot(u, axis)) > 0.9:
        u = np.array([0., 1, 0])
    u -= np.dot(u, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    along = c @ axis
    proj = c - np.outer(along, axis)
    p2d = np.column_stack([proj @ u, proj @ v])

    circ = _fit_circle_lsq(p2d)
    if circ is None:
        return None
    cx, cy, r = circ
    if r < 0.05:
        return None

    # Find inliers
    d = np.abs(np.hypot(p2d[:, 0] - cx, p2d[:, 1] - cy) - r)
    inl = d < 0.20 * r
    if inl.sum() < 4:
        if verbose:
            logger.debug("Region %d faces: only %d inliers", len(idx), inl.sum())
        return None

    ref = _fit_circle_lsq(p2d[inl])
    if ref is None:
        return None
    cx, cy, r = ref
    if r < 0.05:
        return None

    fd = np.abs(np.hypot(p2d[inl, 0] - cx, p2d[inl, 1] - cy) - r)
    fq = 1.0 - (fd.mean() / r) if r > 1e-9 else 0
    if fq < min_fq:
        if verbose:
            logger.debug("Region %d faces: fit quality %.3f < %.3f", len(idx), fq, min_fq)
        return None

    # Angular coverage
    rad = p2d[inl] - [cx, cy]
    rn = np.linalg.norm(rad, axis=1, keepdims=True)
    rn = np.where(rn < 1e-12, 1, rn)
    ru = rad / rn
    ang = np.arctan2(ru[:, 1], ru[:, 0])
    acov = _angular_coverage(ang)
    if acov < np.radians(min_arc):
        if verbose:
            logger.debug("Region %d faces: arc %.0f deg < %.0f", len(idx),
                        np.degrees(acov), min_arc)
        return None

    # Determine type (hole/boss/cylinder)
    n2d = np.column_stack([n[inl] @ u, n[inl] @ v])
    nn = np.linalg.norm(n2d, axis=1, keepdims=True)
    nn = np.where(nn < 1e-12, 1, nn)
    n2d /= nn
    md = (n2d * ru).sum(axis=1).mean()
    ctype = "hole" if md < -0.25 else ("boss" if md > 0.25 else "cylinder")

    ax_c = along[inl]
    L = ax_c.max() - ax_c.min()
    D = 2 * r
    if L / D < min_ld:
        if verbose:
            logger.debug("Region %d faces: L/D ratio %.3f < %.3f", len(idx), L / D, min_ld)
        return None

    return {
        "axis": axis,
        "center": cx * u + cy * v + ax_c.mean() * axis,
        "radius": float(r),
        "length": float(L),
        "n_faces": int(inl.sum()),
        "fit_quality": float(fq),
        "type": ctype,
        "angular_coverage_deg": float(np.degrees(acov))
    }


def _merge_cylinders(cyls: List[Dict], verbose: bool = False) -> List[Dict]:
    """Deduplicate and merge similar cylinders."""
    if len(cyls) <= 1:
        return cyls

    merged = []
    used = [False] * len(cyls)

    for i, c1 in enumerate(cyls):
        if used[i]:
            continue
        grp = [c1]
        for j in range(i + 1, len(cyls)):
            if used[j]:
                continue
            c2 = cyls[j]

            # Don't merge different types
            if c1["type"] != c2["type"]:
                continue

            rm = max(c1["radius"], c2["radius"])

            # Similar radii?
            if abs(c1["radius"] - c2["radius"]) / rm > 0.30:
                continue

            # Co-axial?
            adot = abs(np.dot(c1["axis"], c2["axis"]))
            if adot < np.cos(np.radians(25)):
                continue

            # Distance decomposition
            avg_axis = c1["axis"] + c2["axis"] * np.sign(np.dot(c1["axis"], c2["axis"]))
            avg_axis /= np.linalg.norm(avg_axis) + 1e-12
            dc = c2["center"] - c1["center"]
            along_dist = abs(np.dot(dc, avg_axis))
            cross_dist = np.sqrt(max(0, np.dot(dc, dc) - along_dist ** 2))

            # Cross distance check
            if cross_dist > 0.5 * rm:
                continue

            # Along distance check
            max_along = (c1["length"] + c2["length"]) / 2 + 0.5 * rm
            if along_dist > max_along:
                if verbose:
                    logger.debug("Not merging #%d+#%d: along=%.1f > %.1f",
                                i, j, along_dist, max_along)
                continue

            if verbose:
                logger.debug("Merging #%d+#%d: cross=%.2f, along=%.1f", i, j,
                            cross_dist, along_dist)
            grp.append(c2)
            used[j] = True

        used[i] = True
        best = max(grp, key=lambda c: c["fit_quality"] * c["n_faces"])
        best["n_faces"] = sum(c["n_faces"] for c in grp)
        merged.append(best)

    return merged


# ============================================================================
# Main Detector Classes
# ============================================================================

class CylinderDetector:
    """Topology-based cylinder detector."""

    def __init__(self, verbose: bool = False, min_dih: float = 1.0, max_dih: float = 60.0,
                 min_reg: int = 6, min_fq: float = 0.75, min_arc: float = 45.0,
                 min_ld: float = 0.08, sharp_angle: float = 15.0):
        self.verbose = verbose
        self.min_dih = min_dih
        self.max_dih = max_dih
        self.min_reg = min_reg
        self.min_fq = min_fq
        self.min_arc = min_arc
        self.min_ld = min_ld
        self.sharp_angle = sharp_angle

    def detect(self, md: Dict[str, np.ndarray], offset: np.ndarray) -> List[Dict]:
        """Detect cylinders in mesh data.

        Args:
            md: Mesh data dict with vectors, normals, centroids, areas
            offset: Coordinate offset to subtract from centroids

        Returns:
            List of cylinder dicts with axis, center, radius, length, type, etc.
        """
        N = md["normals"]
        C = md["centroids"] - offset
        A = md["areas"]
        V = md["vectors"]
        nf = len(N)

        logger.info("Cylinder detection: %d faces", nf)

        logger.debug("Building adjacency...")
        adj, me = _build_adjacency(V)
        ne = sum(len(v) for v in adj.values()) // 2
        logger.debug("Edges: %d, median edge length: %.2f", ne, me)

        # Dihedral angle histogram (verbose)
        if self.verbose:
            all_angles = []
            seen = set()
            for fi in range(nf):
                for fj in adj.get(fi, []):
                    if (fi, fj) not in seen:
                        seen.add((fi, fj))
                        seen.add((fj, fi))
                        dot = np.clip(np.dot(N[fi], N[fj]), -1, 1)
                        all_angles.append(np.degrees(np.arccos(dot)))
            aa = np.array(all_angles)
            logger.debug("Dihedral angles: min=%.1f, max=%.1f, median=%.1f",
                        aa.min(), aa.max(), np.median(aa))

        # Phase 1: Sharp edge segmentation
        segments = _segment_by_sharp_edges(adj, N, nf, self.sharp_angle)
        curved_segs = [(seg, len(seg)) for seg in segments
                       if len(seg) >= self.min_reg and _segment_is_curved(seg, N)]
        logger.info("Segments (sharp>%.1f deg): %d, curved (>=%d faces): %d",
                   self.sharp_angle, len(segments), self.min_reg, len(curved_segs))

        # Phase 2: Fit cylinders from curved segments
        cyls = []
        for si, (seg, sz) in enumerate(curved_segs):
            cyl = _fit_cylinder(seg, N, C, A, self.min_fq, self.min_arc,
                               self.min_ld, self.verbose)
            if cyl:
                cyls.append(cyl)
                if self.verbose:
                    logger.debug("Segment %d (%d faces): R=%.2f L=%.2f %s",
                                si, sz, cyl['radius'], cyl['length'], cyl['type'])
            elif self.verbose:
                logger.debug("Segment %d (%d faces): not a cylinder", si, sz)

        # Phase 3 (fallback): Classic method for missed cylinders
        if len(cyls) < 2:
            curved = _mark_curved_faces(adj, N, nf, self.min_dih, self.max_dih)
            regions = _grow_regions(adj, curved, nf, self.min_reg)
            if self.verbose:
                logger.debug("Fallback: curved faces=%d, regions=%d",
                            curved.sum(), len(regions))
            for ri, reg in enumerate(regions):
                cyl = _fit_cylinder(reg, N, C, A, self.min_fq, self.min_arc,
                                   self.min_ld, self.verbose)
                if cyl:
                    cyls.append(cyl)

        # Merge duplicates
        before = len(cyls)
        cyls = _merge_cylinders(cyls, self.verbose)
        if before > len(cyls):
            logger.info("Merged: %d -> %d", before, len(cyls))

        logger.info("Detected cylinders: %d", len(cyls))
        return cyls


class SymmetryDetector:
    """Combined symmetry and cylinder detector."""

    def __init__(self, tol_factor: float = 0.01, score_threshold: float = 0.85,
                 n_azimuth: int = 24, n_polar: int = 12, max_cn: int = 8,
                 max_pts: int = MAX_PTS, detect_cylinders: bool = True,
                 verbose: bool = False, min_dih: float = 1.0, max_dih: float = 60.0,
                 min_reg: int = 6, min_arc: float = 45.0, sharp_angle: float = 15.0):
        self.tf = tol_factor
        self.st = score_threshold
        self.naz = n_azimuth
        self.npol = n_polar
        self.mcn = max_cn
        self.mpts = max_pts
        self.dc = detect_cylinders
        self.cd = CylinderDetector(
            verbose=verbose,
            min_dih=min_dih,
            max_dih=max_dih,
            min_reg=min_reg,
            min_arc=min_arc,
            sharp_angle=sharp_angle
        )

    def detect(self, path: str) -> Dict[str, Any]:
        """Detect symmetries and cylinders in STL file.

        Args:
            path: Path to STL file

        Returns:
            Dict with symmetry_axes, rotation_axes, cylinders, tolerance, n_vertices
        """
        logger.info("Loading: %s", path)
        pf = center(load_vertices(path))
        logger.info("Unique vertices: %d", len(pf))

        bd = np.linalg.norm(pf.max(0) - pf.min(0))
        tol = self.tf * bd
        logger.info("Bbox diagonal: %.4f, tolerance: %.4f", bd, tol)

        pts = maybe_subsample(pf, self.mpts)
        if len(pts) < len(pf):
            logger.info("Subsampled: %d -> %d", len(pf), len(pts))
        tree = KDTree(pts)

        cyls: List[Dict] = []
        ca = np.empty((0, 3))

        if self.dc:
            md = load_mesh_data(path)
            rc = md["vectors"].reshape(-1, 3).mean(axis=0)
            cyls = self.cd.detect(md, rc)
            if cyls:
                ca = np.array([c["axis"] for c in cyls])
                ca /= np.linalg.norm(ca, axis=1, keepdims=True)

        pca = pca_axes(pts)
        grid = sphere_candidates(self.naz, self.npol)
        cands = np.vstack([pca, grid])
        if len(ca):
            cands = np.vstack([cands, ca])
            logger.info("Cylinder axes: %d", len(ca))
        cands = np.unique(np.round(cands, 4), axis=0)
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
        logger.info("Candidate axes: %d", len(cands))

        ns = self.mcn
        logger.info("Checking mirror symmetries...")
        ss = self._batch_symmetry_scores(pts, cands, tree, tol)
        sa = [(float(ss[i]), cands[i].copy()) for i in np.where(ss >= self.st)[0]]
        logger.info("Mirror symmetries found: %d", len(sa))

        ra: Dict[int, List] = {}
        for n in range(2, ns + 1):
            logger.debug("Checking C%d rotations...", n)
            rs = self._batch_rotation_scores(pts, cands, n, tree, tol)
            h = [(float(rs[i]), cands[i].copy()) for i in np.where(rs >= self.st)[0]]
            if h:
                ra[n] = h
                logger.info("C%d rotations found: %d", n, len(h))

        return {
            "symmetry_axes": self._dedup(sa),
            "rotation_axes": {n: self._dedup(v) for n, v in ra.items()},
            "cylinders": cyls,
            "tolerance": tol,
            "n_vertices": len(pf)
        }

    def _batch_symmetry_scores(self, pts: np.ndarray, axes: np.ndarray,
                               tree: KDTree, tol: float) -> np.ndarray:
        """Compute symmetry scores for all axes."""
        Na, Np = len(axes), len(pts)
        proj = pts @ axes.T
        refl = pts[:, None, :] - 2.0 * proj[:, :, None] * axes[None, :, :]
        refl = refl.transpose(1, 0, 2).reshape(Na * Np, 3)
        d, _ = tree.query(refl, **_KDT_KW)
        return (d.reshape(Na, Np) < tol).mean(axis=1)

    def _batch_rotation_scores(self, pts: np.ndarray, axes: np.ndarray,
                               n: int, tree: KDTree, tol: float) -> np.ndarray:
        """Compute rotation scores for all axes."""
        Na, Np = len(axes), len(pts)
        R = self._rot_matrices(axes, 2 * np.pi / n)
        rot = np.einsum('aij,pj->api', R, pts).reshape(Na * Np, 3)
        d, _ = tree.query(rot, **_KDT_KW)
        return (d.reshape(Na, Np) < tol).mean(axis=1)

    @staticmethod
    def _rot_matrices(axes: np.ndarray, angle: float) -> np.ndarray:
        """Build rotation matrices for axes."""
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
        oc = 1 - c
        R = np.empty((len(axes), 3, 3))
        R[:, 0, 0] = c + x * x * oc
        R[:, 0, 1] = x * y * oc - z * s
        R[:, 0, 2] = x * z * oc + y * s
        R[:, 1, 0] = y * x * oc + z * s
        R[:, 1, 1] = c + y * y * oc
        R[:, 1, 2] = y * z * oc - x * s
        R[:, 2, 0] = z * x * oc - y * s
        R[:, 2, 1] = z * y * oc + x * s
        R[:, 2, 2] = c + z * z * oc
        return R

    @staticmethod
    def _dedup(items: List[Tuple[float, np.ndarray]],
               ang: float = 10.0) -> List[Tuple[float, np.ndarray]]:
        """Deduplicate axes by angular similarity."""
        if not items:
            return []
        items = sorted(items, key=lambda x: -x[0])
        kept = []
        for sc, ax in items:
            if not any(abs(np.dot(ax, ka)) > np.cos(np.radians(ang)) for _, ka in kept):
                kept.append((sc, ax))
        return kept


# ============================================================================
# Projection Utilities
# ============================================================================

STANDARD_VIEWS = {
    "front": {
        "name": "Front view",
        "normal": np.array([0., 1., 0.]),
        "u_axis": np.array([1., 0., 0.]),
        "v_axis": np.array([0., 0., 1.]),
    },
    "top": {
        "name": "Top view",
        "normal": np.array([0., 0., -1.]),
        "u_axis": np.array([1., 0., 0.]),
        "v_axis": np.array([0., 1., 0.]),
    },
    "left": {
        "name": "Left view",
        "normal": np.array([1., 0., 0.]),
        "u_axis": np.array([0., 1., 0.]),
        "v_axis": np.array([0., 0., 1.]),
    },
}

_PERP_THRESHOLD = 0.95
_PARA_THRESHOLD = 0.15


def compute_cylinder_projections(cylinders: List[Dict],
                                 extension_factor: float = 0.15) -> List[Dict]:
    """Compute annotation elements for cylinders on each view."""
    annotations = []

    for ci, cyl in enumerate(cylinders):
        axis_3d = cyl["axis"]
        center_3d = cyl["center"]
        R = cyl["radius"]
        L = cyl["length"]
        D = 2 * R

        for vk, view in STANDARD_VIEWS.items():
            vn = view["normal"]
            alignment = abs(float(axis_3d @ vn))

            # Project center
            cu = float(center_3d @ view["u_axis"])
            cv = float(center_3d @ view["v_axis"])

            anno = {
                "view": vk,
                "view_name": view["name"],
                "cylinder_id": ci,
                "cylinder_type": cyl["type"],
                "diameter": D,
                "radius": R,
                "length": L,
                "axis_3d": axis_3d.tolist(),
                "center_3d": center_3d.tolist(),
                "alignment": float(alignment),
            }

            if alignment > _PERP_THRESHOLD:
                # End view: crosshair + circle
                ext = R * (1 + extension_factor)
                anno["annotation_type"] = "crosshair"
                anno["crosshair"] = {
                    "center": (cu, cv),
                    "radius": R,
                    "h_line": ((cu - ext, cv), (cu + ext, cv)),
                    "v_line": ((cu, cv - ext), (cu, cv + ext)),
                }
                anno["centerline"] = None
                anno["contour_lines"] = None

            elif alignment < _PARA_THRESHOLD:
                # Side view: centerline + contours
                du = float(axis_3d @ view["u_axis"])
                dv = float(axis_3d @ view["v_axis"])
                proj_len = np.hypot(du, dv)
                if proj_len > 1e-12:
                    du, dv = du / proj_len, dv / proj_len
                else:
                    du, dv = 1.0, 0.0

                half_L = L / 2
                ext_L = L * extension_factor
                total_half = half_L + ext_L

                anno["annotation_type"] = "centerline"
                anno["centerline"] = {
                    "start": (cu - du * total_half, cv - dv * total_half),
                    "end": (cu + du * total_half, cv + dv * total_half),
                    "center": (cu, cv),
                    "body_start": (cu - du * half_L, cv - dv * half_L),
                    "body_end": (cu + du * half_L, cv + dv * half_L),
                    "direction": (du, dv),
                    "length": L,
                    "angle_deg": float(np.degrees(np.arctan2(dv, du))),
                }

                # Contour lines
                perp_u, perp_v = -dv, du
                contours = []
                for sign in (-1, +1):
                    ou, ov = cu + sign * R * perp_u, cv + sign * R * perp_v
                    contours.append({
                        "start": (ou - du * half_L, ov - dv * half_L),
                        "end": (ou + du * half_L, ov + dv * half_L),
                    })
                anno["contour_lines"] = contours
                anno["crosshair"] = None

            else:
                # Oblique view
                du = float(axis_3d @ view["u_axis"])
                dv = float(axis_3d @ view["v_axis"])
                proj_len = np.hypot(du, dv)
                if proj_len > 1e-12:
                    du, dv = du / proj_len, dv / proj_len
                else:
                    du, dv = 1.0, 0.0

                half_L = L / 2
                ext_L = L * extension_factor
                total_half = half_L + ext_L
                r_major = R
                r_minor = R * np.sqrt(1 - alignment ** 2)

                anno["annotation_type"] = "centerline+circle"
                anno["centerline"] = {
                    "start": (cu - du * total_half, cv - dv * total_half),
                    "end": (cu + du * total_half, cv + dv * total_half),
                    "center": (cu, cv),
                    "body_start": (cu - du * half_L, cv - dv * half_L),
                    "body_end": (cu + du * half_L, cv + dv * half_L),
                    "direction": (du, dv),
                    "length": L,
                    "angle_deg": float(np.degrees(np.arctan2(dv, du))),
                }

                ext = r_major * (1 + extension_factor)
                anno["crosshair"] = {
                    "center": (cu, cv),
                    "radius": R,
                    "r_major": float(r_major),
                    "r_minor": float(r_minor),
                    "h_line": ((cu - ext, cv), (cu + ext, cv)),
                    "v_line": ((cu, cv - ext), (cu, cv + ext)),
                }

                perp_u, perp_v = -dv, du
                contours = []
                for sign in (-1, +1):
                    ou = cu + sign * R * perp_u
                    ov = cv + sign * R * perp_v
                    contours.append({
                        "start": (ou - du * half_L, ov - dv * half_L),
                        "end": (ou + du * half_L, ov + dv * half_L),
                    })
                anno["contour_lines"] = contours

            annotations.append(anno)

    return annotations


def compute_all_projections(results: Dict) -> Dict:
    """Compute all annotation elements from detection results."""
    cyls = results.get("cylinders", [])
    cyl_annos = compute_cylinder_projections(cyls)

    by_view: Dict[str, Dict] = {}
    for vk in STANDARD_VIEWS:
        by_view[vk] = {
            "cylinders": [a for a in cyl_annos if a["view"] == vk],
        }

    return {
        "cylinders": cyl_annos,
        "by_view": by_view,
    }
