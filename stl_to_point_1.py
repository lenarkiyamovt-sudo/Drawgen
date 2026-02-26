import os
import math
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from numpy.linalg import norm
from stl import mesh
import svgwrite
from rtree import index
import xml.etree.ElementTree as ET


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("improved_isometric.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GRID_SIZE = 1e-4
EPS_DEPTH = 1e-6
EPS_SEGMENT = 1e-4
ANGLE_THRESHOLD_DEGREES = 10
ANGLE_THRESHOLD = np.cos(np.radians(ANGLE_THRESHOLD_DEGREES))
MIN_EDGE_LENGTH = 1e-5

BASE_MAX_RECURSION_DEPTH = 3

ENABLE_MERGE = True
ENABLE_PRIORITY = True
ENABLE_SHARP_EDGES = True

rtree_lock = threading.Lock()


class VisibilityCache:
    def __init__(self, grid_size=1e-3):
        self.cache = {}
        self.lock = threading.Lock()
        self.grid_size = grid_size

    def _snap(self, coord):
        return tuple(
            (coord_i // self.grid_size) * self.grid_size for coord_i in coord
        )

    def get(self, pt_x, pt_y, pt_z):
        key = self._snap((pt_x, pt_y, pt_z))
        with self.lock:
            return self.cache.get(key, None)

    def set(self, pt_x, pt_y, pt_z, value):
        key = self._snap((pt_x, pt_y, pt_z))
        with self.lock:
            self.cache[key] = value

visibility_cache = VisibilityCache(grid_size=GRID_SIZE)


def safe_rtree_intersection(spatial_idx: index.Index, bounds) -> List[int]:
    with rtree_lock:
        return list(spatial_idx.intersection(bounds))

def is_point_in_triangle(pt: np.ndarray, tri_2d: np.ndarray) -> bool:
    a, b, c = tri_2d
    v0 = c - a
    v1 = b - a
    v2 = pt - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    invd = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invd
    v = (dot00 * dot12 - dot01 * dot02) * invd
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def interpolate_face_z(pt_2d: np.ndarray, tri_2d: np.ndarray, tri_z: np.ndarray) -> float:
    a, b, c = tri_2d
    z_a, z_b, z_c = tri_z
    v0 = c - a
    v1 = b - a
    v2 = pt_2d - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.inf
    invd = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invd
    v = (dot00 * dot12 - dot01 * dot02) * invd
    w = 1 - u - v
    return w * z_a + u * z_c + v * z_b

def calculate_scaled_eps(transformed_with_z: np.ndarray) -> Tuple[float, float]:
    bbox_min = np.min(transformed_with_z[:, :2], axis=0)
    bbox_max = np.max(transformed_with_z[:, :2], axis=0)
    bbox_size = bbox_max - bbox_min
    model_scale = max(bbox_size[0], bbox_size[1])
    scaled_eps_depth = EPS_DEPTH * model_scale
    scaled_eps_segment = EPS_SEGMENT * model_scale
    return scaled_eps_depth, scaled_eps_segment

def is_point_visible_improved(pt_x: float, pt_y: float, pt_z: float,
                              spatial_idx: index.Index,
                              transformed_with_z: np.ndarray,
                              faces: np.ndarray,
                              depth_sign: float,
                              eps_depth: float) -> bool:
    """Check if a projected edge point is visible (not occluded by any face).

    depth_sign: +1 means larger z' is closer to viewer (face occludes when z_face > z_point),
                -1 means smaller z' is closer (face occludes when z_face < z_point).
    """
    cached_result = visibility_cache.get(pt_x, pt_y, pt_z)
    if cached_result is not None:
        return cached_result

    cands = safe_rtree_intersection(spatial_idx, (pt_x, pt_y, pt_x, pt_y))
    visible = True

    for f_idx in cands:
        face = faces[f_idx]
        tri_2d = transformed_with_z[face, :2]
        tri_z = transformed_with_z[face, 2]
        if is_point_in_triangle(np.array([pt_x, pt_y]), tri_2d):
            z_face = interpolate_face_z(np.array([pt_x, pt_y]), tri_2d, tri_z)
            delta_z = z_face - pt_z
            # Face occludes point if it is between the viewer and the point.
            # depth_sign encodes the depth convention for this view.
            if delta_z * depth_sign > eps_depth:
                visible = False
                break

    visibility_cache.set(pt_x, pt_y, pt_z, visible)
    return visible

def dynamic_max_depth(edge: Tuple[int,int],
                      transformed_with_z: np.ndarray,
                      scaled_eps_depth: float) -> int:
    pA = transformed_with_z[edge[0]]
    pB = transformed_with_z[edge[1]]
    length_2d = norm(pB[:2] - pA[:2])
    dz = abs(pB[2] - pA[2])

    required_depth_length = 0
    if length_2d > scaled_eps_depth:
        required_depth_length = int(math.ceil(math.log2(length_2d / scaled_eps_depth)))

    required_depth_z = 0
    if dz > 10 * scaled_eps_depth:
        required_depth_z = 1

    return BASE_MAX_RECURSION_DEPTH + max(required_depth_length, required_depth_z)

def adaptive_sampling_improved(
    edge: Tuple[int,int],
    transformed_with_z: np.ndarray,
    spatial_idx: index.Index,
    faces: np.ndarray,
    t0: float = 0.0,
    t1: float = 1.0,
    depth: int = 0,
    max_depth: int = 8,
    scaled_eps_depth: float = 1e-6,
    scaled_eps_segment: float = 1e-3,
    depth_sign: float = 1.0,
    pt0: Tuple[np.ndarray, float, bool] = None,
    pt1: Tuple[np.ndarray, float, bool] = None
) -> List[Tuple[float, float, bool]]:
    if pt0 is None:
        pA_2d = (1 - t0) * transformed_with_z[edge[0], :2] + t0 * transformed_with_z[edge[1], :2]
        pA_z = (1 - t0) * transformed_with_z[edge[0], 2] + t0 * transformed_with_z[edge[1], 2]
        visA = is_point_visible_improved(pA_2d[0], pA_2d[1], pA_z,
                                         spatial_idx, transformed_with_z, faces,
                                         depth_sign, scaled_eps_depth)
    else:
        visA = pt0[2]
        pA_2d, pA_z = pt0[0], pt0[1]

    if pt1 is None:
        pB_2d = (1 - t1) * transformed_with_z[edge[0], :2] + t1 * transformed_with_z[edge[1], :2]
        pB_z = (1 - t1) * transformed_with_z[edge[0], 2] + t1 * transformed_with_z[edge[1], 2]
        visB = is_point_visible_improved(pB_2d[0], pB_2d[1], pB_z,
                                         spatial_idx, transformed_with_z, faces,
                                         depth_sign, scaled_eps_depth)
    else:
        visB = pt1[2]
        pB_2d, pB_z = pt1[0], pt1[1]

    if depth >= max_depth:
        pA = (1 - t0)*transformed_with_z[edge[0], :2] + t0*transformed_with_z[edge[1], :2]
        pB = (1 - t1)*transformed_with_z[edge[0], :2] + t1*transformed_with_z[edge[1], :2]
        length_2d = norm(pB - pA)
        if length_2d >= scaled_eps_segment:
            return [(t0, t1, visA)]
        else:
            return []

    mid_t = 0.5 * (t0 + t1)
    pM_2d = (1 - mid_t) * transformed_with_z[edge[0], :2] + mid_t * transformed_with_z[edge[1], :2]
    pM_z = (1 - mid_t) * transformed_with_z[edge[0], 2] + mid_t * transformed_with_z[edge[1], 2]
    visM = is_point_visible_improved(pM_2d[0], pM_2d[1], pM_z,
                                     spatial_idx, transformed_with_z, faces,
                                     depth_sign, scaled_eps_depth)

    if visM != visA or visM != visB:
        left = adaptive_sampling_improved(
            edge, transformed_with_z, spatial_idx, faces,
            t0=t0, t1=mid_t,
            depth=depth+1, max_depth=max_depth,
            scaled_eps_depth=scaled_eps_depth,
            scaled_eps_segment=scaled_eps_segment,
            depth_sign=depth_sign,
            pt0=(pA_2d, pA_z, visA),
            pt1=None
        )
        right = adaptive_sampling_improved(
            edge, transformed_with_z, spatial_idx, faces,
            t0=mid_t, t1=t1,
            depth=depth+1, max_depth=max_depth,
            scaled_eps_depth=scaled_eps_depth,
            scaled_eps_segment=scaled_eps_segment,
            depth_sign=depth_sign,
            pt0=None,
            pt1=(pB_2d, pB_z, visB)
        )
        return left + right
    else:
        pA = (1 - t0)*transformed_with_z[edge[0], :2] + t0*transformed_with_z[edge[1], :2]
        pB = (1 - t1)*transformed_with_z[edge[0], :2] + t1*transformed_with_z[edge[1], :2]
        length_2d = norm(pB - pA)
        if length_2d >= scaled_eps_segment:
            return [(t0, t1, visA)]
        else:
            return []

def process_edge_segments_improved(
    edge: Tuple[int,int],
    transformed_with_z: np.ndarray,
    faces: np.ndarray,
    spatial_idx: index.Index,
    depth_sign: float = 1.0
) -> List[Tuple[float, float, bool]]:
    scaled_eps_depth, scaled_eps_segment = calculate_scaled_eps(transformed_with_z)
    dyn_depth = dynamic_max_depth(edge, transformed_with_z, scaled_eps_depth)
    segs = adaptive_sampling_improved(
        edge, transformed_with_z, spatial_idx, faces,
        t0=0.0, t1=1.0,
        depth=0, max_depth=dyn_depth,
        scaled_eps_depth=scaled_eps_depth,
        scaled_eps_segment=scaled_eps_segment,
        depth_sign=depth_sign
    )
    return segs

def project_point_onto_line(pt: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    vec = pt - origin
    return np.dot(vec, direction)

def lines_are_collinear_and_overlap_improved(
    lineA: Tuple[np.ndarray, np.ndarray, str],
    lineB: Tuple[np.ndarray, np.ndarray, str],
    eps: float = 1e-6,
    model_scale: float = 1.0
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    (pA, pB, stA) = lineA
    (pC, pD, stB) = lineB
    vAB = pB - pA
    vCD = pD - pC

    eps_local = eps * model_scale
    lenAB = norm(vAB)
    lenCD = norm(vCD)
    if lenAB < eps_local or lenCD < eps_local:
        return None

    dA = vAB / lenAB
    dB = vCD / lenCD
    dot_ = abs(np.dot(dA, dB))
    if abs(dot_ - 1.0) > 1e-4:
        return None

    vAC = pC - pA
    cross = vAC[0] * dA[1] - vAC[1] * dA[0]
    if abs(cross) > eps_local:
        return None

    tA1 = project_point_onto_line(pA, pA, dA)
    tA2 = project_point_onto_line(pB, pA, dA)
    tB1 = project_point_onto_line(pC, pA, dA)
    tB2 = project_point_onto_line(pD, pA, dA)

    if tA2 < tA1:
        tA1, tA2 = tA2, tA1
    if tB2 < tB1:
        tB1, tB2 = tB2, tB1

    tMin = max(tA1, tB1)
    tMax = min(tA2, tB2)
    if tMax < tMin - eps_local:
        return None

    new_t1 = min(tA1, tB1)
    new_t2 = max(tA2, tB2)
    pMin_2d = pA + new_t1 * dA
    pMax_2d = pA + new_t2 * dA
    return (pMin_2d, pMax_2d)

def merge_collinear_lines_improved(
    lines: List[Tuple[np.ndarray, np.ndarray, str]],
    transformed_with_z: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    bbox_min = np.min(transformed_with_z[:, :2], axis=0)
    bbox_max = np.max(transformed_with_z[:, :2], axis=0)
    bbox_size = bbox_max - bbox_min
    model_scale = max(bbox_size[0], bbox_size[1])

    lines_sorted = sorted(
        lines,
        key=lambda ln: (ln[2], ln[0][0], ln[0][1], ln[1][0], ln[1][1])
    )
    merged = []
    used = [False] * len(lines_sorted)

    for i in range(len(lines_sorted)):
        if used[i]:
            continue
        base = lines_sorted[i]
        pA, pB, styleA = base
        group_segments = [(pA, pB, styleA)]
        used[i] = True

        for j in range(i+1, len(lines_sorted)):
            if used[j]:
                continue
            cand = lines_sorted[j]
            if cand[2] != styleA:
                continue
            merged_res = lines_are_collinear_and_overlap_improved(
                (group_segments[0][0], group_segments[0][1], styleA),
                (cand[0], cand[1], cand[2]),
                eps=1e-6,
                model_scale=model_scale
            )
            if merged_res is not None:
                pMin, pMax = merged_res
                group_segments[0] = (pMin, pMax, styleA)
                used[j] = True

        merged.append(group_segments[0])
    return merged

def _subtract_intervals(base_interval, subtract_list):
    """Subtract a list of intervals from a base interval.

    All intervals are (start, end) with start <= end.
    Returns list of remaining (start, end) intervals.
    """
    result = [base_interval]
    for s_start, s_end in sorted(subtract_list):
        new_result = []
        for r_start, r_end in result:
            if s_end <= r_start or s_start >= r_end:
                # No overlap
                new_result.append((r_start, r_end))
            else:
                # Overlap — split
                if s_start > r_start:
                    new_result.append((r_start, s_start))
                if s_end < r_end:
                    new_result.append((s_end, r_end))
        result = new_result
    return result


def apply_style_priority_improved(
    lines: List[Tuple[np.ndarray, np.ndarray, str]],
    transformed_with_z: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Apply style priority: where hidden and visible lines overlap,
    keep visible and subtract the overlap from hidden.
    Hidden portions that don't overlap with any visible line are preserved.
    """
    bbox_min = np.min(transformed_with_z[:, :2], axis=0)
    bbox_max = np.max(transformed_with_z[:, :2], axis=0)
    model_scale = max((bbox_max - bbox_min)[0], (bbox_max - bbox_min)[1])
    eps_local = 1e-6 * model_scale

    visible_lines = [(pA, pB, st) for pA, pB, st in lines if st == "visible"]
    hidden_lines  = [(pA, pB, st) for pA, pB, st in lines if st == "hidden"]

    result = list(visible_lines)

    for hA, hB, hStyle in hidden_lines:
        h_vec = hB - hA
        h_len = norm(h_vec)
        if h_len < eps_local:
            continue
        h_dir = h_vec / h_len

        # Collect visible intervals that are collinear with this hidden line
        subtract = []
        for vA, vB, vStyle in visible_lines:
            v_vec = vB - vA
            v_len = norm(v_vec)
            if v_len < eps_local:
                continue
            v_dir = v_vec / v_len

            # Check collinearity
            if abs(abs(np.dot(h_dir, v_dir)) - 1.0) > 1e-4:
                continue

            # Check co-linearity (distance from visible line to hidden line direction)
            vAC = vA - hA
            cross = vAC[0] * h_dir[1] - vAC[1] * h_dir[0]
            if abs(cross) > eps_local:
                continue

            # Project visible endpoints onto hidden line's parametric axis
            tV1 = np.dot(vA - hA, h_dir)
            tV2 = np.dot(vB - hA, h_dir)
            if tV1 > tV2:
                tV1, tV2 = tV2, tV1

            # Clip to hidden line range [0, h_len]
            tV1 = max(tV1, 0.0)
            tV2 = min(tV2, h_len)
            if tV2 > tV1 + eps_local:
                subtract.append((tV1, tV2))

        # Subtract visible intervals from hidden line [0, h_len]
        remaining = _subtract_intervals((0.0, h_len), subtract)

        for r_start, r_end in remaining:
            if r_end - r_start < eps_local:
                continue
            rA = hA + h_dir * r_start
            rB = hA + h_dir * r_end
            result.append((rA, rB, "hidden"))

    return result


# ============================================================================
#  MODEL ORIENTATION & BEST VIEW SELECTION
# ============================================================================

def orient_model_by_normals(vertices: np.ndarray,
                            faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orient model using face-normal-area-weighted PCA.

    For manufactured/engineered parts, most surfaces are planar and aligned
    with the part's natural axes.  By computing a scatter matrix from face
    normals weighted by their areas, we find the dominant surface orientations
    — which ARE the natural axes of the part.

    Algorithm:
      1. Compute face normals and areas for all triangles.
      2. Build scatter matrix: M = sum(area_i * n_i * n_i^T)
      3. Eigendecomposition of M -> 3 eigenvectors = natural axes.
      4. Sort by eigenvalue (descending) -> PC0=most surface area.
      5. Apply rotation: project vertices onto eigenvector basis.
      6. Result: PC0 -> X, PC1 -> Y, PC2 -> Z.

    This is far more robust than vertex-PCA for non-convex shapes
    (L-brackets, T-pipes, complex assemblies).

    Returns:
        (oriented_vertices, rotation_matrix_3x3)
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    aabb = verts.max(axis=0) - verts.min(axis=0)
    logger.info("Original AABB: %.1f x %.1f x %.1f", aabb[0], aabb[1], aabb[2])

    # Face normals and areas
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5         # (N_faces,)
    norms = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)  # unit normals

    # Scatter matrix: M = sum(area_i * n_i * n_i^T)
    # Vectorized: M = (norms.T * areas) @ norms
    M = (norms.T * areas) @ norms  # (3x3)

    eigenvalues, eigenvectors = np.linalg.eigh(M)  # ascending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigvecs = eigenvectors[:, idx]  # columns = eigenvectors

    # Ensure right-handedness
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Log alignment info
    world_axes = np.eye(3)
    for i in range(3):
        pc = eigvecs[:, i]
        dots = [abs(np.dot(pc, world_axes[j])) for j in range(3)]
        best_j = int(np.argmax(dots))
        angle_deg = np.degrees(np.arccos(np.clip(dots[best_j], 0, 1)))
        logger.info("  PC%d -> nearest %s (angle=%.1f deg, eigenvalue=%.0f)",
                     i, "XYZ"[best_j], angle_deg, eigenvalues[i])

    # Apply rotation: project onto eigenvector basis
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    oriented = centered @ eigvecs  # (N, 3)

    # Shift min corner to origin
    oriented -= oriented.min(axis=0)

    obb = oriented.max(axis=0) - oriented.min(axis=0)
    logger.info("After normal-PCA: %.1f x %.1f x %.1f (X x Y x Z)", obb[0], obb[1], obb[2])

    aabb_vol = float(np.prod(aabb))
    obb_vol = float(np.prod(obb))
    if aabb_vol > 0:
        ratio = obb_vol / aabb_vol
        logger.info("OBB/AABB volume ratio: %.1f%%", 100.0 * ratio)

    # CRITICAL CHECK: if PCA made the bounding box WORSE or barely better,
    # the model was already well-aligned.  Keep original orientation.
    # This handles models with angled chamfers/fillets that confuse normal-PCA.
    OBB_GAIN_THRESHOLD = 0.95  # PCA must shrink volume by at least 5%
    if aabb_vol > 0 and obb_vol >= aabb_vol * OBB_GAIN_THRESHOLD:
        logger.info("PCA SKIPPED: model already well-aligned (OBB not significantly tighter)")
        # Return original vertices shifted to origin
        result = verts.copy()
        result -= result.min(axis=0)
        return result.astype(np.float32), np.eye(3)

    logger.info("PCA APPLIED: OBB significantly tighter than AABB")
    R = eigvecs.T  # rows = new basis
    return oriented.astype(np.float32), R


def score_view_direction(vertices: np.ndarray,
                         faces: np.ndarray,
                         face_normals: np.ndarray,
                         view_dir: np.ndarray) -> Dict[str, float]:
    """Score a view direction by how much DETAIL it reveals.

    Improved scoring — prioritizes views that show internal features
    (holes, ribs, pockets) over plain flat faces.

    Metrics:
      - visible_area:    sum of projected areas of front-facing triangles
      - n_silhouette:    number of silhouette edges (contour complexity)
      - n_visible_edges: total sharp/feature edges visible from this direction
      - n_distinct_normals: number of distinct face orientations visible
        (more = more geometric features like chamfers, holes)

    Score = visible_area * (1 + 0.3*n_silhouette + 0.5*n_distinct_normals)
    This heavily favors views that show geometric complexity, not just area.
    """
    vd = view_dir / np.linalg.norm(view_dir)
    dots = face_normals @ vd  # (N_faces,)

    # Front-facing: normal points toward viewer (dot < 0)
    front_mask = dots < 0

    # Projected area
    v0 = vertices[faces[front_mask, 0]]
    v1 = vertices[faces[front_mask, 1]]
    v2 = vertices[faces[front_mask, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    proj_areas = np.abs(cross @ vd) * 0.5
    visible_area = float(proj_areas.sum())

    # Silhouette edges
    edge_face_map = defaultdict(list)
    for fi in range(len(faces)):
        f = faces[fi]
        for j in range(3):
            ek = tuple(sorted((int(f[j]), int(f[(j + 1) % 3]))))
            edge_face_map[ek].append(fi)

    n_silhouette = 0
    for ek, flist in edge_face_map.items():
        if len(flist) == 2:
            if dots[flist[0]] * dots[flist[1]] < 0:
                n_silhouette += 1

    # Count distinct normal directions among visible faces
    # (quantized to 15-degree bins to avoid noise)
    front_normals = face_normals[front_mask]
    if len(front_normals) > 0:
        # Quantize normals to bins
        quantized = np.round(front_normals * 4) / 4  # ~15 deg resolution
        unique_normals = set(map(tuple, quantized))
        n_distinct = len(unique_normals)
    else:
        n_distinct = 0

    score = visible_area * (1.0 + 0.3 * n_silhouette + 0.5 * n_distinct)
    return {
        'score': score,
        'visible_area': visible_area,
        'n_front_faces': int(front_mask.sum()),
        'n_silhouette': n_silhouette,
        'n_distinct_normals': n_distinct,
    }


def select_best_front_and_reorient(vertices: np.ndarray,
                                    faces: np.ndarray,
                                    face_normals: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate 6 axis-aligned views, pick the best front, reorient model.

    After reorientation:
      - Best front direction -> -Z  (standard front view in ESKD)
      - Longest remaining dim -> X  (width on drawing)
      - Other remaining dim   -> Y  (height on drawing)

    Returns:
        (reoriented_vertices, reoriented_face_normals)
    """
    candidates = [
        ('front -Z', np.array([0, 0, -1.0])),
        ('front +Z', np.array([0, 0,  1.0])),
        ('front -X', np.array([-1, 0, 0.0])),
        ('front +X', np.array([1, 0,  0.0])),
        ('front -Y', np.array([0, -1, 0.0])),
        ('front +Y', np.array([0, 1,  0.0])),
    ]

    best_name = None
    best_score = -1.0
    best_dir = None

    for name, vdir in candidates:
        info = score_view_direction(vertices, faces, face_normals, vdir)
        logger.info("  %-10s area=%-10.0f sil=%-4d normals=%-3d score=%.0f",
                     name, info['visible_area'], info['n_silhouette'],
                     info.get('n_distinct_normals', 0), info['score'])
        if info['score'] > best_score:
            best_score = info['score']
            best_name = name
            best_dir = vdir.copy()

    logger.info("=> Best front: %s (score=%.0f)", best_name, best_score)

    # ---- Build axis permutation ----
    # front_axis: which world axis the camera looks along
    front_axis = int(np.argmax(np.abs(best_dir)))
    remaining = [i for i in range(3) if i != front_axis]

    bb = vertices.max(axis=0) - vertices.min(axis=0)

    # Longest remaining -> X, other -> Y
    if bb[remaining[0]] >= bb[remaining[1]]:
        right_axis, up_axis = remaining[0], remaining[1]
    else:
        right_axis, up_axis = remaining[1], remaining[0]

    # New basis vectors
    new_x = np.zeros(3)
    new_x[right_axis] = 1.0
    new_y = np.zeros(3)
    new_y[up_axis] = 1.0
    new_z = -best_dir  # camera looks along -Z -> new_Z = -view_dir

    # Ensure right-handed
    if np.dot(np.cross(new_x, new_y), new_z) < 0:
        new_y = -new_y

    R = np.stack([new_x, new_y, new_z], axis=0)

    new_verts = (vertices @ R.T).astype(np.float32)
    new_normals = (face_normals @ R.T).astype(np.float32)

    # Shift min corner to origin
    new_verts -= new_verts.min(axis=0)

    bb_new = new_verts.max(axis=0) - new_verts.min(axis=0)
    logger.info("After front reorient: %.1f x %.1f x %.1f (X=width, Y=height, Z=depth)",
                bb_new[0], bb_new[1], bb_new[2])

    return new_verts, new_normals


class TopologyProcessor:
    def __init__(self, vertices: List[np.ndarray], faces: List[Tuple[int,int,int]]):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.edge_faces = defaultdict(set)
        self.angle_threshold = ANGLE_THRESHOLD
        self.min_edge_length = MIN_EDGE_LENGTH
        self.face_normals = None
        self._build_topology()
        logger.info("Initialized TopologyProcessor")

    def _build_topology(self):
        logger.info("Building topology...")
        for f_idx, fc in enumerate(self.faces):
            for i in range(3):
                v1, v2 = sorted((fc[i], fc[(i+1) % 3]))
                self.edge_faces[(v1, v2)].add(f_idx)
        self.face_normals = self._compute_face_normals()
        logger.info(f"Found {len(self.edge_faces)} edges")

    def _compute_face_normals(self) -> np.ndarray:
        v01 = self.vertices[self.faces[:,1]] - self.vertices[self.faces[:,0]]
        v02 = self.vertices[self.faces[:,2]] - self.vertices[self.faces[:,0]]
        n = np.cross(v01, v02)
        l = norm(n, axis=1)
        mask = l > 1e-12
        n[mask] /= l[mask, None]
        n[~mask] = np.array([0, 0, 1], dtype=np.float32)
        return n

    def process(self) -> List[Tuple[int,int]]:
        edge_items = list(self.edge_faces.items())
        args = []
        for ed, fs in edge_items:
            args.append((ed, fs, self.face_normals, self.vertices,
                         self.angle_threshold, self.min_edge_length))

        valid = []
        self.smooth_edges = []  # Edges rejected by sharp filter (candidates for silhouettes)
        with ThreadPoolExecutor() as exe:
            results = list(exe.map(self._process_edge, args))
        for ed, ish in results:
            if ish:
                valid.append(ed)
            else:
                self.smooth_edges.append(ed)
        return valid

    @staticmethod
    def _process_edge(args):
        (edge, faces_idx, face_normals, vertices, angle_threshold, min_edge_length) = args
        if not ENABLE_SHARP_EDGES:
            return edge, True

        # Граничные рёбра (1 грань) — всегда валидны (контуры отверстий)
        if len(faces_idx) == 1:
            return edge, True

        if len(faces_idx) != 2:
            return edge, True

        fs = list(faces_idx)
        fn0 = face_normals[fs[0]]
        fn1 = face_normals[fs[1]]
        dotp = np.dot(fn0, fn1)
        length = norm(vertices[edge[1]] - vertices[edge[0]])
        is_sharp = (dotp < angle_threshold) and (length >= min_edge_length)
        return edge, is_sharp


class ViewProcessor:
    def __init__(self,
                 vertices: np.ndarray,
                 edges: List[Tuple[int,int]],
                 faces: np.ndarray,
                 edge_faces: Optional[Dict] = None,
                 face_normals: Optional[np.ndarray] = None,
                 smooth_edges: Optional[List[Tuple[int,int]]] = None):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.edge_faces = edge_faces or {}
        self.face_normals = face_normals
        self.smooth_edges = smooth_edges or []

        self.view_matrices = {
            "front": np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=float),
            "top": np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0,-1, 0]
            ], dtype=float),
            "right": np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1,0, 0]
            ], dtype=float),
            "left": np.array([
                [0, 0, -1],
                [0, 1,  0],
                [1, 0,  0]
            ], dtype=float),
            "back": np.array([
                [-1, 0,  0],
                [ 0, 1,  0],
                [ 0, 0, -1]
            ], dtype=float),
            "bottom": np.array([
                [ 1, 0,  0],
                [ 0, 0, -1],
                [ 0, 1,  0]
            ], dtype=float),
            "isometric": np.array([
                [ 0.707, -0.579,  0.406],
                [ 0.707,  0.579, -0.406],
                [ 0.000,  0.574,  0.819]
            ], dtype=float),
        }
        logger.info("Initialized ViewProcessor")

    def get_view_direction(self, view_type: str) -> np.ndarray:
        if view_type == "isometric":
            return np.array([0, 0, -1], dtype=float)
        elif view_type == "front":
            return np.array([0, 0, -1], dtype=float)
        elif view_type == "top":
            return np.array([0, -1, 0], dtype=float)
        elif view_type == "right":
            return np.array([1, 0, 0], dtype=float)
        elif view_type == "left":
            return np.array([-1, 0, 0], dtype=float)
        elif view_type == "back":
            return np.array([0, 0, 1], dtype=float)
        elif view_type == "bottom":
            return np.array([0, 1, 0], dtype=float)
        else:
            return np.array([0, 0, -1], dtype=float)

    def _get_silhouette_edges(self, view_direction: np.ndarray) -> List[Tuple[int,int]]:
        """Detect silhouette edges for the given view direction.

        A silhouette edge is an edge shared by two faces where one face
        points toward the viewer and the other points away.
        """
        if self.face_normals is None or not self.smooth_edges:
            return []

        silhouette = []
        for edge in self.smooth_edges:
            edge_key = tuple(sorted(edge))
            if edge_key not in self.edge_faces:
                continue
            faces_idx = self.edge_faces[edge_key]
            if len(faces_idx) != 2:
                continue
            fs = list(faces_idx)
            # Dot product of each face normal with view direction
            d0 = np.dot(self.face_normals[fs[0]], view_direction)
            d1 = np.dot(self.face_normals[fs[1]], view_direction)
            # Silhouette: one face toward viewer, other away
            # (opposite signs of dot product)
            if d0 * d1 < 0:
                silhouette.append(edge)
            # Also include edges where one face is edge-on (nearly perpendicular)
            elif abs(d0) < 0.01 or abs(d1) < 0.01:
                if abs(d0 - d1) > 0.01:
                    silhouette.append(edge)

        return silhouette

    def transform_to_view(self, pts: np.ndarray, view_type: str) -> np.ndarray:
        if view_type not in self.view_matrices:
            raise ValueError(f"Unknown view type {view_type}")
        return pts @ self.view_matrices[view_type]

    def _compute_depth_sign(self, view_type: str) -> float:
        """Compute depth sign for the given view.

        After view transform, z' represents depth.  We need to know whether
        larger z' means *closer* to the viewer (+1) or *farther* (-1).

        The sign is derived from the z-component of the view direction
        transformed into view coordinates:
          view_dir_view = view_direction_original @ view_matrix
        If the z-component is negative, the viewer looks in -z' → larger z'
        is closer → depth_sign = +1.
        """
        view_dir_orig = self.get_view_direction(view_type)
        view_matrix = self.view_matrices[view_type]
        view_dir_view = view_dir_orig @ view_matrix
        z_comp = view_dir_view[2]
        if abs(z_comp) < 1e-9:
            # Fallback: shouldn't happen with well-formed matrices
            return 1.0
        return -1.0 if z_comp > 0 else 1.0

    def process_view(self, view_type: str, display_mode="hidden_line_dashed"):
        visibility_cache.cache.clear()
        transf_pts = self.transform_to_view(self.vertices, view_type)
        transformed_with_z = np.hstack((transf_pts[:, :2], transf_pts[:, 2:3]))

        r_idx = build_rtree_index(transformed_with_z, self.faces)
        view_dir = self.get_view_direction(view_type)
        depth_sign = self._compute_depth_sign(view_type)

        # Detect silhouette edges for this view and add them to the processing set
        silhouette_edges = self._get_silhouette_edges(view_dir)
        all_edges_for_view = list(self.edges) + silhouette_edges

        edge_segments_map = {}
        lines = []
        for e in all_edges_for_view:
            segs = process_edge_segments_improved(
                tuple(e),
                transformed_with_z,
                self.faces,
                r_idx,
                depth_sign=depth_sign
            )
            edge_segments_map[tuple(e)] = segs

        for e_list in all_edges_for_view:
            ed_tup = tuple(e_list)
            if ed_tup not in edge_segments_map:
                continue
            p0 = transformed_with_z[ed_tup[0]]
            p1 = transformed_with_z[ed_tup[1]]
            segs = edge_segments_map[ed_tup]
            for (ts, te, vis) in segs:
                if ts >= te:
                    continue
                pA = (1 - ts)*p0[:2] + ts*p1[:2]
                pB = (1 - te)*p0[:2] + te*p1[:2]
                style = "visible" if vis else "hidden"
                lines.append((pA, pB, style))

        if ENABLE_MERGE:
            lines = merge_collinear_lines_improved(lines, transformed_with_z)

        if ENABLE_PRIORITY:
            lines = apply_style_priority_improved(lines, transformed_with_z)

        visible_lines = []
        hidden_lines = []
        for (pA, pB, sty) in lines:
            if sty == "visible":
                visible_lines.append((pA, pB))
            else:
                hidden_lines.append((pA, pB))

        return transformed_with_z, visible_lines, hidden_lines


def build_rtree_index(transformed_with_z: np.ndarray, faces: np.ndarray) -> index.Index:
    p = index.Property()
    p.dimension = 2
    r_idx = index.Index(properties=p)

    for f_idx, face in enumerate(faces):
        tri_2d = transformed_with_z[face, :2]
        mi = tri_2d.min(axis=0)
        ma = tri_2d.max(axis=0)
        r_idx.insert(f_idx, (mi[0], mi[1], ma[0], ma[1]))
    return r_idx


def calculate_eskd_line_parameters(front_view_mm: float = 100.0, S: float = None) -> Dict[str, Dict[str, str]]:
    """Calculate ESKD line parameters per GOST 2.303-68.

    Parameters depend on the size of the main view ON PAPER (in mm).
    S — единая толщина основной линии для всего чертежа.

    GOST 2.303-68:
      - S (основная линия): 0.5-1.4 mm — depends on format and complexity
      - Штриховая: толщина S/3..S/2, штрихи 2-8 мм, промежутки 1-2 мм
      - "Длину штрихов следует выбирать в зависимости от величины изображения"

    Practical mapping (front view size -> S and dash parameters):
      < 40mm:   S=0.5, dash=2, gap=1
      40-80mm:  S=0.5, dash=3, gap=1
      80-150mm: S=0.7, dash=4, gap=1.5
      > 150mm:  S=0.7, dash=5, gap=2
    """
    fv = front_view_mm

    # Определяем S если не передан
    if S is None:
        S = 0.7 if fv >= 80 else 0.5

    # Параметры штрихов зависят только от размера изображения
    if fv < 40:
        dash_length = 2.0
        gap_length = 1.0
    elif fv < 80:
        dash_length = 3.0
        gap_length = 1.0
    elif fv < 150:
        dash_length = 4.0
        gap_length = 1.5
    else:
        dash_length = 5.0
        gap_length = 2.0

    thin_width = S / 2.0  # S/2 per GOST 2.303-68 (upper range for readability)

    return {
        'visible': {
            'stroke': 'black',
            'stroke_width': f'{S}mm',
            'stroke_linecap': 'butt'
        },
        'hidden': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_dasharray': f'{dash_length},{gap_length}',
            'stroke_linecap': 'butt'
        },
        'hidden_solid': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt'
        },
        'thin': {
            'stroke': 'black',
            'stroke_width': f'{thin_width:.2f}mm',
            'stroke_linecap': 'butt'
        },
        '_params': {
            'S': S,
            'thin_width': thin_width,
            'dash_length': dash_length,
            'gap_length': gap_length,
        }
    }


class ESKDDrawingSheet:
    """ESKD drawing sheet with GOST 2.301-68 formats and GOST 2.302-68 scales."""

    # GOST 2.301-68: (short_side, long_side) in mm
    FORMATS = {
        'A4': (210, 297),
        'A3': (297, 420),
        'A2': (420, 594),
        'A1': (594, 841),
        'A0': (841, 1189),
    }
    FORMATS_ORDERED = ['A4', 'A3', 'A2', 'A1', 'A0']

    # GOST 2.302-68: standard reduction scales, sorted descending
    # 1:1, 1:2, 1:2.5, 1:4, 1:5, 1:10, 1:15, 1:20, 1:25, 1:40, 1:50, 1:75, 1:100...
    GOST_REDUCTION_SCALES = [
        1.0,                            # 1:1
        0.5,                            # 1:2
        1.0/2.5,                        # 1:2.5  = 0.4
        0.25,                           # 1:4
        0.2,                            # 1:5
        0.1,                            # 1:10
        1.0/15,                         # 1:15
        0.05,                           # 1:20
        1.0/25,                         # 1:25   = 0.04
        1.0/40,                         # 1:40   = 0.025
        0.02,                           # 1:50
        1.0/75,                         # 1:75
        0.01,                           # 1:100
        1.0/200,                        # 1:200
        1.0/400,                        # 1:400
        1.0/500,                        # 1:500
        1.0/800,                        # 1:800
        0.001,                          # 1:1000
    ]

    # Minimum front view size on sheet (mm)
    MIN_MAIN_VIEW_MM = 40.0

    # Margins: left=20 (binding), others=5.  Title block: 185x55 bottom-right.
    MARGIN_LEFT = 20.0
    MARGIN_OTHER = 5.0
    TITLE_BLOCK_W = 185.0
    TITLE_BLOCK_H = 55.0

    # ГОСТ 2.303-68: толщина основной линии (мм).
    # Значение по умолчанию; пересчитывается в generate_drawing()
    # по размеру главного вида: <80мм → 0.5, ≥80мм → 0.7.
    # Используется для линий видов (контуры, штриховые).
    S = 0.5
    # Толщина линий рамки и штампа (мм).
    # Постоянна и не зависит от масштаба чертежа.
    # ГОСТ 2.104-2006: рамка = основная линия, тонкие разделители = S_FRAME/3.
    S_FRAME = 0.25
    # Высота шрифта подписей в штампе (мм), ГОСТ 2.304-81 тип Б
    STAMP_FONT_H = 1.4

    def __init__(self, format_name=None, orientation='landscape'):
        self.views_data = {}
        self.scale = 1.0
        self.eskd_styles = None

        if format_name and format_name in self.FORMATS:
            self._set_format(format_name, orientation)
        else:
            # Will be set by auto_select_format_and_scale()
            self.format_name = None
            self.orientation = orientation
            self.width = 0
            self.height = 0

    def _set_format(self, format_name, orientation='landscape'):
        """Set sheet dimensions for given format."""
        short, long = self.FORMATS[format_name]
        self.format_name = format_name
        self.orientation = orientation
        if orientation == 'landscape':
            self.width, self.height = long, short
        else:
            self.width, self.height = short, long
        logger.info("Sheet: %s %s %dx%d mm", format_name, orientation,
                     self.width, self.height)

    def _get_available_area(self):
        """Usable area for views (mm), accounting for margins and title block."""
        w = self.width - self.MARGIN_LEFT - self.MARGIN_OTHER
        h = self.height - 2 * self.MARGIN_OTHER - self.TITLE_BLOCK_H
        return w, h

    @property
    def margin(self):
        return 10  # legacy compat
        
    def add_view_data(self, view_name, transformed_coords, visible_lines, hidden_lines):
        self.views_data[view_name] = {
            'coords': transformed_coords,
            'visible': visible_lines,
            'hidden': hidden_lines,
            'bbox': self._calculate_bbox(transformed_coords)
        }

    # ============================================================
    #  Алгоритм исключения лишних видов
    #  ГОСТ 2.305-2008: «Количество изображений должно быть
    #  наименьшим, но обеспечивающим полное представление
    #  о предмете»
    # ============================================================

    # Противоположные пары видов
    OPPOSITE_PAIRS = [
        ('front', 'back'),
        ('top',   'bottom'),
        ('left',  'right'),
    ]

    # Какие оси покрывает каждый вид (для контроля полноты)
    AXIS_MAP = {
        'front':  frozenset({'X', 'Y'}),
        'back':   frozenset({'X', 'Y'}),
        'top':    frozenset({'X', 'Z'}),
        'bottom': frozenset({'X', 'Z'}),
        'left':   frozenset({'Y', 'Z'}),
        'right':  frozenset({'Y', 'Z'}),
    }

    @staticmethod
    def _normalize_edge(pA, pB, precision=4):
        """Нормализует ребро в каноническую форму (отсортированные конечные точки)."""
        a = (round(pA[0], precision), round(pA[1], precision))
        b = (round(pB[0], precision), round(pB[1], precision))
        return (a, b) if a <= b else (b, a)

    def _edge_set(self, edges, precision=4):
        """Множество нормализованных рёбер."""
        return frozenset(
            self._normalize_edge(pA, pB, precision) for pA, pB in edges
        )

    def _is_simple_rectangle(self, view_name):
        """Вид — простой прямоугольник без внутренних рёбер?

        Проверяет: все видимые рёбра лежат на контуре bbox,
        и скрытых рёбер нет (или тоже на контуре).
        Такой вид не несёт дополнительной информации о форме.
        """
        vdata = self.views_data[view_name]
        bb = vdata['bbox']
        tol = (bb['width'] + bb['height']) * 0.005  # 0.5% от размера

        def on_contour(pA, pB):
            ax, ay = pA[0], pA[1]
            bx, by = pB[0], pB[1]
            # Ребро на контуре, если обе точки на одной стороне bbox
            on_left   = abs(ax - bb['min_x']) < tol and abs(bx - bb['min_x']) < tol
            on_right  = abs(ax - bb['max_x']) < tol and abs(bx - bb['max_x']) < tol
            on_top    = abs(ay - bb['min_y']) < tol and abs(by - bb['min_y']) < tol
            on_bottom = abs(ay - bb['max_y']) < tol and abs(by - bb['max_y']) < tol
            return on_left or on_right or on_top or on_bottom

        for pA, pB in vdata['visible']:
            if not on_contour(pA, pB):
                return False
        # Скрытые линии тоже не должны нести информации
        if len(vdata['hidden']) > 0:
            return False
        return True

    def _mirror_edges_x(self, edge_set, bbox):
        """Зеркально отражает набор рёбер по оси X (для сравнения left↔right)."""
        cx = (bbox['min_x'] + bbox['max_x']) / 2
        mirrored = set()
        for (ax, ay), (bx, by) in edge_set:
            ma = (round(2*cx - ax, 4), ay)
            mb = (round(2*cx - bx, 4), by)
            mirrored.add((ma, mb) if ma <= mb else (mb, ma))
        return frozenset(mirrored)

    def _mirror_edges_y(self, edge_set, bbox):
        """Зеркально отражает набор рёбер по оси Y (для сравнения top↔bottom)."""
        cy = (bbox['min_y'] + bbox['max_y']) / 2
        mirrored = set()
        for (ax, ay), (bx, by) in edge_set:
            ma = (ax, round(2*cy - ay, 4))
            mb = (bx, round(2*cy - by, 4))
            mirrored.add((ma, mb) if ma <= mb else (mb, ma))
        return frozenset(mirrored)

    def _view_info_score(self, view_name):
        """Информационный счёт вида.

        Чем выше — тем больше уникальной информации несёт вид.
        - Видимые рёбра ценнее скрытых (чертёж читается проще)
        - Внутренние рёбра (не на контуре) ценнее контурных
        """
        vdata = self.views_data[view_name]
        n_vis = len(vdata['visible'])
        n_hid = len(vdata['hidden'])
        # Штраф за «пустой» вид
        if self._is_simple_rectangle(view_name):
            return 0.0
        return n_vis * 1.0 + n_hid * 0.3

    def _edges_similarity(self, set_a, set_b):
        """Jaccard similarity двух множеств рёбер (0..1)."""
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 1.0

    def select_necessary_views(self):
        """Определяет минимальный набор видов по ГОСТ 2.305-2008.

        Алгоритм:
        1. Фронтальный вид — всегда обязателен
        2. Для каждой противоположной пары (фронт/зад, верх/низ, лево/право):
           - Противоположные виды содержат одну и ту же геометрию,
             только visible↔hidden меняются местами
           - Оставляем тот, у которого больше видимых линий
           - Противоположный исключаем
        3. Проверяем «пустые» виды (простой прямоугольник без внутренних рёбер)
        4. Проверяем зеркальную симметрию: если left ≈ mirror(right),
           один из них избыточен
        5. Гарантируем покрытие всех 3 осей (X, Y, Z)
        6. Проверяем: если все рёбра вида уже видны на других выбранных
           видах, вид можно исключить

        Returns:
            list[str] — имена видов для включения в чертёж
            list[tuple] — причины исключения: (view_name, reason)
        """
        if len(self.views_data) <= 3:
            return list(self.views_data.keys()), []

        excluded = {}   # view_name → reason
        scores = {}

        # Подготовка: множества рёбер для каждого вида
        edge_sets = {}
        for vname, vdata in self.views_data.items():
            vis = self._edge_set(vdata['visible'])
            hid = self._edge_set(vdata['hidden'])
            edge_sets[vname] = {
                'vis': vis, 'hid': hid, 'all': vis | hid,
                'n_vis': len(vis), 'n_hid': len(hid),
            }
            scores[vname] = self._view_info_score(vname)

        logger.info("View scores: %s",
                     {v: f"{s:.1f}" for v, s in sorted(scores.items())})

        # --- Шаг 1: Исключить противоположные виды ---
        # Фронтальный вид ВСЕГДА обязателен (ГОСТ 2.305-2008 п.5.1)
        for primary, opposite in self.OPPOSITE_PAIRS:
            if primary not in edge_sets or opposite not in edge_sets:
                continue
            if primary in excluded or opposite in excluded:
                continue

            ep = edge_sets[primary]
            eo = edge_sets[opposite]

            # front никогда не исключается — если в паре, всегда побеждает
            if primary == 'front':
                preferred, secondary = primary, opposite
            elif opposite == 'front':
                preferred, secondary = opposite, primary
            elif ep['n_vis'] >= eo['n_vis']:
                preferred, secondary = primary, opposite
            else:
                preferred, secondary = opposite, primary

            # Оба пусты? Исключить secondary
            if scores[secondary] == 0 and scores[preferred] == 0:
                excluded[secondary] = (
                    "pustoj pryamougolnik (para %s/%s)" % (primary, opposite))
                continue

            # Если у secondary значительно меньше информации
            if scores[preferred] > 0 and scores[secondary] / max(scores[preferred], 1) < 0.3:
                excluded[secondary] = (
                    "malo informacii: %.0f vs %.0f (%s)" % (
                        scores[secondary], scores[preferred], preferred))
                continue

            # Противоположный вид — исключаем secondary
            excluded[secondary] = (
                "protivopolozhnyj vid k %s (vis: %d vs %d)" % (
                    preferred, ep['n_vis'], eo['n_vis']))

        # --- Шаг 2: Исключить пустые виды ---
        for vname in list(self.views_data.keys()):
            if vname in excluded or vname == 'front':
                continue
            if self._is_simple_rectangle(vname):
                excluded[vname] = "prostoj pryamougolnik bez vnutrennih ryober"

        # --- Шаг 3: Проверить зеркальную симметрию оставшихся ---
        remaining = [v for v in self.views_data if v not in excluded]
        SYM_PAIRS = [('left', 'right'), ('top', 'bottom')]
        for va, vb in SYM_PAIRS:
            if va not in remaining or vb not in remaining:
                continue
            es_a = edge_sets[va]['all']
            es_b = edge_sets[vb]['all']
            bb_a = self.views_data[va]['bbox']

            # Зеркалим A и сравниваем с B
            if va in ('left', 'right'):
                mirrored_a = self._mirror_edges_x(es_a, bb_a)
            else:
                mirrored_a = self._mirror_edges_y(es_a, bb_a)

            similarity = self._edges_similarity(mirrored_a, es_b)
            if similarity > 0.85:
                # Виды зеркально похожи — оставляем с бо́льшим score
                if scores[va] <= scores[vb]:
                    loser = va
                else:
                    loser = vb
                excluded[loser] = (
                    f"zerkalno simmetrich {va if loser == vb else vb} "
                    f"(similarity={similarity:.0%})")

        # --- Шаг 4: Гарантировать покрытие всех 3 осей ---
        selected = [v for v in self.views_data if v not in excluded]
        covered_axes = set()
        for v in selected:
            covered_axes |= self.AXIS_MAP.get(v, set())

        if len(covered_axes) < 3:
            missing = {'X', 'Y', 'Z'} - covered_axes
            # Вернуть ранее исключённый вид, покрывающий недостающую ось
            candidates = sorted(excluded.keys(),
                                key=lambda v: -scores.get(v, 0))
            for v in candidates:
                if self.AXIS_MAP.get(v, set()) & missing:
                    logger.info("  Restored %s (axes %s)", v,
                                self.AXIS_MAP.get(v))
                    del excluded[v]
                    covered_axes |= self.AXIS_MAP.get(v, set())
                    missing = {'X', 'Y', 'Z'} - covered_axes
                    if not missing:
                        break

        # --- Шаг 5: Финальная проверка информационной избыточности ---
        selected = [v for v in self.views_data if v not in excluded]
        # Для каждого не-фронтального вида: если его score минимален
        # и другие виды покрывают его оси, можно попробовать убрать
        for v in sorted(selected, key=lambda x: scores.get(x, 0)):
            if v == 'front':
                continue
            if len(selected) <= 2:
                break
            other = [o for o in selected if o != v]
            other_axes = set()
            for o in other:
                other_axes |= self.AXIS_MAP.get(o, set())
            if len(other_axes) >= 3 and scores.get(v, 0) < 2.0:
                excluded[v] = f"izbytoch (score={scores[v]:.1f}, axes covered)"
                selected.remove(v)

        # Результат
        selected = [v for v in ['front', 'top', 'bottom', 'right', 'left', 'back']
                     if v in self.views_data and v not in excluded]
        reasons = [(v, r) for v, r in sorted(excluded.items())]

        logger.info("Selected %d views: %s", len(selected), selected)
        for v, r in reasons:
            logger.info("  X %s: %s", v, r)

        return selected, reasons
        
    def _calculate_bbox(self, transformed_coords):
        min_x = float(np.min(transformed_coords[:, 0]))
        max_x = float(np.max(transformed_coords[:, 0]))
        min_y = float(np.min(transformed_coords[:, 1]))
        max_y = float(np.max(transformed_coords[:, 1]))
        
        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
        
    def _compute_layout_dims_model(self):
        """Compute required layout dimensions in MODEL units (before scaling).

        ГОСТ 2.305-2008, метод первого угла — любой подмножество стандартных видов:
                      [bottom]
          [right] [front] [left] [back]
                      [top]

        Работает корректно для любого набора присутствующих видов.
        Returns (required_width, required_height) in model units.
        Spacing (mm) добавляется отдельно при расчёте масштаба.
        """
        def bw(name):
            return self.views_data[name]['bbox']['width'] if name in self.views_data else 0.0
        def bh(name):
            return self.views_data[name]['bbox']['height'] if name in self.views_data else 0.0

        # По горизонтали: right + front + left + back (те, что есть)
        model_w = bw('right') + (bw('front') or 1.0) + bw('left') + bw('back')
        # По вертикали: bottom + front + top (те, что есть)
        model_h = bh('bottom') + (bh('front') or 1.0) + bh('top')
        return (model_w, model_h)

    def _front_view_size_mm(self, scale):
        """Size of the front view in mm at given scale."""
        if 'front' not in self.views_data:
            # Fallback: use largest available view
            if not self.views_data:
                return 50.0
            best = max(self.views_data.values(),
                       key=lambda v: v['bbox']['width'] * v['bbox']['height'])
            fw = best['bbox']['width'] * scale
            fh = best['bbox']['height'] * scale
            return min(fw, fh)
        fw = self.views_data['front']['bbox']['width'] * scale
        fh = self.views_data['front']['bbox']['height'] * scale
        return min(fw, fh)

    @classmethod
    def _sensible_scale(cls, working_scale: float) -> float:
        """Snap working_scale to nearest GOST 2.302-68 standard scale (<=working_scale).
        
        Simply picks the largest standard scale that fits.
        The discrete GOST scale steps already provide natural margins.
        """
        scales = cls.GOST_REDUCTION_SCALES  # sorted descending
        for s in scales:
            if s <= working_scale * (1.0 + 1e-9):
                return s
        return scales[-1]

    def auto_select_format_and_scale(self):
        """Auto-select GOST format and scale to MAXIMIZE view size.

        Strategy — maximize scale (= best detail visibility):
          For each format A4 -> A0:
            compute max GOST scale that fits
          Pick the (format, scale) combo with the LARGEST scale.
          If two formats give the same scale — prefer the smaller format.
        """
        model_w, model_h = self._compute_layout_dims_model()
        spacing_mm = 30.0
        # Число зазоров зависит от реально присутствующих видов
        horiz = ['right', 'front', 'left', 'back']
        vert  = ['bottom', 'front', 'top']
        n_gaps_w = max(0, sum(1 for v in horiz if v in self.views_data) - 1)
        n_gaps_h = max(0, sum(1 for v in vert  if v in self.views_data) - 1)

        logger.info("Layout model dims: %.0f x %.0f (model units)", model_w, model_h)

        best_format = None
        best_scale = 0.0

        for fmt in self.FORMATS_ORDERED:
            short, long = self.FORMATS[fmt]
            sheet_w, sheet_h = long, short  # landscape

            avail_w = sheet_w - self.MARGIN_LEFT - self.MARGIN_OTHER
            avail_h = sheet_h - 2 * self.MARGIN_OTHER - self.TITLE_BLOCK_H

            views_w = avail_w - n_gaps_w * spacing_mm
            views_h = avail_h - n_gaps_h * spacing_mm

            if views_w <= 0 or views_h <= 0:
                continue

            raw_scale = min(views_w / model_w, views_h / model_h)
            gost_scale = self._sensible_scale(raw_scale)

            # Verify it actually fits (GOST snap should guarantee this, but check)
            layout_w = model_w * gost_scale + n_gaps_w * spacing_mm
            layout_h = model_h * gost_scale + n_gaps_h * spacing_mm
            if layout_w > avail_w + 0.1 or layout_h > avail_h + 0.1:
                continue

            front_mm = self._front_view_size_mm(gost_scale)
            inv_gost = 1.0 / gost_scale if gost_scale > 0 else 9999
            fill_pct = (layout_w * layout_h) / (avail_w * avail_h) * 100

            logger.info("  %s: avail %dx%d, GOST 1:%.4g, front=%.1fmm, fill=%.0f%%",
                         fmt, int(avail_w), int(avail_h),
                         inv_gost, front_mm, fill_pct)

            # Pick this combo if scale is LARGER than current best
            # (= views will be bigger = more detail visible)
            if gost_scale > best_scale:
                best_scale = gost_scale
                best_format = fmt

        if best_format is None:
            best_format = 'A0'
            short, long = self.FORMATS['A0']
            avail_w = long - self.MARGIN_LEFT - self.MARGIN_OTHER
            avail_h = short - 2 * self.MARGIN_OTHER - self.TITLE_BLOCK_H
            raw = min((avail_w - n_gaps_w * spacing_mm) / model_w,
                      (avail_h - n_gaps_h * spacing_mm) / model_h)
            best_scale = self._sensible_scale(raw)

        self._set_format(best_format)
        self.scale = best_scale
        inv_s = 1.0 / best_scale if best_scale > 0 else 9999
        logger.info("=> Selected: %s  scale 1:%.4g  (front view %.1f mm)",
                     best_format, inv_s, self._front_view_size_mm(best_scale))

    def calculate_optimal_scale(self):
        """Calculate scale for current format (if format already set)."""
        if self.format_name is None:
            self.auto_select_format_and_scale()
            return self.scale

        model_w, model_h = self._compute_layout_dims_model()
        spacing_mm = 30.0
        avail_w, avail_h = self._get_available_area()
        horiz = ['right', 'front', 'left', 'back']
        vert  = ['bottom', 'front', 'top']
        n_gaps_w = max(0, sum(1 for v in horiz if v in self.views_data) - 1)
        n_gaps_h = max(0, sum(1 for v in vert  if v in self.views_data) - 1)
        views_w = avail_w - n_gaps_w * spacing_mm
        views_h = avail_h - n_gaps_h * spacing_mm

        if model_w <= 0 or model_h <= 0:
            return 1.0

        raw = min(views_w / model_w, views_h / model_h)
        return self._sensible_scale(raw)
    
    def _make_layout_entry(self, view_name, x, y):
        bbox = self.views_data[view_name]['bbox']
        w = bbox['width'] * self.scale
        h = bbox['height'] * self.scale
        return {
            'x': x, 'y': y,
            'width': w, 'height': h,
            'center_x': x + w / 2,
            'center_y': y + h / 2,
            'offset_x': -bbox['min_x'] * self.scale,
            'offset_y': -bbox['min_y'] * self.scale
        }

    def arrange_views(self):
        """Расставляет виды по ГОСТ 2.305-2008, метод первого угла.

        Работает с ЛЮБЫМ подмножеством стандартных видов:
                      [bottom]
          [right] [front] [left] [back]
                      [top]

        Каждый присутствующий вид всегда занимает свою ГОСТ-позицию
        относительно фронтального вида.  Если какого-то вида нет,
        его место не занято, но позиции соседних видов не смещаются.
        """
        if not self.views_data:
            return {}

        self.scale = self.calculate_optimal_scale()
        layout = {}
        spacing = 30.0  # мм между видами
        s = self.scale

        # --- вспомогательные размеры (0 если вид отсутствует) ---
        def W(name):
            return self.views_data[name]['bbox']['width']  * s if name in self.views_data else 0.0
        def H(name):
            return self.views_data[name]['bbox']['height'] * s if name in self.views_data else 0.0

        front_w  = W('front')  or 1.0
        front_h  = H('front')  or 1.0
        right_w  = W('right')
        left_w   = W('left')
        back_w   = W('back')
        top_h    = H('top')
        bottom_h = H('bottom')

        # --- ширина и высота блока всех видов (без зазоров) ---
        row_views   = [n for n in ['right', 'front', 'left', 'back'] if n in self.views_data]
        col_views   = [n for n in ['bottom', 'front', 'top']          if n in self.views_data]
        n_gaps_w    = max(0, len(row_views) - 1)
        n_gaps_h    = max(0, len(col_views) - 1)

        block_w = right_w + front_w + left_w + back_w + n_gaps_w * spacing
        block_h = bottom_h + front_h + top_h + n_gaps_h * spacing

        # --- размещаем блок в рабочей области листа ---
        usable_w, usable_h = self._get_available_area()
        block_x = self.MARGIN_LEFT   + max(0.0, (usable_w - block_w) / 2)
        block_y = self.MARGIN_OTHER  + max(0.0, (usable_h - block_h) / 2)

        # --- x-координата левого края фронтального вида ---
        # Слева от него стоит right (если есть) с зазором
        front_x = block_x + right_w + (spacing if 'right' in self.views_data else 0.0)
        # --- y-координата верхнего края фронтального вида ---
        # Сверху него стоит bottom (если есть) с зазором
        front_y = block_y + bottom_h + (spacing if 'bottom' in self.views_data else 0.0)

        # --- ФРОНТАЛЬНЫЙ ---
        if 'front' in self.views_data:
            layout['front'] = self._make_layout_entry('front', front_x, front_y)

        # --- ВИД СПРАВА — ЛЕВЕЕ фронтального (1-й угол) ---
        if 'right' in self.views_data:
            right_h = H('right')
            rx = block_x
            ry = front_y + (front_h - right_h) / 2
            layout['right'] = self._make_layout_entry('right', rx, ry)

        # --- ВИД СЛЕВА — ПРАВЕЕ фронтального (1-й угол) ---
        if 'left' in self.views_data:
            left_h = H('left')
            lx = front_x + front_w + spacing
            ly = front_y + (front_h - left_h) / 2
            layout['left'] = self._make_layout_entry('left', lx, ly)

        # --- ВИД СЗАДИ — правее вида слева ---
        if 'back' in self.views_data:
            back_h = H('back')
            # back_x: после left (если есть), иначе после front
            if 'left' in self.views_data:
                bx = front_x + front_w + spacing + left_w + spacing
            else:
                bx = front_x + front_w + spacing
            by = front_y + (front_h - back_h) / 2
            layout['back'] = self._make_layout_entry('back', bx, by)

        # --- ВИД СВЕРХУ — ПОД фронтальным (1-й угол) ---
        if 'top' in self.views_data:
            top_w = W('top')
            tx = front_x + (front_w - top_w) / 2
            ty = front_y + front_h + spacing
            layout['top'] = self._make_layout_entry('top', tx, ty)

        # --- ВИД СНИЗУ — НАД фронтальным (1-й угол) ---
        if 'bottom' in self.views_data:
            bot_w = W('bottom')
            btx = front_x + (front_w - bot_w) / 2
            bty = block_y
            layout['bottom'] = self._make_layout_entry('bottom', btx, bty)

        return layout

    
    def add_projection_lines(self, dwg, layout):
        connections = [
            ('front', 'top', 'vertical'),
            ('front', 'bottom', 'vertical'),
            ('front', 'left', 'horizontal'),
            ('front', 'right', 'horizontal'),
            ('right', 'back', 'horizontal'),
        ]
        for view_a, view_b, direction in connections:
            if view_a not in layout or view_b not in layout:
                continue
            la = layout[view_a]
            lb = layout[view_b]
            if direction == 'vertical':
                cx = la['center_x']
                if la['y'] > lb['y']:
                    y_start = lb['y'] + lb['height'] + 2
                    y_end = la['y'] - 2
                else:
                    y_start = la['y'] + la['height'] + 2
                    y_end = lb['y'] - 2
                dwg.add(dwg.line(
                    start=(cx, y_start), end=(cx, y_end),
                    **self.eskd_styles['projection']
                ))
            else:
                cy = la['center_y']
                if la['x'] > lb['x']:
                    x_start = lb['x'] + lb['width'] + 2
                    x_end = la['x'] - 2
                else:
                    x_start = la['x'] + la['width'] + 2
                    x_end = lb['x'] - 2
                dwg.add(dwg.line(
                    start=(x_start, cy), end=(x_end, cy),
                    **self.eskd_styles['projection']
                ))
    
    def add_frame(self, dwg):
        """ГОСТ 2.104-2006: рамка чертежа. Поля: слева 20мм, остальные 5мм."""
        ml = self.MARGIN_LEFT   # 20
        mo = self.MARGIN_OTHER  # 5
        w, h = self.width, self.height
        S = self.S_FRAME  # толщина линий рамки

        # Внешняя граница листа — тонкая (S/4 ≈ 0.09мм)
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h),
                          fill='none', stroke='black',
                          stroke_width=f'{S/4:.2f}mm'))
        # Внутренняя рамка — основная линия (S = 0.35мм)
        dwg.add(dwg.rect(insert=(ml, mo), size=(w - ml - mo, h - 2*mo),
                          fill='none', stroke='black',
                          stroke_width=f'{S}mm'))

    # ================================================================ #
    #  ГОСТ 2.104-2006 Форма 1 — Основная надпись 185 × 55 мм         #
    #  Транслировано из OpenSCAD Ugl_shtamp_1                          #
    #                                                                    #
    #  Толщины линий (OpenSCAD: s=0.5):                                 #
    #    S   = 0.5мм — основная (контур, структура)                     #
    #    S/3 = 0.17мм — тонкая (внутренние разделители)                 #
    #                                                                    #
    #  Координаты OpenSCAD: x→право, y→верх (y=0 = низ штампа)         #
    #  SVG: x→право, y→вниз → svgY = y0 + 55 - openscadY              #
    # ================================================================ #

    def _line(self, g, dwg, x1, y1, x2, y2, thick=False):
        S = self.S_FRAME
        sw = S if thick else round(S/3, 2)
        g.add(dwg.line(start=(x1, y1), end=(x2, y2),
                       stroke='black', stroke_width=f'{sw}mm'))

    def _text(self, g, dwg, x, y, text, size=None, anchor='start'):
        """Render stamp text.

        y — SVG-координата ЦЕНТРА ячейки (в мм).
        Функция сдвигает baseline на 0.35*size вниз, чтобы текст
        оказался визуально по центру без использования dominant-baseline
        (ненадёжен в большинстве SVG-рендереров).
        """
        if size is None:
            size = self.STAMP_FONT_H
        # baseline = center + 0.35*em  (cap-height ≈ 0.7*em → center at 0.35*em)
        y_bl = y + size * 0.35
        g.add(dwg.text(text, insert=(x, y_bl),
                       font_family='ISOCPEUR, Arial, sans-serif',
                       font_style='italic', font_weight='normal',
                       font_size=f'{size}mm', text_anchor=anchor,
                       fill='black'))

    def add_title_block(self, dwg):
        """ГОСТ 2.104-2006 Форма 1 — основная надпись 185×55 мм.
        Прямая трансляция OpenSCAD Ugl_shtamp_1.

        Толщины из OpenSCAD:
          Горизонтальные (левый блок): y=30,35 → S; остальные → S/3
          Горизонтальные (правый блок): y=15,40 (120мм), y=20,35 (50мм) → S
          Вертикальные: x=0,17,40,55,65,7,135,150,155,167 → S
                        x=140,145 → S/3
        """
        mo = self.MARGIN_OTHER
        w, h = self.width, self.height
        S = self.S_FRAME  # линии штампа

        x0 = w - mo - 185.0
        y0 = h - mo - 55.0

        def X(ox): return x0 + ox
        def Y(oy): return y0 + 55.0 - oy

        g = dwg.g(id='title-block')

        # ========== КОНТУР ==========
        g.add(dwg.rect(insert=(x0, y0), size=(185, 55),
                        fill='white', stroke='black',
                        stroke_width=f'{S}mm'))

        # ========== ГОРИЗОНТАЛЬНЫЕ ЛИНИИ ==========

        # Левый блок (x=0..65): каждые 5мм
        # OpenSCAD: (y==30)||(y==35) ? s : s/3
        for oy in range(5, 55, 5):
            thick = oy in (30, 35)
            self._line(g, dwg, X(0), Y(oy), X(65), Y(oy), thick)

        # Правый блок: y=15, 40 → S, 120мм (x=65..185)
        for oy in (15, 40):
            self._line(g, dwg, X(65), Y(oy), X(185), Y(oy), True)

        # Правый блок: y=20, 35 → S, 50мм (x=135..185)
        for oy in (20, 35):
            self._line(g, dwg, X(135), Y(oy), X(185), Y(oy), True)

        # ========== ВЕРТИКАЛЬНЫЕ ЛИНИИ ==========

        # x=0, 17, 40, 55, 65: S, полная высота (y=0..55)
        # x=0 уже в rect; x=65 — главный разделитель
        for ox in (17, 40, 55):
            self._line(g, dwg, X(ox), Y(0), X(ox), Y(55), True)
        self._line(g, dwg, X(65), Y(0), X(65), Y(55), True)

        # x=7: S, y=30..55 (верхняя часть — заголовок Изм/Лист/...)
        self._line(g, dwg, X(7), Y(30), X(7), Y(55), True)

        # x=135: S, y=0..40
        self._line(g, dwg, X(135), Y(0), X(135), Y(40), True)

        # x=140, 145: S/3 (тонкие), y=20..35
        self._line(g, dwg, X(140), Y(20), X(140), Y(35), False)
        self._line(g, dwg, X(145), Y(20), X(145), Y(35), False)

        # x=150, 167: S, y=20..40
        self._line(g, dwg, X(150), Y(20), X(150), Y(40), True)
        self._line(g, dwg, X(167), Y(20), X(167), Y(40), True)

        # x=155: S, y=15..20
        self._line(g, dwg, X(155), Y(15), X(155), Y(20), True)

        # ========== ТЕКСТ (OpenSCAD: size=3 → 2.5мм в SVG) ==========
        # Позиции точно из OpenSCAD (x, y) → SVG (X(x), Y(y))

        # ── ТЕКСТ: y = ЦЕНТР ячейки; _text добавляет смещение baseline ──────
        # Заголовок «Изм/Лист/Nдокум/Подп/Дата» — ячейка y=30..35, центр 32.5
        self._text(g, dwg, X(1),    Y(31.5), 'Изм.', 0.9)
        self._text(g, dwg, X(8.5),  Y(31.5), 'Лист', 0.9)
        self._text(g, dwg, X(21),   Y(31.5), 'N докум.', 0.9)
        self._text(g, dwg, X(42),   Y(31.5), 'Подп.', 0.9)
        self._text(g, dwg, X(56),   Y(31.5), 'Дата', 0.9)

        # Роли — ячейки по 5 мм, центры:
        self._text(g, dwg, X(1),  Y(26.5), 'Разраб.', 0.9)    # 25..30
        self._text(g, dwg, X(1),  Y(21.5), 'Пров.', 0.9)      # 20..25
        self._text(g, dwg, X(1),  Y(16.5), 'Т.контр.', 0.9)   # 15..20
        self._text(g, dwg, X(1),  Y( 6.5), 'Н.контр.', 0.9)   #  5..10
        self._text(g, dwg, X(1),  Y( 1.5), 'Утв.', 0.9)       #  0.. 5

        # Правый блок: подписи граф
        self._text(g, dwg, X(139), Y(36.5), 'Лит.', 0.9)      # 35..40
        self._text(g, dwg, X(153), Y(36.5), 'Масса', 0.9)
        self._text(g, dwg, X(167), Y(36.5), 'Масштаб', 0.9)
        self._text(g, dwg, X(139), Y(16.5), 'Лист', 0.9)      # 10..15
        self._text(g, dwg, X(157), Y(16.5), 'Листов', 0.9)

        # Значение масштаба — ячейка y=15..35, центр 25
        if self.scale and self.scale > 0:
            inv = 0.9 / self.scale
            if abs(inv - round(inv)) < 0.01:
                scale_text = f"1:{int(round(inv))}"
            else:
                scale_text = f"1:{inv:.2g}"
            self._text(g, dwg, X(174), Y(25), scale_text, 1.8, 'middle')

        dwg.add(g)

    def add_additional_stamps(self, dwg):
        """Дополнительные графы по ГОСТ 2.104-2006.

        Транслировано из OpenSCAD dop_shtamp + Ramka.

        1. Доп. штамп (dop_shtamp) — по ЛЕВОМУ краю рамки (у корешка),
           вертикальная полоса ~12мм × 145мм, 5 строк.
           OpenSCAD: translate([185-w+17, 0.2, 0]) dop_shtamp()
           → размещается у левого поля (x=20), от низа рамки вверх.

        2. Графа 26 — обозначение документа, повёрнутое на 180°
           (72×14мм, верхний левый угол)

        3. Нижняя строка: «Копировал», «Формат»
        """
        mo = self.MARGIN_OTHER
        ml = self.MARGIN_LEFT
        w, h = self.width, self.height
        S = self.S_FRAME  # линии доп. штампа
        g = dwg.g(id='additional-stamps')

        font_kw = dict(
            font_family='ISOCPEUR, Arial, sans-serif',
            font_style='italic', font_weight='normal', fill='black'
        )

        # ============================================================
        #  dop_shtamp — ЛЕВЫЙ край рамки (у корешка)
        #
        #  OpenSCAD Ramka: translate([185-w+17, 0.2, 0]) dop_shtamp()
        #  В SVG: x = ml (левый край внутренней рамки)
        #         y = от нижнего края рамки вверх на 145мм
        #
        #  OpenSCAD dop_shtamp:
        #    Вертикали: x=-4 и x=1 (= 5мм друг от друга)
        #    Горизонтали: y=0, 25, 59, 84, 112, 150 (ширина 12.5мм)
        #    Надписи повёрнуты на 90°
        # ============================================================

        dg_w = 12.0    # ширина полосы
        dg_h = 150.0   # высота полосы (OpenSCAD: y=0..150)

        # Позиция: вдоль левой рамки, от низа вверх
        frame_left = ml          # x = 20мм (левый край рамки)
        frame_bottom = h - mo    # нижний край рамки

        # Штамп в поле сшивки (левое поле 20мм, ширина штампа 12мм).
        # Правый край штампа совпадает с левым краем внутренней рамки.
        dg_x = frame_left - dg_w   # = ml - 12 = 8мм от левого края листа
        dg_y = frame_bottom - dg_h  # верхний край полосы

        # Контур
        g.add(dwg.rect(insert=(dg_x, dg_y), size=(dg_w, dg_h),
                        fill='white', stroke='black',
                        stroke_width=f'{S}mm'))

        # Горизонтальные разделители
        # OpenSCAD: y=25, 59, 84, 112 (от низа полосы)
        for rb in [25, 59, 84, 112]:
            ry = frame_bottom - rb
            g.add(dwg.line(start=(dg_x, ry), end=(dg_x + dg_w, ry),
                           stroke='black',
                           stroke_width=f'{S/3:.2f}mm'))

        # Вертикальный разделитель (OpenSCAD: x=-4 и x=1 → ~5мм)
        split_x = dg_x + 5.0
        g.add(dwg.line(start=(split_x, dg_y), end=(split_x, frame_bottom),
                       stroke='black', stroke_width=f'{S}mm'))

        # Тексты (повёрнутые на 90° против часовой)
        # OpenSCAD y-позиции (от низа):
        #   y=5 → Инв. № подл.   (строка 0-25)
        #   y=35 → Подп. и дата   (строка 25-59)
        #   y=60 → Взам. инв. №  (строка 59-84)
        #   y=85 → Инв. № дубл.  (строка 84-112)
        #   y=120 → Подп. и дата  (строка 112-145)
        labels_dg = [
            (12.5,  'Инв. № подл.'),
            (42.0,  'Подп. и дата'),
            (71.5,  'Взам. инв. №'),
            (98.0,  'Инв. № дубл.'),
            (131.0, 'Подп. и дата'),
        ]
        for from_bottom, label in labels_dg:
            # ty = вертикальная середина ячейки (текст вращается на -90°,
            #       поэтому "y после поворота" → горизонталь исходного текста)
            ty = frame_bottom - from_bottom
            # tx = центр ПРАВОЙ колонки: dg_x + split(5) + half(3.5)
            tx = dg_x + 8.5
            # При rotate(-90): local_y → SVG_x смещение.
            # Сдвигаем baseline вправо (local_y = +0.5мм) для
            # визуального центрирования 1.4мм шрифта в колонке.
            txt = dwg.text(label, insert=(0, 0.5),
                           font_size='1.4mm', text_anchor='middle',
                           **font_kw)
            txt['transform'] = f'translate({tx},{ty}) rotate(-90)'
            g.add(txt)

        # ============================================================
        #  Графа 26 — обозначение документа (повёрнутое на 180°)
        #  Верхний левый угол внутри рамки, 72 × 14 мм
        #  OpenSCAD Ramka: translate([-w+185+72/2+25, h-17, 0])
        # ============================================================
        g26_w = 72.0
        g26_h = 14.0
        # Графа 26 — в поле сшивки, правый край = левый край внутренней рамки
        g26_x = ml - g26_w   # правый край совпадает с рамкой (только если g26_w < ml)
        # Но ГОСТ допускает только 12мм в поле сшивки.
        # Графа 26 (72мм) шире поля — размещаем по левому краю рамки как «накладку»
        g26_x = ml
        g26_y = mo

        g.add(dwg.rect(insert=(g26_x, g26_y), size=(g26_w, g26_h),
                        fill='white', stroke='black',
                        stroke_width=f'{S}mm'))

        # ============================================================
        #  Нижняя строка: «Копировал» и «Формат»
        #  OpenSCAD Ramka:
        #    translate([60, -4, 0])  text("Копировал")
        #    translate([150, -4, 0]) text("Формат A1")
        # ============================================================
        tb_x0 = w - mo - 185.0
        # y ниже штампа: в поле между рамкой и краем листа
        below_y = frame_bottom + 3.5

        self._text(g, dwg, tb_x0 + 60, below_y, 'Копировал', anchor='middle')

        fmt_text = self.format_name or ''
        self._text(g, dwg, tb_x0 + 150, below_y,
                   f'Формат {fmt_text}', anchor='middle')

        dwg.add(g)

    def add_view_labels(self, dwg, layout):
        """Add view labels per GOST 2.316-68: type B italic, h=5mm, underlined."""
        view_titles = {
            'front': 'Вид спереди',
            'back': 'Вид сзади',
            'top': 'Вид сверху',
            'bottom': 'Вид снизу',
            'right': 'Вид справа',
            'left': 'Вид слева',
        }

        # ГОСТ 2.304-81 тип Б: ISOCPEUR or sans-serif italic, h=5mm
        font_h = 5  # mm
        label_gap = 4  # mm gap between label and view top edge

        for view_name, view_layout in layout.items():
            if view_name not in view_titles:
                continue
            title = view_titles[view_name]

            cx = view_layout['center_x']
            # Label above the view
            ty = view_layout['y'] - label_gap

            # Text element: GOST type B italic
            txt = dwg.text(
                title,
                insert=(cx, ty),
                font_family='ISOCPEUR, Arial, sans-serif',
                font_size=f'{font_h}mm',
                font_style='italic',
                font_weight='normal',
                text_anchor='middle',
                fill='black',
            )
            dwg.add(txt)

            # Underline: horizontal line under the text
            # Estimate text width: ~3mm per character for h=5mm GOST type B
            char_w = font_h * 0.6  # approximate character width ratio for type B
            text_w = len(title) * char_w
            underline_y = ty + 1.5  # slightly below baseline
            dwg.add(dwg.line(
                start=(cx - text_w / 2, underline_y),
                end=(cx + text_w / 2, underline_y),
                stroke='black',
                stroke_width=f'{round(self.S / 2, 2)}mm',
            ))
    
    def generate_drawing(self, filename="unified_eskd_drawing.svg"):
        if not self.views_data:
            logger.error("No view data provided for drawing generation")
            return

        # 0. Определяем минимальный набор видов
        selected_views, exclusion_reasons = self.select_necessary_views()
        # Фильтруем views_data — оставляем только необходимые виды
        self._all_views_data = dict(self.views_data)  # сохраняем полный набор
        self.views_data = {v: self.views_data[v] for v in selected_views
                           if v in self.views_data}

        # 1. Располагаем виды — устанавливает self.scale и self.width/height
        layout = self.arrange_views()

        # 2. Рассчитываем единый S для всего чертежа (ГОСТ 2.303-68)
        #    S зависит от размера главного вида на бумаге
        front_mm = self._front_view_size_mm(self.scale)
        if front_mm >= 80:
            self.S = 0.7
        else:
            self.S = 0.5

        #    Стили линий видов используют тот же S
        self.eskd_styles = calculate_eskd_line_parameters(front_mm, self.S)
        params = self.eskd_styles['_params']
        half_sw = params['S'] / 2
        dash_len = params['dash_length']
        gap_len = params['gap_length']
        logger.info("Line params: S=%.1fmm (front=%.0fmm), dash=%.0fmm, gap=%.1fmm",
                     self.S, front_mm, dash_len, gap_len)

        # 3. Создаём SVG с правильными размерами
        dwg = svgwrite.Drawing(
            filename,
            size=(f"{self.width}mm", f"{self.height}mm"),
            viewBox=f"0 0 {self.width} {self.height}",
            debug=False
        )

        # Проекционные связи убраны — не нужны для 6-видовой компоновки
        # self.add_projection_lines(dwg, layout)

        # Рамка чертежа (ГОСТ 2.104-2006)
        self.add_frame(dwg)

        # Группа для всех видов
        views_group = dwg.g()

        # Отрисовка видов
        for view_name, view_layout in layout.items():
            view_data = self.views_data[view_name]
            view_group = dwg.g()

            translate_x = view_layout['x'] + view_layout['offset_x']
            translate_y = view_layout['y'] + view_layout['offset_y']

            hidden_solid_style = self.eskd_styles['hidden_solid']

            # Рисуем скрытые линии — ручная генерация штрихов (ЕСКД)
            for pA, pB in view_data['hidden']:
                x1 = pA[0] * self.scale + translate_x
                y1 = pA[1] * self.scale + translate_y
                x2 = pB[0] * self.scale + translate_x
                y2 = pB[1] * self.scale + translate_y

                # dash/gap are FIXED in mm — NOT scaled with drawing scale
                segs, is_thin = generate_dashed_segments(
                    (x1, y1), (x2, y2), dash_len, gap_len
                )
                style = hidden_solid_style
                for s, e in segs:
                    line = dwg.line(start=s, end=e, **style)
                    line['style'] = "vector-effect: non-scaling-stroke;"
                    line['data-line-type'] = 'hidden'
                    view_group.add(line)

            # Видимые линии — удлиняем на полтолщины (FIXED mm)
            visible_style = self.eskd_styles['visible']
            for pA, pB in view_data['visible']:
                x1 = pA[0] * self.scale + translate_x
                y1 = pA[1] * self.scale + translate_y
                x2 = pB[0] * self.scale + translate_x
                y2 = pB[1] * self.scale + translate_y

                pA_ext, pB_ext = extend_line((x1, y1), (x2, y2), half_sw)
                line = dwg.line(start=pA_ext, end=pB_ext, **visible_style)
                line['style'] = "vector-effect: non-scaling-stroke;"
                view_group.add(line)

            views_group.add(view_group)

        dwg.add(views_group)

        # Добавляем основную надпись (штамп)
        self.add_title_block(dwg)

        # Добавляем дополнительные графы (Гр.19, 21, 22, 26)
        self.add_additional_stamps(dwg)

        # ГОСТ 2.305-2008: подписи видов НЕ нужны, если виды расположены
        # в стандартной проекционной связи (метод первого угла).
        # Подписи с буквой и стрелкой нужны только для вынесенных видов.
        # self.add_view_labels(dwg, layout)

        # Сохраняем чертеж
        dwg.save()
        logger.info(f"Unified ESKD drawing saved to: {filename}")
        return filename


def generate_unified_drawing(view_processor, output_filename="unified_eskd_drawing.svg",
                              format_name=None):
    """Generate unified ESKD drawing.
    
    format_name: 'A4'...'A0' for explicit format, or None for auto-select.
    """
    views = ["front", "back", "top", "bottom", "left", "right"]

    # Create sheet (format_name=None triggers auto-select)
    sheet = ESKDDrawingSheet(format_name=format_name)

    # Load view data
    for view in views:
        transformed, visible, hidden = view_processor.process_view(view, "hidden_line_dashed")
        sheet.add_view_data(view, transformed, visible, hidden)

    # Generate drawing (arrange_views will call calculate_optimal_scale -> auto_select)
    filename = sheet.generate_drawing(output_filename)

    return filename


class ESKDLineProcessor:
    def __init__(self):
        self.LINE_WIDTHS = {
            'thick': 1.0,
            'thin': 0.3,
            'dash': 0.3,
            'dashdot': 0.3,
            'contour': 1.0,
            'dimension': 0.3,
            'section': 0.5,
            'axis': 0.3,
        }
        self.DASH_PATTERNS = {
            'dash': '8,3',
            'dashdot': '8,2,1,2',
            'axis': '15,2,2,2',
        }

    def _calculate_scale_factor(self, svg_root: ET.Element) -> float:
        viewbox = svg_root.get('viewBox')
        if viewbox:
            _, _, width, height = map(float, viewbox.split())
            max_size = max(width, height)
            return max_size / 1000.0
        return 1.0

    def _adjust_line_width(self, line_type: str, scale_factor: float) -> float:
        base_width = self.LINE_WIDTHS.get(line_type, 0.3)
        return base_width * scale_factor

    @staticmethod
    def _find_parent(root: ET.Element, child: ET.Element) -> Optional[ET.Element]:
        for elem in list(root):
            if elem is child:
                return root
            parent = ESKDLineProcessor._find_parent(elem, child)
            if parent is not None:
                return parent
        return None

    def _merge_collinear_lines(self, lines: List[Dict]) -> List[Dict]:
        def are_collinear(line1: Dict, line2: Dict, tol: float = 1e-6) -> bool:
            x1, y1 = map(float, line1['start'])
            x2, y2 = map(float, line1['end'])
            x3, y3 = map(float, line2['start'])
            x4, y4 = map(float, line2['end'])

            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x4 - x3, y4 - y3])
            norm1 = norm(v1)
            norm2 = norm(v2)
            if norm1 < tol or norm2 < tol:
                return False

            v1n = v1 / norm1
            v2n = v2 / norm2

            if abs(abs(np.dot(v1n, v2n)) - 1) > tol:
                return False

            v3 = np.array([x3 - x1, y3 - y1])
            cross_val = v1n[0]*v3[1] - v1n[1]*v3[0]
            if abs(cross_val) > tol:
                return False

            return True

        def merge_two_lines(line1: Dict, line2: Dict) -> Dict:
            all_points = [
                line1['start'],
                line1['end'],
                line2['start'],
                line2['end']
            ]
            max_dist = 0
            best_pair = (line1['start'], line1['end'])
            for i in range(len(all_points)):
                for j in range(i+1, len(all_points)):
                    ptA = np.array(all_points[i], dtype=float)
                    ptB = np.array(all_points[j], dtype=float)
                    dist_sq = np.sum((ptB - ptA) ** 2)
                    if dist_sq > max_dist:
                        max_dist = dist_sq
                        best_pair = (all_points[i], all_points[j])
            return {
                'start': best_pair[0],
                'end':   best_pair[1],
                'style': line1['style'],
                'stroke_width': line1.get('stroke_width', '0.3mm')
            }

        result = []
        used = [False] * len(lines)

        for i in range(len(lines)):
            if used[i]:
                continue
            current = lines[i]
            used[i] = True
            merged_something = True
            while merged_something:
                merged_something = False
                for j in range(i+1, len(lines)):
                    if not used[j]:
                        if lines[j]['style'] == current['style']:
                            if are_collinear(current, lines[j]):
                                current = merge_two_lines(current, lines[j])
                                used[j] = True
                                merged_something = True
            result.append(current)

        return result

    @staticmethod
    def _classify_line(elem: ET.Element) -> Tuple[str, str]:
        """Classify line type and extract stroke-width. Returns (type, stroke_width)."""
        # First check data-line-type attribute (set by manual dash generation)
        data_type = elem.get('data-line-type', '')
        if data_type in ('hidden', 'axis'):
            sw = elem.get('stroke-width', '')
            style = elem.get('style', '')
            if not sw and 'stroke-width' in style:
                parts = style.split('stroke-width:')
                if len(parts) > 1:
                    sw = parts[1].split(';')[0].strip()
            if not sw:
                sw = '0.3mm'
            return data_type, sw

        da = elem.get('stroke-dasharray', '')
        sw = elem.get('stroke-width', '')
        style = elem.get('style', '')

        # Check stroke color — red lines are axes
        stroke_color = elem.get('stroke', '')
        if not stroke_color and 'stroke:' in style:
            parts = style.split('stroke:')
            if len(parts) > 1:
                stroke_color = parts[1].split(';')[0].strip()

        # Check stroke-dasharray inside style attribute
        if not da and 'stroke-dasharray' in style:
            parts = style.split('stroke-dasharray:')
            if len(parts) > 1:
                da = parts[1].split(';')[0].strip()

        # Extract stroke-width from style if not in attributes
        if not sw and 'stroke-width' in style:
            parts = style.split('stroke-width:')
            if len(parts) > 1:
                sw = parts[1].split(';')[0].strip()

        if not sw:
            sw = '0.3mm'

        # Red stroke = axis line
        if stroke_color.lower() in ('red', '#ff0000', '#f00'):
            return 'axis', sw

        line_type = 'visible'
        if da:
            comma_count = da.count(',')
            if comma_count >= 3:
                line_type = 'axis'
            elif comma_count >= 1:
                line_type = 'hidden'
        return line_type, sw

    def process_svg(self, svg_path: str):
        tree = ET.parse(svg_path)
        root = tree.getroot()

        namespace = '{http://www.w3.org/2000/svg}'
        visible_lines = []
        preserved_elems = []  # Lines to keep as-is (projection, hidden dashes, axis)
        all_line_elems = root.findall('.//' + namespace + 'line')

        for elem in all_line_elems:
            # Detect projection lines by stroke color (#000 vs black)
            stroke = elem.get('stroke', '')
            style_attr = elem.get('style', '')
            is_projection = (stroke == '#000' or
                             'stroke:#000;' in style_attr or
                             'stroke:#000000;' in style_attr or
                             elem.get('data-line-type') == 'projection')
            if is_projection:
                preserved_elems.append(elem)
                continue

            # Hidden dashes and axis lines are already correct — preserve them
            data_type = elem.get('data-line-type', '')
            if data_type in ('hidden', 'axis'):
                preserved_elems.append(elem)
                continue

            # Red axis lines (from stroke color) — preserve
            stroke_color = stroke.lower()
            if not stroke_color and 'stroke:' in style_attr:
                parts = style_attr.split('stroke:')
                if len(parts) > 1:
                    stroke_color = parts[1].split(';')[0].strip().lower()
            if stroke_color in ('red', '#ff0000', '#f00'):
                preserved_elems.append(elem)
                continue

            line_type, sw = self._classify_line(elem)
            line_data = {
                'start': (elem.get('x1'), elem.get('y1')),
                'end':   (elem.get('x2'), elem.get('y2')),
                'style': line_type,
                'stroke_width': sw,
                'element': elem
            }
            visible_lines.append(line_data)

        # Remove only visible geometry lines (merge them)
        for line_obj in visible_lines:
            elem = line_obj['element']
            parent = self._find_parent(root, elem)
            if parent is not None:
                parent.remove(elem)

        merged_lines = self._merge_collinear_lines(visible_lines)

        for mline in merged_lines:
            new_line = ET.SubElement(root, namespace + 'line')
            new_line.set('x1', str(mline['start'][0]))
            new_line.set('y1', str(mline['start'][1]))
            new_line.set('x2', str(mline['end'][0]))
            new_line.set('y2', str(mline['end'][1]))

            sw = mline.get('stroke_width', '0.3mm')
            new_line.set(
                'style',
                f"stroke:black;stroke-width:{sw};"
                "vector-effect:non-scaling-stroke;"
            )

        out_eskd = svg_path.replace('.svg', '_eskd.svg')
        tree.write(out_eskd, encoding='utf-8', xml_declaration=True)
        logger.info(f"ESKD-processed SVG saved to: {out_eskd}")


def extend_line(pA, pB, extension):
    """Extend line segment by `extension` at both ends along its direction."""
    dx = float(pB[0] - pA[0])
    dy = float(pB[1] - pA[1])
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-12:
        return pA, pB
    ux, uy = dx / length, dy / length
    new_a = (float(pA[0]) - ux * extension, float(pA[1]) - uy * extension)
    new_b = (float(pB[0]) + ux * extension, float(pB[1]) + uy * extension)
    return new_a, new_b


def generate_dashed_segments(pA, pB, dash_len, gap_len):
    """Generate ESKD-compliant dash segments for a line.

    Ensures first and last segments are full dashes.
    If line is too short for 2 dashes + 1 gap, returns the whole line
    with is_thin_fallback=True (draw as thin solid instead).

    Returns: (segments, is_thin_fallback)
      segments: list of ((x1,y1), (x2,y2)) tuples
      is_thin_fallback: bool — if True, draw as thin solid line
    """
    dx = float(pB[0] - pA[0])
    dy = float(pB[1] - pA[1])
    length = math.sqrt(dx * dx + dy * dy)

    min_length = 2 * dash_len + gap_len
    if length < min_length:
        return [((float(pA[0]), float(pA[1])), (float(pB[0]), float(pB[1])))], True

    # n dashes + (n-1) gaps = length → n = (length + gap) / (dash + gap)
    n = max(2, round((length + gap_len) / (dash_len + gap_len)))
    actual_dash = (length - (n - 1) * gap_len) / n
    if actual_dash < dash_len * 0.4:
        n -= 1
        if n < 2:
            return [((float(pA[0]), float(pA[1])), (float(pB[0]), float(pB[1])))], True
        actual_dash = (length - (n - 1) * gap_len) / n
    actual_gap = gap_len if n > 1 else 0

    ux, uy = dx / length, dy / length
    segments = []
    pos = 0.0
    ax, ay = float(pA[0]), float(pA[1])
    for i in range(n):
        start_pos = pos
        end_pos = pos + actual_dash
        s = (ax + ux * start_pos, ay + uy * start_pos)
        e = (ax + ux * end_pos, ay + uy * end_pos)
        segments.append((s, e))
        pos = end_pos + actual_gap

    return segments, False


def generate_svg_for_view(view_processor: ViewProcessor,
                          view: str,
                          outfile: str,
                          display_mode: str):
    transformed_with_z, visible_lines, hidden_lines = view_processor.process_view(
        view, display_mode=display_mode
    )

    min_x = float(np.min(transformed_with_z[:, 0]))
    max_x = float(np.max(transformed_with_z[:, 0]))
    min_y = float(np.min(transformed_with_z[:, 1]))
    max_y = float(np.max(transformed_with_z[:, 1]))

    model_size = max(max_x - min_x, max_y - min_y)
    margin = model_size * 0.05
    width = max_x - min_x + 2 * margin
    height = max_y - min_y + 2 * margin

    eskd_styles = calculate_eskd_line_parameters(model_size)  # model_size ~ view size in model units
    params = eskd_styles['_params']
    half_sw = params['S'] / 2
    dash_len = params['dash_length']
    gap_len = params['gap_length']

    dwg = svgwrite.Drawing(
        outfile,
        viewBox=f"{min_x - margin} {min_y - margin} {width} {height}",
        size=('297mm', '210mm'),
        debug=False
    )
    g = dwg.g()

    if display_mode == "wireframe":
        style = eskd_styles['visible']
        for pA, pB in visible_lines + hidden_lines:
            pA_ext, pB_ext = extend_line(pA, pB, half_sw)
            line = dwg.line(start=pA_ext, end=pB_ext, **style)
            line['style'] = "vector-effect: non-scaling-stroke;"
            g.add(line)

    elif display_mode == "hidden_line_removal":
        style = eskd_styles['visible']
        for pA, pB in visible_lines:
            pA_ext, pB_ext = extend_line(pA, pB, half_sw)
            line = dwg.line(start=pA_ext, end=pB_ext, **style)
            line['style'] = "vector-effect: non-scaling-stroke;"
            g.add(line)

    else:
        hidden_solid_style = eskd_styles['hidden_solid']
        visible_style = eskd_styles['visible']

        # Hidden lines: manual dash generation (ESKD: start/end with full dash)
        for pA, pB in hidden_lines:
            segs, is_thin = generate_dashed_segments(pA, pB, dash_len, gap_len)
            if is_thin:
                style = hidden_solid_style
                for s, e in segs:
                    line = dwg.line(start=s, end=e, **style)
                    line['style'] = "vector-effect: non-scaling-stroke;"
                    line['data-line-type'] = 'hidden'
                    g.add(line)
            else:
                style = hidden_solid_style
                for s, e in segs:
                    line = dwg.line(start=s, end=e, **style)
                    line['style'] = "vector-effect: non-scaling-stroke;"
                    line['data-line-type'] = 'hidden'
                    g.add(line)

        # Visible lines: extend by half stroke-width
        for pA, pB in visible_lines:
            pA_ext, pB_ext = extend_line(pA, pB, half_sw)
            line = dwg.line(start=pA_ext, end=pB_ext, **visible_style)
            line['style'] = "vector-effect: non-scaling-stroke;"
            g.add(line)

    dwg.add(g)
    dwg.save()


def process_all_views(view_proc: ViewProcessor, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    views = ["front", "back", "top", "bottom", "right", "left", "isometric"]

    for view in views:
        outfile = os.path.join(output_dir, f"{view}_view_hidden_line_dashed.svg")
        generate_svg_for_view(view_proc, view, outfile, "hidden_line_dashed")
        logger.info(f"Processed {view} view in hidden_line_dashed mode")


def main():
    input_stl = "Shveller_16P.stl"

    logger.info("Loading %s ...", input_stl)
    stl_mesh = mesh.Mesh.from_file(input_stl)

    vertices = []
    faces = []
    v_map = {}
    for tri in stl_mesh.vectors:
        f_idx = []
        for v in tri:
            vt = tuple(map(lambda x: round(float(x), 6), v))
            if vt not in v_map:
                v_map[vt] = len(vertices)
                vertices.append(np.array(vt, dtype=float))
            f_idx.append(v_map[vt])
        faces.append(tuple(f_idx))

    logger.info("Loaded %d unique vertices, %d faces.", len(vertices), len(faces))

    # ===== Step 1: PCA orientation with snap to 90 deg =====
    logger.info("=" * 60)
    logger.info("Step 1: Face-normal-weighted PCA orientation")
    logger.info("=" * 60)
    raw_verts = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)
    oriented_verts, pca_R = orient_model_by_normals(raw_verts, faces_arr)

    # Recompute face normals for oriented geometry
    v01 = oriented_verts[faces_arr[:, 1]] - oriented_verts[faces_arr[:, 0]]
    v02 = oriented_verts[faces_arr[:, 2]] - oriented_verts[faces_arr[:, 0]]
    raw_normals = np.cross(v01, v02).astype(np.float64)
    nrm_len = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    nrm_len[nrm_len < 1e-12] = 1.0
    pca_normals = (raw_normals / nrm_len).astype(np.float32)

    # ===== Step 2: Select best front view and reorient =====
    logger.info("=" * 60)
    logger.info("Step 2: Best front view selection")
    logger.info("=" * 60)
    final_verts, final_normals = select_best_front_and_reorient(
        oriented_verts, faces_arr, pca_normals
    )

    # ===== Step 3: Rebuild topology on the oriented vertices =====
    logger.info("=" * 60)
    logger.info("Step 3: Topology processing")
    logger.info("=" * 60)
    final_verts_list = [final_verts[i] for i in range(len(final_verts))]
    topo = TopologyProcessor(final_verts_list, faces)
    valid_edges = topo.process()

    view_proc = ViewProcessor(
        topo.vertices, valid_edges, topo.faces,
        edge_faces=topo.edge_faces,
        face_normals=topo.face_normals,
        smooth_edges=topo.smooth_edges
    )

    # ===== Step 4: Generate unified ESKD drawing =====
    logger.info("=" * 60)
    logger.info("Step 4: Generate ESKD drawing")
    logger.info("=" * 60)
    # format_name=None -> auto-select format and scale per GOST
    generate_unified_drawing(view_proc, "unified_eskd_drawing.svg", format_name=None)

    logger.info("Done.")


if __name__ == "__main__":
    main()