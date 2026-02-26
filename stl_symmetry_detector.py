"""
STL Symmetry & Rotation Axis Detector  (v4.1 — topology-based cylinders)
=========================================================================
v4.1: Топологическая детекция цилиндров.
  - KDTree дедупликация вершин -> корректная смежность граней
  - Двугранные углы -> маркировка кривых граней
  - Region growing по ВСЕМ рёбрам между кривыми гранями
  - PCA нормалей -> ось цилиндра, фит окружности МНК

Зависимости: numpy, scipy, numpy-stl
    pip install numpy-stl scipy
"""

import argparse, sys, os, time
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.linalg import eigh
from stl import mesh
import scipy
from collections import defaultdict

_SCIPY_HAS_WORKERS = tuple(int(x) for x in scipy.__version__.split(".")[:2]) >= (1, 6)
_KDT_KW = {"workers": -1} if _SCIPY_HAS_WORKERS else {}
MAX_PTS = 5000

# ── Загрузка ──────────────────────────────────────────────────────────────

def load_vertices(path):
    m = mesh.Mesh.from_file(path)
    v = m.vectors.reshape(-1, 3)
    return np.unique(np.round(v, 5), axis=0).astype(np.float64)

def load_mesh_data(path):
    m = mesh.Mesh.from_file(path)
    V = m.vectors.astype(np.float64)
    e1, e2 = V[:,1]-V[:,0], V[:,2]-V[:,0]
    cr = np.cross(e1, e2)
    nm = np.linalg.norm(cr, axis=1, keepdims=True)
    nm = np.where(nm < 1e-12, 1.0, nm)
    return {"vectors": V, "normals": cr/nm,
            "centroids": V.mean(axis=1), "areas": np.linalg.norm(cr, axis=1)*0.5}

def center(pts): return pts - pts.mean(axis=0)

def maybe_subsample(pts, mx=MAX_PTS):
    return pts if len(pts) <= mx else pts[np.random.choice(len(pts), mx, False)]

def pca_axes(pts):
    _, vecs = eigh(np.cov(pts.T))
    return vecs.T[::-1]

def sphere_candidates(n_az=18, n_pol=9):
    dirs = []
    for i in range(n_pol+1):
        th = np.pi*i/(2*n_pol)
        for j in range(n_az):
            ph = 2*np.pi*j/n_az
            dirs.append([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    return np.unique(np.round(dirs,6), axis=0)

# ── Топологическая детекция цилиндров ─────────────────────────────────────

def _build_adjacency(vectors):
    """Смежность граней через KDTree-дедупликацию вершин."""
    n_f = len(vectors)
    all_v = vectors.reshape(-1, 3)
    # Медианное ребро
    el = []
    for fi in range(n_f):
        tri = vectors[fi]
        for k in range(3):
            el.append(np.linalg.norm(tri[(k+1)%3] - tri[k]))
    med_e = np.median(el)
    tol = med_e * 0.01

    tree = KDTree(all_v)
    vid = np.full(len(all_v), -1, dtype=int)
    nxt = 0
    for i in range(len(all_v)):
        if vid[i] >= 0: continue
        for j in tree.query_ball_point(all_v[i], tol):
            if vid[j] < 0: vid[j] = nxt
        nxt += 1

    e2f = defaultdict(set)
    for fi in range(n_f):
        vi = [vid[fi*3+k] for k in range(3)]
        for k in range(3):
            ek = (min(vi[k], vi[(k+1)%3]), max(vi[k], vi[(k+1)%3]))
            e2f[ek].add(fi)

    adj = defaultdict(set)
    for faces in e2f.values():
        fl = list(faces)
        for a in fl:
            for b in fl:
                if a != b: adj[a].add(b)
    return adj, med_e


def _mark_curved_faces(adj, normals, n_f, min_deg=3.0, max_deg=60.0):
    """Грань 'кривая' если хотя бы 1 сосед с двугранным углом в [min, max]."""
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


def _grow_regions(adj, curved, n_f, min_sz=6):
    """BFS по ВСЕМ рёбрам между кривыми гранями."""
    vis = np.zeros(n_f, dtype=bool)
    regions = []
    for s in range(n_f):
        if vis[s] or not curved[s]:
            vis[s] = True; continue
        reg = set(); q = [s]; vis[s] = True
        while q:
            fi = q.pop()
            reg.add(fi)
            for fj in adj.get(fi, []):
                if not vis[fj] and curved[fj]:
                    vis[fj] = True; q.append(fj)
        if len(reg) >= min_sz:
            regions.append(sorted(reg))
    return regions


def _segment_by_sharp_edges(adj, normals, n_f, sharp_angle_deg=15.0):
    """
    Сегментация меша по острым рёбрам.
    BFS через рёбра с малым двугранным углом (< sharp_angle) → патчи.
    Острые рёбра (стык плоскостей, перегибы) разделяют меш на логические части.
    """
    sharp_rad = np.radians(sharp_angle_deg)
    vis = np.zeros(n_f, dtype=bool)
    segments = []
    for s in range(n_f):
        if vis[s]: continue
        seg = set(); q = [s]; vis[s] = True
        while q:
            fi = q.pop()
            seg.add(fi)
            for fj in adj.get(fi, []):
                if vis[fj]: continue
                dot = np.clip(np.dot(normals[fi], normals[fj]), -1, 1)
                if np.arccos(dot) < sharp_rad:
                    vis[fj] = True; q.append(fj)
        segments.append(sorted(seg))
    return segments


def _segment_is_curved(face_ids, normals, min_spread=0.05):
    """Проверяет, является ли сегмент криволинейным (не плоским)."""
    n = normals[np.array(face_ids)]
    if len(n) < 3: return False
    spread = np.std(n, axis=0).mean()
    return spread > min_spread


def _fit_circle_lsq(pts):
    n = len(pts)
    if n < 3: return None
    x, y = pts[:,0], pts[:,1]
    A = np.column_stack([2*x, 2*y, np.ones(n)])
    b = x**2 + y**2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = res[0], res[1]
    r2 = res[2] + cx**2 + cy**2
    return (cx, cy, np.sqrt(r2)) if r2 > 0 else None


def _angular_coverage(angles):
    if len(angles) < 2: return 0.0
    sa = np.sort(angles % (2*np.pi))
    gaps = np.diff(sa)
    gaps = np.append(gaps, 2*np.pi - sa[-1] + sa[0])
    return 2*np.pi - gaps.max()


def _fit_cylinder(face_ids, normals, centroids, areas,
                   min_fq=0.75, min_arc=90.0, min_ld=0.08, verbose=False):
    idx = np.array(face_ids)
    n, c = normals[idx], centroids[idx]
    if len(idx) < 4: return None

    # PCA нормалей -> ось
    cov = np.cov(n.T)
    eigv, eigvec = eigh(cov)
    axis = eigvec[:,0].copy()
    axis /= np.linalg.norm(axis)
    if axis[2] < 0 or (axis[2]==0 and axis[1]<0): axis = -axis

    # Нормали ⊥ оси?
    dots = np.abs(n @ axis)
    if (dots < 0.35).mean() < 0.60:
        if verbose: print(f"      [{len(idx)} гр.] перп.доля мала")
        return None

    # Базис
    u = np.array([1.,0,0])
    if abs(np.dot(u, axis)) > 0.9: u = np.array([0.,1,0])
    u -= np.dot(u, axis)*axis; u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    along = c @ axis
    proj = c - np.outer(along, axis)
    p2d = np.column_stack([proj @ u, proj @ v])

    circ = _fit_circle_lsq(p2d)
    if circ is None: return None
    cx, cy, r = circ
    if r < 0.05: return None

    # Инлайеры
    d = np.abs(np.hypot(p2d[:,0]-cx, p2d[:,1]-cy) - r)
    inl = d < 0.20*r
    if inl.sum() < 4:
        if verbose: print(f"      [{len(idx)} гр.] инлайеров={inl.sum()}")
        return None

    ref = _fit_circle_lsq(p2d[inl])
    if ref is None: return None
    cx, cy, r = ref
    if r < 0.05: return None

    fd = np.abs(np.hypot(p2d[inl,0]-cx, p2d[inl,1]-cy) - r)
    fq = 1.0 - (fd.mean()/r) if r > 1e-9 else 0
    if fq < min_fq:
        if verbose: print(f"      [{len(idx)} гр.] fq={fq:.3f}")
        return None

    # Покрытие
    rad = p2d[inl] - [cx, cy]
    rn = np.linalg.norm(rad, axis=1, keepdims=True)
    rn = np.where(rn<1e-12, 1, rn)
    ru = rad/rn
    ang = np.arctan2(ru[:,1], ru[:,0])
    acov = _angular_coverage(ang)
    if acov < np.radians(min_arc):
        if verbose: print(f"      [{len(idx)} гр.] дуга={np.degrees(acov):.0f}°")
        return None

    # Тип
    n2d = np.column_stack([n[inl]@u, n[inl]@v])
    nn = np.linalg.norm(n2d, axis=1, keepdims=True)
    nn = np.where(nn<1e-12, 1, nn)
    n2d /= nn
    md = (n2d * ru).sum(axis=1).mean()
    ctype = "hole" if md < -0.25 else ("boss" if md > 0.25 else "cylinder")

    ax_c = along[inl]
    L = ax_c.max() - ax_c.min()
    D = 2*r
    if L/D < min_ld:
        if verbose: print(f"      [{len(idx)} гр.] L/D={L/D:.3f}")
        return None

    return {"axis": axis, "center": cx*u + cy*v + ax_c.mean()*axis,
            "radius": float(r), "length": float(L),
            "n_faces": int(inl.sum()), "fit_quality": float(fq),
            "type": ctype, "angular_coverage_deg": float(np.degrees(acov))}


def _merge_cylinders(cyls, verbose=False):
    """
    Дедупликация цилиндров.
    Мёрж только если:
      - радиусы похожи (±30%)
      - оси сонаправлены (< 25°)
      - ПОПЕРЕЧНОЕ расстояние (⊥ оси) < 0.5 * max_radius
      - ПРОДОЛЬНОЕ расстояние (вдоль оси) < (L1+L2)/2 (перекрытие)
    """
    if len(cyls) <= 1: return cyls
    merged, used = [], [False]*len(cyls)
    for i, c1 in enumerate(cyls):
        if used[i]: continue
        grp = [c1]
        for j in range(i+1, len(cyls)):
            if used[j]: continue
            c2 = cyls[j]
            # Не мержить разные типы (hole + boss = разные поверхности!)
            if c1["type"] != c2["type"]: continue
            rm = max(c1["radius"], c2["radius"])
            # Радиусы похожи?
            if abs(c1["radius"]-c2["radius"])/rm > 0.30: continue
            # Оси сонаправлены?
            adot = abs(np.dot(c1["axis"], c2["axis"]))
            if adot < np.cos(np.radians(25)): continue
            # Разложение вектора центров на продольную и поперечную
            avg_axis = (c1["axis"] + c2["axis"] * np.sign(np.dot(c1["axis"], c2["axis"])))
            avg_axis /= np.linalg.norm(avg_axis) + 1e-12
            dc = c2["center"] - c1["center"]
            along_dist = abs(np.dot(dc, avg_axis))
            cross_dist = np.sqrt(max(0, np.dot(dc,dc) - along_dist**2))
            # Поперёк: центры совпадают
            if cross_dist > 0.5 * rm: continue
            # Вдоль: перекрытие или близость
            max_along = (c1["length"] + c2["length"]) / 2 + 0.5 * rm
            if along_dist > max_along:
                if verbose:
                    print(f"    🔧 НЕ мёрж #{i}+#{j}: along={along_dist:.1f} > {max_along:.1f}")
                continue
            if verbose:
                print(f"    🔧 мёрж #{i}+#{j}: cross={cross_dist:.2f}, along={along_dist:.1f}")
            grp.append(c2); used[j] = True
        used[i] = True
        best = max(grp, key=lambda c: c["fit_quality"]*c["n_faces"])
        best["n_faces"] = sum(c["n_faces"] for c in grp)
        merged.append(best)
    return merged


class CylinderDetector:
    def __init__(self, verbose=False, min_dih=1.0, max_dih=60.0,
                 min_reg=6, min_fq=0.75, min_arc=45.0, min_ld=0.08,
                 sharp_angle=15.0):
        self.verbose = verbose
        self.min_dih, self.max_dih = min_dih, max_dih
        self.min_reg = min_reg
        self.min_fq, self.min_arc, self.min_ld = min_fq, min_arc, min_ld
        self.sharp_angle = sharp_angle

    def detect(self, md, offset):
        N = md["normals"]; C = md["centroids"] - offset
        A = md["areas"]; V = md["vectors"]
        nf = len(N)
        print(f"    🔧 Детекция цилиндров: {nf} граней")

        print(f"    🔧 Построение смежности...", end=" ", flush=True)
        adj, me = _build_adjacency(V)
        ne = sum(len(v) for v in adj.values())//2
        print(f"рёбер={ne}, med_edge={me:.2f}")

        # Гистограмма двугранных углов (verbose)
        if self.verbose:
            all_angles = []
            seen = set()
            for fi in range(nf):
                for fj in adj.get(fi, []):
                    if (fi,fj) not in seen:
                        seen.add((fi,fj)); seen.add((fj,fi))
                        dot = np.clip(np.dot(N[fi], N[fj]), -1, 1)
                        all_angles.append(np.degrees(np.arccos(dot)))
            aa = np.array(all_angles)
            print(f"    🔧 Углы: min={aa.min():.1f}°, max={aa.max():.1f}°, "
                  f"median={np.median(aa):.1f}°")
            for lo, hi in [(0,1),(1,3),(3,10),(10,30),(30,60),(60,90),(90,120),(120,180)]:
                cnt = ((aa >= lo) & (aa < hi)).sum()
                if cnt > 0:
                    print(f"        [{lo:3d}°-{hi:3d}°): {cnt}")

        # ═══ Фаза 1: Сегментация по острым рёбрам ═══
        segments = _segment_by_sharp_edges(adj, N, nf, self.sharp_angle)
        curved_segs = [(seg, len(seg)) for seg in segments
                       if len(seg) >= self.min_reg and _segment_is_curved(seg, N)]
        print(f"    🔧 Сегментов (sharp>{self.sharp_angle}°): {len(segments)}, "
              f"кривых (≥{self.min_reg} гр.): {len(curved_segs)} "
              f"({[s for _,s in curved_segs]})")

        # ═══ Фаза 2: Фит цилиндров из кривых сегментов ═══
        cyls = []
        for si, (seg, sz) in enumerate(curved_segs):
            cyl = _fit_cylinder(seg, N, C, A,
                                self.min_fq, self.min_arc, self.min_ld,
                                self.verbose)
            if cyl:
                cyls.append(cyl)
                if self.verbose:
                    print(f"      сег[{si}]({sz}): R={cyl['radius']:.2f} "
                          f"L={cyl['length']:.2f} {cyl['type']}")
            elif self.verbose:
                print(f"      сег[{si}]({sz}): не цилиндр")

        # ═══ Фаза 3 (fallback): Старый метод для пропущенных ═══
        # Если фаза 1 нашла мало, попробуем классический подход
        if len(cyls) < 2:
            curved = _mark_curved_faces(adj, N, nf, self.min_dih, self.max_dih)
            regions = _grow_regions(adj, curved, nf, self.min_reg)
            if self.verbose:
                print(f"    🔧 Fallback: кривых граней={curved.sum()}, "
                      f"регионов={len(regions)} ({[len(r) for r in regions]})")
            for ri, reg in enumerate(regions):
                cyl = _fit_cylinder(reg, N, C, A,
                                    self.min_fq, self.min_arc, self.min_ld,
                                    self.verbose)
                if cyl:
                    cyls.append(cyl)

        b4 = len(cyls)
        cyls = _merge_cylinders(cyls, self.verbose)
        if b4 > len(cyls): print(f"    🔧 Мёрж: {b4} -> {len(cyls)}")
        print(f"    🔧 Обнаружено цилиндров: {len(cyls)}")
        return cyls


# ── Симметрии ─────────────────────────────────────────────────────────────

def batch_symmetry_scores(pts, axes, tree, tol):
    Na, Np = len(axes), len(pts)
    proj = pts @ axes.T
    refl = pts[:,None,:] - 2.0*proj[:,:,None]*axes[None,:,:]
    refl = refl.transpose(1,0,2).reshape(Na*Np, 3)
    d, _ = tree.query(refl, **_KDT_KW)
    return (d.reshape(Na,Np) < tol).mean(axis=1)

def _rot_matrices(axes, angle):
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axes[:,0], axes[:,1], axes[:,2]
    oc = 1-c
    R = np.empty((len(axes),3,3))
    R[:,0,0]=c+x*x*oc;  R[:,0,1]=x*y*oc-z*s; R[:,0,2]=x*z*oc+y*s
    R[:,1,0]=y*x*oc+z*s; R[:,1,1]=c+y*y*oc;  R[:,1,2]=y*z*oc-x*s
    R[:,2,0]=z*x*oc-y*s; R[:,2,1]=z*y*oc+x*s; R[:,2,2]=c+z*z*oc
    return R

def batch_rotation_scores(pts, axes, n, tree, tol):
    Na, Np = len(axes), len(pts)
    R = _rot_matrices(axes, 2*np.pi/n)
    rot = np.einsum('aij,pj->api', R, pts).reshape(Na*Np, 3)
    d, _ = tree.query(rot, **_KDT_KW)
    return (d.reshape(Na,Np) < tol).mean(axis=1)

def _dedup(items, ang=10.0):
    if not items: return []
    items = sorted(items, key=lambda x: -x[0])
    kept = []
    for sc, ax in items:
        if not any(abs(np.dot(ax, ka)) > np.cos(np.radians(ang)) for _, ka in kept):
            kept.append((sc, ax))
    return kept


# ── Главный детектор ──────────────────────────────────────────────────────

class SymmetryDetector:
    def __init__(self, tol_factor=0.01, score_threshold=0.85,
                 n_azimuth=24, n_polar=12, max_cn=8,
                 max_pts=MAX_PTS, detect_cylinders=True, verbose=False,
                 min_dih=1.0, max_dih=60.0, min_reg=6, min_arc=45.0,
                 sharp_angle=15.0):
        self.tf = tol_factor; self.st = score_threshold
        self.naz = n_azimuth; self.npol = n_polar
        self.mcn = max_cn; self.mpts = max_pts
        self.dc = detect_cylinders
        self.cd = CylinderDetector(verbose=verbose,
                                    min_dih=min_dih, max_dih=max_dih,
                                    min_reg=min_reg, min_arc=min_arc,
                                    sharp_angle=sharp_angle)

    def detect(self, path):
        print(f"\n📂  Загрузка: {path}")
        pf = center(load_vertices(path))
        print(f"    Вершин (уникальных): {len(pf):,}")
        bd = np.linalg.norm(pf.max(0)-pf.min(0))
        tol = self.tf * bd
        print(f"    Диагональ bbox: {bd:.4f}  |  допуск: {tol:.4f}")

        pts = maybe_subsample(pf, self.mpts)
        if len(pts) < len(pf):
            print(f"    Прореживание: {len(pf)} -> {len(pts)}")
        tree = KDTree(pts)

        cyls, ca = [], np.empty((0,3))
        if self.dc:
            md = load_mesh_data(path)
            rc = md["vectors"].reshape(-1,3).mean(axis=0)
            cyls = self.cd.detect(md, rc)
            if cyls:
                ca = np.array([c["axis"] for c in cyls])
                ca /= np.linalg.norm(ca, axis=1, keepdims=True)

        pca = pca_axes(pts)
        grid = sphere_candidates(self.naz, self.npol)
        cands = np.vstack([pca, grid])
        if len(ca): cands = np.vstack([cands, ca]); print(f"    Осей цил.: {len(ca)}")
        cands = np.unique(np.round(cands,4), axis=0)
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
        print(f"    Кандидатов осей: {len(cands)}")

        ns = self.mcn
        print(f"    [1/{ns}] Зеркальные...", end=" ", flush=True)
        ss = batch_symmetry_scores(pts, cands, tree, tol)
        sa = [(float(ss[i]), cands[i].copy()) for i in np.where(ss >= self.st)[0]]
        print(f"найдено: {len(sa)}")

        ra = {}
        for n in range(2, ns+1):
            print(f"    [{n}/{ns}] C{n}...", end=" ", flush=True)
            rs = batch_rotation_scores(pts, cands, n, tree, tol)
            h = [(float(rs[i]), cands[i].copy()) for i in np.where(rs >= self.st)[0]]
            print(f"найдено: {len(h)}")
            if h: ra[n] = h

        return {"symmetry_axes": _dedup(sa), "rotation_axes": {n: _dedup(v) for n,v in ra.items()},
                "cylinders": cyls, "tolerance": tol, "n_vertices": len(pf)}


# ── Вывод ─────────────────────────────────────────────────────────────────

def fv(v): return f"[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]"

def print_results(r, path=""):
    print("\n" + "="*60 + "\n  РЕЗУЛЬТАТЫ АНАЛИЗА СИММЕТРИИ")
    if path: print(f"  Файл: {path}")
    print("="*60)

    sym = r["symmetry_axes"]
    print(f"\n🔷  Оси ЗЕРКАЛЬНОЙ симметрии: {len(sym)}")
    for i,(sc,ax) in enumerate(sym,1): print(f"    #{i}  ось={fv(ax)}   score={sc:.3f}")
    if not sym: print("    Не найдено.")

    rot = r["rotation_axes"]
    if rot:
        print(f"\n🔶  Оси ВРАЩАТЕЛЬНОЙ симметрии:")
        for n in sorted(rot):
            for i,(sc,ax) in enumerate(rot[n],1):
                print(f"    C{n} #{i}  ось={fv(ax)}   score={sc:.3f}")
    else: print("\n🔶  Оси вращательной симметрии: не найдено.")

    cyls = r.get("cylinders", [])
    if cyls:
        for ct, em, lb in [("hole","🔵","ОТВЕРСТИЯ"),("boss","🟠","БОБЫШКИ"),("cylinder","⚪","ЦИЛИНДРЫ")]:
            sub = [c for c in cyls if c["type"]==ct]
            if sub:
                print(f"\n{em}  Цилиндрические {lb}: {len(sub)}")
                for i,c in enumerate(sub,1):
                    print(f"    #{i}  dia={c['radius']*2:.3f}  R={c['radius']:.3f}  "
                          f"L={c['length']:.3f}  ось={fv(c['axis'])}  центр={fv(c['center'])}  "
                          f"качество={c['fit_quality']:.2f}  граней={c['n_faces']}  "
                          f"дуга={c['angular_coverage_deg']:.0f}")
    else: print("\n🔵  Цилиндрические элементы: не обнаружено.")

    print(f"\n    Допуск: {r['tolerance']:.5f}  |  Вершин: {r['n_vertices']:,}")
    print("="*60 + "\n")


# ── Проекции для аннотации чертежей ───────────────────────────────────────

# Стандартные ортогональные виды (ГОСТ 2.305):
#   Главный (Front):  смотрим по +Y → проекция на XZ  (u=X, v=Z)
#   Сверху  (Top):    смотрим по -Z → проекция на XY  (u=X, v=Y)
#   Слева   (Left):   смотрим по +X → проекция на YZ  (u=Y, v=Z)

STANDARD_VIEWS = {
    "front": {
        "name": "Главный вид",
        "normal": np.array([0., 1., 0.]),    # направление взгляда
        "u_axis": np.array([1., 0., 0.]),     # горизонталь в проекции
        "v_axis": np.array([0., 0., 1.]),     # вертикаль в проекции
    },
    "top": {
        "name": "Вид сверху",
        "normal": np.array([0., 0., -1.]),
        "u_axis": np.array([1., 0., 0.]),
        "v_axis": np.array([0., 1., 0.]),
    },
    "left": {
        "name": "Вид слева",
        "normal": np.array([1., 0., 0.]),
        "u_axis": np.array([0., 1., 0.]),
        "v_axis": np.array([0., 0., 1.]),
    },
}

# Пороги классификации: |axis · view_normal|
_PERP_THRESHOLD = 0.95   # ось ⊥ виду → перекрестие (видим торец)
_PARA_THRESHOLD = 0.15   # ось ∥ виду → осевая линия (видим бок)


def _project_point(pt_3d, view):
    """Проекция 3D-точки на плоскость вида → (u, v)."""
    return float(pt_3d @ view["u_axis"]), float(pt_3d @ view["v_axis"])


def _project_vector(vec_3d, view):
    """Проекция 3D-вектора на плоскость вида → (du, dv), нормированный."""
    du = float(vec_3d @ view["u_axis"])
    dv = float(vec_3d @ view["v_axis"])
    length = np.hypot(du, dv)
    if length < 1e-12:
        return (0.0, 0.0), 0.0
    return (du / length, dv / length), length


def compute_cylinder_projections(cylinders, extension_factor=0.15):
    """
    Вычисляет аннотационные элементы для каждого цилиндра на каждом виде.

    Для каждого вида и цилиндра возвращает dict:
    {
        "view": "front" | "top" | "left",
        "view_name": "Главный вид" | ...,
        "cylinder_id": int,
        "cylinder_type": "hole" | "boss" | "cylinder",
        "diameter": float,
        "radius": float,

        # Тип аннотации на данном виде:
        "annotation_type": "centerline" | "crosshair" | "centerline+circle",

        # Для centerline (осевая линия в боковой проекции):
        "centerline": {
            "start": (u1, v1),       # начало осевой (с выносом)
            "end": (u2, v2),         # конец осевой
            "center": (uc, vc),      # центр
            "direction": (du, dv),   # единичный вектор направления
            "length": float,         # длина осевой (без выноса)
            "angle_deg": float,      # угол к горизонтали
        },
        # Контурные линии цилиндра (образующие):
        "contour_lines": [           # 2 линии — верхняя и нижняя образующие
            {"start": (u,v), "end": (u,v)},
            {"start": (u,v), "end": (u,v)},
        ],

        # Для crosshair (перекрестие в торцевой проекции):
        "crosshair": {
            "center": (uc, vc),
            "radius": float,         # радиус окружности в проекции
            "h_line": ((u1,v), (u2,v)),   # горизонтальная осевая
            "v_line": ((u,v1), (u,v2)),   # вертикальная осевая
        },
    }

    Parameters
    ----------
    cylinders : list[dict]
        Результат детекции цилиндров (из SymmetryDetector.detect)
    extension_factor : float
        Вынос осевых линий за пределы контура (доля от длины)

    Returns
    -------
    list[dict]  — аннотационные элементы
    """
    annotations = []

    for ci, cyl in enumerate(cylinders):
        axis_3d = cyl["axis"]
        center_3d = cyl["center"]
        R = cyl["radius"]
        L = cyl["length"]
        D = 2 * R

        for vk, view in STANDARD_VIEWS.items():
            vn = view["normal"]

            # Насколько ось совпадает с направлением взгляда
            alignment = abs(float(axis_3d @ vn))

            # Проекция центра
            cu, cv = _project_point(center_3d, view)

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
                # ─── Торцевой вид: перекрестие + окружность ───
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
                # ─── Боковой вид: осевая линия + контурные образующие ───
                (du, dv), proj_len = _project_vector(axis_3d, view)
                half_L = L / 2
                ext_L = L * extension_factor

                # Осевая линия
                total_half = half_L + ext_L
                anno["annotation_type"] = "centerline"
                anno["centerline"] = {
                    "start": (cu - du * total_half, cv - dv * total_half),
                    "end":   (cu + du * total_half, cv + dv * total_half),
                    "center": (cu, cv),
                    "body_start": (cu - du * half_L, cv - dv * half_L),
                    "body_end":   (cu + du * half_L, cv + dv * half_L),
                    "direction": (du, dv),
                    "length": L,
                    "angle_deg": float(np.degrees(np.arctan2(dv, du))),
                }

                # Контурные линии (образующие: ±R перпендикулярно оси)
                perp_u, perp_v = -dv, du  # ⊥ к направлению оси в 2D
                contours = []
                for sign in (-1, +1):
                    ou, ov = cu + sign * R * perp_u, cv + sign * R * perp_v
                    contours.append({
                        "start": (ou - du * half_L, ov - dv * half_L),
                        "end":   (ou + du * half_L, ov + dv * half_L),
                    })
                anno["contour_lines"] = contours
                anno["crosshair"] = None

            else:
                # ─── Наклонный вид: осевая + эллиптическая проекция ───
                (du, dv), proj_len = _project_vector(axis_3d, view)
                half_L = L / 2
                ext_L = L * extension_factor

                # Осевая линия (проекция оси)
                total_half = half_L + ext_L
                # Эллиптический радиус: R * sqrt(1 - alignment²)
                r_major = R
                r_minor = R * np.sqrt(1 - alignment ** 2)

                anno["annotation_type"] = "centerline+circle"
                anno["centerline"] = {
                    "start": (cu - du * total_half, cv - dv * total_half),
                    "end":   (cu + du * total_half, cv + dv * total_half),
                    "center": (cu, cv),
                    "body_start": (cu - du * half_L, cv - dv * half_L),
                    "body_end":   (cu + du * half_L, cv + dv * half_L),
                    "direction": (du, dv),
                    "length": L,
                    "angle_deg": float(np.degrees(np.arctan2(dv, du))),
                }

                # Перекрестие (эллиптическое)
                ext = r_major * (1 + extension_factor)
                anno["crosshair"] = {
                    "center": (cu, cv),
                    "radius": R,
                    "r_major": float(r_major),
                    "r_minor": float(r_minor),
                    "h_line": ((cu - ext, cv), (cu + ext, cv)),
                    "v_line": ((cu, cv - ext), (cu, cv + ext)),
                }

                # Контурные (образующие)
                perp_u, perp_v = -dv, du
                contours = []
                for sign in (-1, +1):
                    ou = cu + sign * R * perp_u
                    ov = cv + sign * R * perp_v
                    contours.append({
                        "start": (ou - du * half_L, ov - dv * half_L),
                        "end":   (ou + du * half_L, ov + dv * half_L),
                    })
                anno["contour_lines"] = contours

            annotations.append(anno)

    return annotations


def compute_symmetry_projections(symmetry_axes, rotation_axes, bbox_diag):
    """
    Проекции осей симметрии на стандартные виды.
    Осевые линии симметрии рисуются штрихпунктирной через весь вид.
    """
    annotations = []
    half_ext = bbox_diag * 0.6  # длина линии симметрии

    all_axes = []
    for sc, ax in symmetry_axes:
        all_axes.append(("mirror", sc, ax))
    for n, axlist in rotation_axes.items():
        for sc, ax in axlist:
            all_axes.append((f"C{n}", sc, ax))

    for sym_type, score, axis_3d in all_axes:
        for vk, view in STANDARD_VIEWS.items():
            vn = view["normal"]
            alignment = abs(float(axis_3d @ vn))

            # Если ось ⊥ виду — она не видна на этом виде (точка)
            if alignment > _PERP_THRESHOLD:
                continue

            (du, dv), proj_len = _project_vector(axis_3d, view)
            if proj_len < 0.01:
                continue

            annotations.append({
                "view": vk,
                "view_name": view["name"],
                "sym_type": sym_type,
                "score": score,
                "annotation_type": "symmetry_axis",
                "line": {
                    "start": (-du * half_ext, -dv * half_ext),
                    "end": (du * half_ext, dv * half_ext),
                    "direction": (du, dv),
                    "angle_deg": float(np.degrees(np.arctan2(dv, du))),
                },
            })

    return annotations


def compute_all_projections(results):
    """
    Главная функция: из результатов детекции → все аннотационные элементы.

    Returns
    -------
    dict с ключами:
        "cylinders": list — аннотации цилиндров по видам
        "symmetry":  list — аннотации осей симметрии по видам
        "by_view": {
            "front": {"cylinders": [...], "symmetry": [...]},
            "top":   {...},
            "left":  {...},
        }
    """
    cyls = results.get("cylinders", [])
    sa = results.get("symmetry_axes", [])
    ra = results.get("rotation_axes", {})
    bd = results.get("tolerance", 1.0) / results.get("n_vertices", 1) * 100

    cyl_annos = compute_cylinder_projections(cyls)
    sym_annos = compute_symmetry_projections(sa, ra, results["tolerance"] / 0.01)

    # Группировка по видам
    by_view = {}
    for vk in STANDARD_VIEWS:
        by_view[vk] = {
            "cylinders": [a for a in cyl_annos if a["view"] == vk],
            "symmetry": [a for a in sym_annos if a["view"] == vk],
        }

    return {
        "cylinders": cyl_annos,
        "symmetry": sym_annos,
        "by_view": by_view,
    }


def print_projections(proj):
    """Печать аннотационных элементов."""
    print("\n" + "=" * 70)
    print("  ПРОЕКЦИИ ДЛЯ АННОТАЦИИ ЧЕРТЕЖА")
    print("=" * 70)

    for vk in ("front", "top", "left"):
        vdata = proj["by_view"][vk]
        vname = STANDARD_VIEWS[vk]["name"]
        nc = len(vdata["cylinders"])
        ns = len(vdata["symmetry"])
        if nc == 0 and ns == 0:
            continue

        print(f"\n  📐 {vname} ({vk})")
        print(f"  {'─' * 60}")

        for a in vdata["cylinders"]:
            cid = a["cylinder_id"]
            ct = a["cylinder_type"]
            D = a["diameter"]
            at = a["annotation_type"]
            emoji = {"hole": "⊙", "boss": "◉", "cylinder": "○"}
            print(f"    {emoji.get(ct,'○')} Цилиндр #{cid+1} ({ct}) ⌀{D:.2f}  →  {at}")

            if a["centerline"]:
                cl = a["centerline"]
                print(f"       Осевая: ({cl['start'][0]:+.2f},{cl['start'][1]:+.2f}) → "
                      f"({cl['end'][0]:+.2f},{cl['end'][1]:+.2f})  "
                      f"∠{cl['angle_deg']:.1f}°  L={cl['length']:.2f}")

            if a["crosshair"]:
                ch = a["crosshair"]
                c = ch["center"]
                print(f"       Перекрестие: центр=({c[0]:+.2f},{c[1]:+.2f})  R={ch['radius']:.2f}")
                if "r_minor" in ch:
                    print(f"       Эллипс: a={ch['r_major']:.2f} b={ch['r_minor']:.2f}")

            if a["contour_lines"]:
                for j, cl in enumerate(a["contour_lines"]):
                    print(f"       Образующая {j+1}: ({cl['start'][0]:+.2f},{cl['start'][1]:+.2f}) → "
                          f"({cl['end'][0]:+.2f},{cl['end'][1]:+.2f})")

        for a in vdata["symmetry"]:
            ln = a["line"]
            print(f"    ─·─ {a['sym_type']} (score={a['score']:.2f}): "
                  f"({ln['start'][0]:+.1f},{ln['start'][1]:+.1f}) → "
                  f"({ln['end'][0]:+.1f},{ln['end'][1]:+.1f})  "
                  f"∠{ln['angle_deg']:.1f}°")

    print("\n" + "=" * 70)




def main():
    ap = argparse.ArgumentParser(description="STL симметрия + цилиндры v4.1")
    ap.add_argument("stl_file")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-cn", type=int, default=8)
    ap.add_argument("--max-pts", type=int, default=MAX_PTS)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--no-cylinders", action="store_true")
    ap.add_argument("--min-dih", type=float, default=1.0, help="Мин. двугранный угол (°)")
    ap.add_argument("--max-dih", type=float, default=60.0, help="Макс. двугранный угол (°)")
    ap.add_argument("--min-reg", type=int, default=6, help="Мин. размер региона")
    ap.add_argument("--min-arc", type=float, default=45.0, help="Мин. дуга покрытия (°)")
    ap.add_argument("--sharp-angle", type=float, default=15.0, help="Порог острого ребра (°)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    a = ap.parse_args()

# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="STL симметрия + цилиндры v4.2 + проекции")
    ap.add_argument("stl_file")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-cn", type=int, default=8)
    ap.add_argument("--max-pts", type=int, default=MAX_PTS)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--no-cylinders", action="store_true")
    ap.add_argument("--min-dih", type=float, default=1.0, help="Мин. двугранный угол (°)")
    ap.add_argument("--max-dih", type=float, default=60.0, help="Макс. двугранный угол (°)")
    ap.add_argument("--min-reg", type=int, default=6, help="Мин. размер региона")
    ap.add_argument("--min-arc", type=float, default=45.0, help="Мин. дуга покрытия (°)")
    ap.add_argument("--sharp-angle", type=float, default=15.0, help="Порог острого ребра (°)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--projections", action="store_true",
                    help="Вывести проекции осевых линий")
    ap.add_argument("--json", type=str, default=None, metavar="FILE.json",
                    help="Экспорт проекций в JSON для пайплайна аннотации")
    a = ap.parse_args()

    det = SymmetryDetector(a.tol, a.threshold,
                           36 if a.dense else 24, 18 if a.dense else 12,
                           a.max_cn, a.max_pts,
                           not a.no_cylinders, a.verbose,
                           min_dih=a.min_dih, max_dih=a.max_dih,
                           min_reg=a.min_reg, min_arc=a.min_arc,
                           sharp_angle=a.sharp_angle)
    t0 = time.perf_counter()
    res = det.detect(a.stl_file)
    print_results(res, a.stl_file)

    # Проекции
    proj = compute_all_projections(res)
    if a.projections:
        print_projections(proj)

    if a.json:
        import json

        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

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
            } for i, c in enumerate(res.get("cylinders", []))],
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
        print(f"💾  JSON: {a.json}")

    print(f"⏱  Время: {time.perf_counter()-t0:.2f} сек")
    if a.viz or a.save:
        visualize(a.stl_file, res, save_path=a.save)


# ── Демо ──────────────────────────────────────────────────────────────────

def demo():
    import tempfile
    from stl import mesh as sm

    def make_cyl(p, n=32, r=1.0, h=2.0):
        a = np.linspace(0,2*np.pi,n,endpoint=False)
        T = np.c_[r*np.cos(a),r*np.sin(a),np.full(n,h/2)]
        B = np.c_[r*np.cos(a),r*np.sin(a),np.full(n,-h/2)]
        t = []
        for i in range(n):
            j=(i+1)%n
            t += [[T[i],B[i],T[j]],[B[i],B[j],T[j]],
                  [T[i],T[j],[0,0,h/2]],[B[i],[0,0,-h/2],B[j]]]
        m = sm.Mesh(np.zeros(len(t), dtype=sm.Mesh.dtype))
        for k,tri in enumerate(np.array(t)): m.vectors[k]=tri
        m.save(p)

    def make_box(p, lx=2, ly=1.5, lz=1):
        v = np.array([[sx*lx/2,sy*ly/2,sz*lz/2] for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])
        f = [[0,1,2],[1,3,2],[4,6,5],[5,6,7],[0,4,1],[1,4,5],[2,3,6],[3,7,6],[0,2,4],[2,6,4],[1,5,3],[3,5,7]]
        t = np.array([[v[i[0]],v[i[1]],v[i[2]]] for i in f])
        m = sm.Mesh(np.zeros(len(t), dtype=sm.Mesh.dtype))
        for k,tri in enumerate(t): m.vectors[k]=tri
        m.save(p)

    def make_plate(p, ps=10, ph=1, hr=0.5, hpos=None, ns=24):
        t = []
        hs,hh = ps/2, ph/2
        v = np.array([[sx*hs,sy*hs,sz*hh] for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])
        fi = [[0,1,2],[1,3,2],[4,6,5],[5,6,7],[0,4,1],[1,4,5],[2,3,6],[3,7,6],[0,2,4],[2,6,4],[1,5,3],[3,5,7]]
        for f in fi: t.append([v[f[0]],v[f[1]],v[f[2]]])
        if hpos is None: hpos = [(2,2),(-2,2),(2,-2),(-2,-2),(0,0)]
        for hx,hy in hpos:
            a = np.linspace(0,2*np.pi,ns,endpoint=False)
            for i in range(ns):
                j=(i+1)%ns
                a1,a2=a[i],a[j]
                p1=[hx+hr*np.cos(a1),hy+hr*np.sin(a1),hh]
                p2=[hx+hr*np.cos(a1),hy+hr*np.sin(a1),-hh]
                p3=[hx+hr*np.cos(a2),hy+hr*np.sin(a2),hh]
                p4=[hx+hr*np.cos(a2),hy+hr*np.sin(a2),-hh]
                t.append([p1,p3,p2]); t.append([p2,p3,p4])
        m = sm.Mesh(np.zeros(len(t), dtype=sm.Mesh.dtype))
        for k,tri in enumerate(np.array(t)): m.vectors[k]=tri
        m.save(p)

    td = tempfile.mkdtemp()
    det = SymmetryDetector(tol_factor=0.02, score_threshold=0.80,
                           detect_cylinders=True, verbose=True)
    for nm, bld in [("Цилиндр", lambda p: make_cyl(p,n=32)),
                    ("Параллелепипед", lambda p: make_box(p)),
                    ("Пластина с 5 отверстиями", lambda p: make_plate(p))]:
        fp = os.path.join(td, nm+".stl")
        bld(fp)
        print(f"\n{'='*50}\n--- {nm} ---")
        t0 = time.perf_counter()
        r = det.detect(fp)
        print(f"⏱  {time.perf_counter()-t0:.3f} сек")
        print_results(r, nm)


# ── Визуализация ──────────────────────────────────────────────────────────

def visualize(stl_path, results, save_path=None):
    import matplotlib
    if save_path: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D

    ms = mesh.Mesh.from_file(stl_path)
    tris = ms.vectors; ctr = tris.reshape(-1,3).mean(axis=0)
    tc = tris - ctr; bp = tc.reshape(-1,3)
    hl = np.linalg.norm(bp.max(0)-bp.min(0))*0.65

    sa = results["symmetry_axes"]; ra = results["rotation_axes"]
    cyls = results.get("cylinders", [])
    rv = [av for al in ra.values() for _,av in al]

    def acol(av):
        s = any(abs(np.dot(av,sv))>0.98 for _,sv in sa)
        r = any(abs(np.dot(av,rv_))>0.98 for rv_ in rv)
        if s and r: return "#FFD700"
        return "#2979FF" if s else "#FF3D00"

    fig = plt.figure(figsize=(13,9), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e"); ax.grid(False)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor("#333355")

    poly = Poly3DCollection(tc, alpha=0.13, linewidth=0)
    poly.set_facecolor("#4fc3f7"); ax.add_collection3d(poly)
    for tri in tc:
        xs=[tri[i%3][0] for i in range(4)]
        ys=[tri[i%3][1] for i in range(4)]
        zs=[tri[i%3][2] for i in range(4)]
        ax.plot(xs,ys,zs, color="#1565c0", lw=0.18, alpha=0.3)

    drawn, labels = [], []
    def draw_ax(av, col, lbl):
        for dv in drawn:
            if abs(np.dot(av,dv))>0.997: return
        drawn.append(av.copy())
        p1,p2 = -av*hl, av*hl
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],
                color=col, lw=2.5, alpha=0.93, zorder=5)
        for s in (-1,1): ax.scatter(*(av*hl*s), color=col, s=28, alpha=0.95, zorder=6)
        tip = av*hl*1.09
        if not any(abs(np.dot(av,d))>0.997 for d in labels):
            labels.append(av.copy())
            ax.text(*tip, f"  {lbl}", color=col, fontsize=7.5, fontweight="bold", alpha=0.92)

    for sc,av in sa: draw_ax(av, acol(av), f"M {sc:.2f}")
    for n,al in sorted(ra.items()):
        for sc,av in al: draw_ax(av, acol(av), f"C{n} {sc:.2f}")

    for c in cyls:
        col = {"hole":"#00E676","boss":"#FF9100"}.get(c["type"],"#B0BEC5")
        ct,ad = c["center"], c["axis"]
        ll = max(c["length"]*0.7, hl*0.3)
        p1,p2 = ct-ad*ll, ct+ad*ll
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],
                color=col, lw=2, alpha=0.85, ls="--", zorder=5)
        ax.scatter(*ct, color=col, s=40, marker="o", alpha=0.9, zorder=6)
        ax.text(*p2, f"  dia={c['radius']*2:.1f}", color=col, fontsize=7, fontweight="bold", alpha=0.85)
        th = np.linspace(0,2*np.pi,40)
        ud = np.array([1.,0,0])
        if abs(np.dot(ud,ad))>0.9: ud=np.array([0.,1,0])
        ud -= np.dot(ud,ad)*ad; ud /= np.linalg.norm(ud)
        vd = np.cross(ad, ud)
        circ = ct[None,:] + c["radius"]*(np.cos(th)[:,None]*ud[None,:]+np.sin(th)[:,None]*vd[None,:])
        ax.plot(circ[:,0],circ[:,1],circ[:,2], color=col, lw=1.2, alpha=0.7, zorder=5)

    leg = [Line2D([0],[0],color="#2979FF",lw=2.5,label="Зеркальная"),
           Line2D([0],[0],color="#FF3D00",lw=2.5,label="Вращательная"),
           Line2D([0],[0],color="#FFD700",lw=2.5,label="Зерк.+вращ.")]
    if any(c["type"]=="hole" for c in cyls):
        leg.append(Line2D([0],[0],color="#00E676",lw=2,ls="--",label="Отверстие"))
    if any(c["type"]=="boss" for c in cyls):
        leg.append(Line2D([0],[0],color="#FF9100",lw=2,ls="--",label="Бобышка"))
    ax.legend(handles=leg, loc="upper left", facecolor="#0d0d1a",
              edgecolor="#445", labelcolor="white", fontsize=8.5, framealpha=0.9)

    t = os.path.basename(stl_path)
    ns,nr,nc = len(sa), sum(len(v) for v in ra.values()), len(cyls)
    ax.set_title(f"{t}\nЗерк.: {ns}  Вращ.: {nr}  Цилиндров: {nc}",
                 color="white", fontsize=10, pad=12)
    ax.tick_params(colors="#555577", labelsize=6)
    mx = np.abs(bp).max()*1.15
    ax.set_xlim(-mx,mx); ax.set_ylim(-mx,mx); ax.set_zlim(-mx,mx)
    for l in "XYZ": getattr(ax,f"set_{l.lower()}label")(l, color="#7777aa", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"💾  Сохранено: {save_path}")
    else: plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--demo": main()
    else: demo()