"""
Точка входа: генерация ЕСКД-чертежа из STL-файла.

Использование:
    python main.py <stl_file> [--output OUTPUT] [--format {A4,A3,A2,A1,A0}]

Пример:
    python main.py "detail.stl" --output "drawing.svg"
"""

import argparse
import logging
import sys
from typing import Dict, List

import numpy as np

from stl_drawing.drawing.sheet import ESKDDrawingSheet
from stl_drawing.io.stl_loader import STLLoadError, load_stl
from stl_drawing.orientation.pca import orient_model_by_normals
from stl_drawing.orientation.view_scorer import select_best_front_and_reorient
from stl_drawing.projection.view_processor import VIEW_DIRECTIONS, VIEW_MATRICES, ViewProcessor
from stl_drawing.topology.processor import TopologyProcessor
from stl_symmetry_detector import CylinderDetector

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("stl_drawing.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Виды, включаемые в ЕСКД-чертёж (6 ортогональных проекций)
DRAWING_VIEWS = ["front", "back", "top", "bottom", "left", "right"]


# ---------------------------------------------------------------------------
# Пайплайн
# ---------------------------------------------------------------------------

def run_pipeline(
    stl_path: str,
    output_svg: str,
    designation: str = "",
    part_name: str = "",
    org_name: str = "",
    surname: str = "",
) -> str:
    """Полный пайплайн: STL → ЕСКД SVG.

    Шаги:
      1. Загрузка STL.
      2. PCA-ориентация по нормалям граней.
      3. Выбор лучшего главного вида, переориентация.
      4. Топологическая обработка: классификация рёбер.
      5. Обработка 6 видов (проекция + hidden line removal).
      6. Генерация SVG ЕСКД.

    Args:
        stl_path: путь к STL-файлу.
        output_svg: путь к выходному SVG.
        designation: обозначение документа для штампа.
        part_name: наименование изделия для штампа.
        org_name: наименование организации для штампа.
        surname: фамилия подписанта для штампа.

    Returns:
        Путь к сохранённому SVG-файлу.

    Raises:
        STLLoadError: если файл не загружен.
        RuntimeError: при ошибках в пайплайне.
    """
    # --- Шаг 1: Загрузка STL ---
    logger.info("=" * 60)
    logger.info("Шаг 1: Загрузка STL")
    logger.info("=" * 60)
    vertices, faces = load_stl(stl_path)

    # --- Шаг 2: PCA-ориентация ---
    logger.info("=" * 60)
    logger.info("Шаг 2: PCA-ориентация по нормалям граней")
    logger.info("=" * 60)
    oriented_verts, _pca_rotation = orient_model_by_normals(vertices, faces)

    # Пересчитываем нормали для ориентированной геометрии
    pca_normals = _compute_face_normals(oriented_verts, faces)

    # --- Шаг 3: Выбор главного вида ---
    logger.info("=" * 60)
    logger.info("Шаг 3: Выбор лучшего главного вида")
    logger.info("=" * 60)
    final_verts, final_normals = select_best_front_and_reorient(
        oriented_verts, faces, pca_normals
    )

    # --- Шаг 4: Топология ---
    logger.info("=" * 60)
    logger.info("Шаг 4: Топологическая обработка рёбер")
    logger.info("=" * 60)
    topo = TopologyProcessor(final_verts, faces)
    sharp_edges = topo.classify_edges()

    view_proc = ViewProcessor(
        vertices=topo.vertices,
        sharp_edges=sharp_edges,
        faces=topo.faces,
        edge_faces=topo.edge_faces,
        face_normals=topo.face_normals,
        smooth_edges=topo.smooth_edges,
    )

    # --- Шаг 4.5: Детекция цилиндров (осевые линии) ---
    logger.info("=" * 60)
    logger.info("Шаг 4.5: Детекция цилиндрических элементов")
    logger.info("=" * 60)
    cylinders = _detect_cylinders(final_verts, faces)
    logger.info("Обнаружено цилиндров: %d", len(cylinders))

    # --- Шаг 5: Обработка видов ---
    logger.info("=" * 60)
    logger.info("Шаг 5: Генерация видов (%d проекций)", len(DRAWING_VIEWS))
    logger.info("=" * 60)
    sheet = ESKDDrawingSheet()

    for view_name in DRAWING_VIEWS:
        logger.info("  Обработка вида: %s", view_name)
        projected, visible, hidden = view_proc.process_view(view_name)
        centerlines = _compute_centerlines(cylinders, view_name)
        sheet.add_view_data(view_name, projected, visible, hidden,
                            centerlines=centerlines)

    # --- Шаг 6: Генерация чертежа ---
    logger.info("=" * 60)
    logger.info("Шаг 6: Генерация ЕСКД-чертежа")
    logger.info("=" * 60)
    sheet.set_metadata(designation, part_name, org_name, surname)
    result_path = sheet.generate_drawing(output_svg)

    logger.info("Готово. Чертёж: %s", result_path)
    return result_path


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Вычислить единичные нормали граней."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float64)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    return (cross / lengths).astype(np.float32)


def _detect_cylinders(vertices: np.ndarray, faces: np.ndarray) -> List[dict]:
    """Детекция цилиндрических элементов для построения осевых линий.

    Args:
        vertices: ориентированные вершины (N, 3).
        faces: грани (M, 3).

    Returns:
        Список цилиндров с полями axis, center, radius, length, type.
    """
    vectors = vertices[faces].astype(np.float64)
    e1 = vectors[:, 1] - vectors[:, 0]
    e2 = vectors[:, 2] - vectors[:, 0]
    cr = np.cross(e1, e2)
    nm = np.linalg.norm(cr, axis=1, keepdims=True)
    nm = np.where(nm < 1e-12, 1.0, nm)

    md = {
        "vectors": vectors,
        "normals": cr / nm,
        "centroids": vectors.mean(axis=1),
        "areas": np.linalg.norm(cr, axis=1) * 0.5,
    }
    offset = vectors.reshape(-1, 3).mean(axis=0)

    detector = CylinderDetector()
    cyls = detector.detect(md, offset)

    # Восстановить центры в исходную систему координат
    for c in cyls:
        c["center"] = c["center"] + offset

    return cyls


# Порог выравнивания оси цилиндра с направлением взгляда
_AXIS_PERP_THRESHOLD = 0.95  # ось ⊥ плоскости вида → перекрестие


def _compute_centerlines(
    cylinders: List[dict],
    view_name: str,
) -> List[Dict]:
    """Вычислить 2D-проекции осевых линий цилиндров на данный вид.

    Для каждого цилиндра:
      - Если ось почти перпендикулярна плоскости вида (торец):
        рисуем перекрестие (горизонтальная + вертикальная осевые).
      - Иначе: рисуем осевую линию вдоль проекции оси.

    Координаты возвращаются в единицах модели (как visible/hidden lines).

    Args:
        cylinders: результат _detect_cylinders().
        view_name: имя вида ('front', 'top', и т.д.).

    Returns:
        Список словарей с описанием осевых линий.
    """
    if not cylinders or view_name not in VIEW_MATRICES:
        return []

    M = VIEW_MATRICES[view_name]
    view_dir = VIEW_DIRECTIONS[view_name]
    result: List[Dict] = []

    for cyl in cylinders:
        axis = np.asarray(cyl["axis"], dtype=np.float64)
        center = np.asarray(cyl["center"], dtype=np.float64)
        L = float(cyl["length"])
        R = float(cyl["radius"])

        alignment = abs(float(axis @ view_dir))

        if alignment > _AXIS_PERP_THRESHOLD:
            # Ось направлена на наблюдателя → перекрестие
            center_proj = (center @ M.T)[:2]
            result.append({
                'type': 'crosshair',
                'center': center_proj,
                'radius': R,
            })
        else:
            # Осевая линия вдоль проекции оси (без выноса — он добавится в мм при рендеринге)
            half = L / 2
            p1 = center - axis * half
            p2 = center + axis * half
            p1_proj = (p1 @ M.T)[:2]
            p2_proj = (p2 @ M.T)[:2]

            length_2d = float(np.linalg.norm(p2_proj - p1_proj))
            if length_2d < R * 0.1:
                continue

            result.append({
                'type': 'centerline',
                'start': p1_proj,
                'end': p2_proj,
            })

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Генерация ЕСКД-чертежа из STL-файла.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stl_file",
        help="Путь к входному STL-файлу.",
    )
    parser.add_argument(
        "--output", "-o",
        default="unified_eskd_drawing.svg",
        help="Путь к выходному SVG-файлу (по умолчанию: unified_eskd_drawing.svg).",
    )
    parser.add_argument(
        "--designation", "-d",
        default="",
        help="Обозначение документа (Графа 2, 26), напр. 'АБВГ.123456.001'.",
    )
    parser.add_argument(
        "--part-name", "-p",
        default="",
        dest="part_name",
        help="Наименование изделия (Графа 1), напр. 'Швеллер 16П L=751'.",
    )
    parser.add_argument(
        "--org-name", "-n",
        default="",
        dest="org_name",
        help="Наименование организации (Графа 8), напр. 'ООО ПромТех'.",
    )
    parser.add_argument(
        "--surname", "-s",
        default="",
        help="Фамилия подписанта (Разраб./Пров./Т.контр./Н.контр./Утв.).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        run_pipeline(
            args.stl_file,
            args.output,
            designation=args.designation,
            part_name=args.part_name,
            org_name=args.org_name,
            surname=args.surname,
        )
    except STLLoadError as exc:
        logger.critical("Ошибка загрузки STL: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.critical("Ошибка конфигурации: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Неожиданная ошибка: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
