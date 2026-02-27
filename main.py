"""
Точка входа: генерация ЕСКД-чертежа из STL-файла.

Использование:
    python main.py <stl_file> [--output OUTPUT] [--format {A4,A3,A2,A1,A0}]

Пример:
    python main.py "detail.stl" --output "drawing.svg"
    python main.py "detail.stl" --output "drawing.svg" --dxf  # + DXF
    python main.py "detail.stl" --config project.eskd.json    # с конфигом
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Обеспечить поддержку Unicode (emoji и т.п.) на Windows-консоли
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

from stl_drawing.drawing.sheet import ESKDDrawingSheet
from stl_drawing.io.stl_loader import STLLoadError, load_stl
from stl_drawing.orientation.pca import orient_model_by_normals
from stl_drawing.orientation.view_scorer import select_best_front_and_reorient
from stl_drawing.projection.view_processor import VIEW_DIRECTIONS, VIEW_MATRICES, ViewProcessor
from stl_drawing.topology.processor import TopologyProcessor
from stl_drawing.features import CylinderDetector
from stl_drawing.project_config import ProjectConfig, load_config, apply_config_to_globals

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
    config: Optional[ProjectConfig] = None,
    output_dxf: Optional[str] = None,
    thickness_scale: float = 1.0,
) -> str:
    """Полный пайплайн: STL → ЕСКД SVG (+ DXF).

    Шаги:
      1. Загрузка STL.
      2. PCA-ориентация по нормалям граней.
      3. Выбор лучшего главного вида, переориентация.
      4. Топологическая обработка: классификация рёбер.
      5. Обработка 6 видов (проекция + hidden line removal).
      6. Генерация SVG ЕСКД (+ DXF при наличии output_dxf).

    Args:
        stl_path: путь к STL-файлу.
        output_svg: путь к выходному SVG.
        designation: обозначение документа для штампа.
        part_name: наименование изделия для штампа.
        org_name: наименование организации для штампа.
        surname: фамилия подписанта для штампа.
        config: конфигурация проекта (опционально).
        output_dxf: путь к выходному DXF (опционально).

    Returns:
        Путь к сохранённому SVG-файлу.

    Raises:
        STLLoadError: если файл не загружен.
        RuntimeError: при ошибках в пайплайне.
    """
    # Применить конфигурацию к глобальным параметрам
    if config is not None:
        apply_config_to_globals(config)
        # Использовать значения из конфига если не указаны явно
        if not designation and config.title_block.document_number:
            designation = config.title_block.document_number
        if not part_name and config.title_block.document_name:
            part_name = config.title_block.document_name
        if not org_name and config.title_block.organization:
            org_name = config.title_block.organization
        if not surname and config.title_block.designer:
            surname = config.title_block.designer
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
        centerlines = _compute_centerlines(cylinders, view_name, final_verts)
        sheet.add_view_data(view_name, projected, visible, hidden,
                            centerlines=centerlines)

    # Передать данные цилиндров для оразмеривания
    sheet.set_cylinders(cylinders)

    # --- Шаг 6: Генерация чертежа ---
    logger.info("=" * 60)
    logger.info("Шаг 6: Генерация ЕСКД-чертежа")
    logger.info("=" * 60)
    sheet.set_metadata(designation, part_name, org_name, surname)
    result_path = sheet.generate_drawing(output_svg, thickness_scale=thickness_scale)

    logger.info("Готово. SVG: %s", result_path)

    # --- Шаг 7: Генерация DXF (опционально) ---
    if output_dxf:
        logger.info("=" * 60)
        logger.info("Шаг 7: Генерация DXF-чертежа")
        logger.info("=" * 60)
        _generate_dxf(sheet, output_dxf)
        logger.info("Готово. DXF: %s", output_dxf)

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


def _generate_dxf(sheet: ESKDDrawingSheet, output_path: str) -> None:
    """Генерация DXF-файла из данных чертежа.

    Args:
        sheet: объект ESKDDrawingSheet с данными видов.
        output_path: путь к выходному DXF-файлу.
    """
    from stl_drawing.drawing.dxf_renderer import DxfRenderer, DxfStyle

    renderer = DxfRenderer()

    # Получаем размеры формата из sheet
    width_mm = sheet.sheet_w
    height_mm = sheet.sheet_h
    scale = sheet.scale

    renderer.create_drawing(width_mm, height_mm)

    # Добавляем рамку
    frame_style = DxfStyle(layer='FRAME')
    margin = 5.0
    renderer.add_rectangle((margin, margin), width_mm - 2*margin, height_mm - 2*margin, frame_style)

    # Добавляем линии из всех видов
    contour_style = DxfStyle(layer='CONTOUR')
    hidden_style = DxfStyle(layer='HIDDEN')
    center_style = DxfStyle(layer='CENTER')

    for view_name, view_layout in sheet.layout.items():
        view_data = sheet.active_views.get(view_name, {})

        # Позиция вида на листе
        translate_x = view_layout['x'] + view_layout['offset_x']
        translate_y = view_layout['y'] + view_layout['offset_y']

        # Visible lines (формат: (start_array, end_array) или (x1, y1, x2, y2))
        for line in view_data.get('visible', []):
            if isinstance(line, tuple) and len(line) == 2:
                # Формат: (array([x1,y1]), array([x2,y2]))
                start, end = line
                x1, y1 = float(start[0]), float(start[1])
                x2, y2 = float(end[0]), float(end[1])
            else:
                x1, y1, x2, y2 = line[:4]
            renderer.add_line(
                (x1 * scale + translate_x, y1 * scale + translate_y),
                (x2 * scale + translate_x, y2 * scale + translate_y),
                contour_style
            )

        # Hidden lines
        for line in view_data.get('hidden', []):
            if isinstance(line, tuple) and len(line) == 2:
                start, end = line
                x1, y1 = float(start[0]), float(start[1])
                x2, y2 = float(end[0]), float(end[1])
            else:
                x1, y1, x2, y2 = line[:4]
            renderer.add_line(
                (x1 * scale + translate_x, y1 * scale + translate_y),
                (x2 * scale + translate_x, y2 * scale + translate_y),
                hidden_style
            )

        # Centerlines
        for cl in view_data.get('centerlines', []):
            if cl.get('type') == 'centerline':
                start = cl['start']
                end = cl['end']
                renderer.add_line(
                    (start[0] * scale + translate_x, start[1] * scale + translate_y),
                    (end[0] * scale + translate_x, end[1] * scale + translate_y),
                    center_style
                )
            elif cl.get('type') == 'crosshair':
                cx, cy = cl['center']
                r = cl['radius'] * scale
                # Horizontal
                renderer.add_line(
                    (cx * scale - r*1.2 + translate_x, cy * scale + translate_y),
                    (cx * scale + r*1.2 + translate_x, cy * scale + translate_y),
                    center_style
                )
                # Vertical
                renderer.add_line(
                    (cx * scale + translate_x, cy * scale - r*1.2 + translate_y),
                    (cx * scale + translate_x, cy * scale + r*1.2 + translate_y),
                    center_style
                )

    renderer.save(output_path)


def _compute_centerlines(
    cylinders: List[dict],
    view_name: str,
    model_verts: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Вычислить 2D-проекции осевых линий цилиндров на данный вид.

    Для каждого цилиндра:
      - Если ось почти перпендикулярна плоскости вида (торец):
        рисуем перекрестие (горизонтальная + вертикальная осевые).
      - Иначе: рисуем осевую линию вдоль проекции оси.
        Длина осевой определяется полным габаритом тела вращения
        (вершины модели вблизи оси цилиндра), а не только длиной
        цилиндрической поверхности.

    Координаты возвращаются в единицах модели (как visible/hidden lines).

    Args:
        cylinders: результат _detect_cylinders().
        view_name: имя вида ('front', 'top', и т.д.).
        model_verts: вершины модели (N, 3) для определения габарита тела.

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
            # Осевая линия вдоль проекции оси.
            # Определяем полный габарит тела вращения вдоль оси:
            # ищем вершины модели вблизи оси цилиндра (в пределах 3R)
            # и берём их крайние проекции на ось.
            half = L / 2
            min_proj = -half
            max_proj = half

            if model_verts is not None and len(model_verts) > 0:
                offsets = model_verts - center
                along_axis = offsets @ axis              # проекция на ось
                perp = offsets - np.outer(along_axis, axis)  # перпендикулярная компонента
                dist_from_axis = np.linalg.norm(perp, axis=1)

                # Вершины в пределах 3R от оси — часть тела вращения
                nearby_mask = dist_from_axis < R * 3.0
                if nearby_mask.any():
                    nearby_proj = along_axis[nearby_mask]
                    body_min = float(nearby_proj.min())
                    body_max = float(nearby_proj.max())
                    # Используем максимум из габарита тела и длины цилиндра
                    min_proj = min(min_proj, body_min)
                    max_proj = max(max_proj, body_max)

            p1 = center + axis * min_proj
            p2 = center + axis * max_proj
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
        "--dxf",
        action="store_true",
        help="Дополнительно генерировать DXF-файл.",
    )
    parser.add_argument(
        "--dxf-output",
        default=None,
        dest="dxf_output",
        help="Путь к выходному DXF-файлу (по умолчанию: как SVG, но с .dxf).",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Путь к конфигурационному файлу .eskd.json.",
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
    parser.add_argument(
        "--thickness-scale", "-t",
        type=float,
        default=1.0,
        dest="thickness_scale",
        help="Множитель толщины всех линий (0.7 = −30%%, 1.5 = +50%%). По умолчанию: 1.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Загрузить конфигурацию
    config = load_config(
        stl_path=args.stl_file,
        explicit_config=args.config,
    )

    # Определить путь к DXF
    output_dxf = None
    if args.dxf or args.dxf_output:
        if args.dxf_output:
            output_dxf = args.dxf_output
        else:
            # Заменить расширение .svg на .dxf
            output_dxf = str(Path(args.output).with_suffix('.dxf'))

    try:
        run_pipeline(
            args.stl_file,
            args.output,
            designation=args.designation,
            part_name=args.part_name,
            org_name=args.org_name,
            surname=args.surname,
            config=config,
            output_dxf=output_dxf,
            thickness_scale=args.thickness_scale,
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
