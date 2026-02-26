"""
Загрузка и нормализация STL-файлов.

Поддерживает:
- Бинарный формат STL (автодетекция)
- ASCII формат STL (автодетекция)

Единственная ответственность: прочитать файл и вернуть
дедуплицированные вершины + индексы граней.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from stl import mesh

logger = logging.getLogger(__name__)


class STLFormat(Enum):
    """STL file format type."""
    BINARY = "binary"
    ASCII = "ascii"
    UNKNOWN = "unknown"


@dataclass
class STLInfo:
    """Metadata about loaded STL file."""
    filepath: str
    format: STLFormat
    file_size_bytes: int
    n_triangles: int
    n_unique_vertices: int
    solid_name: Optional[str] = None

    @property
    def file_size_kb(self) -> float:
        """File size in kilobytes."""
        return self.file_size_bytes / 1024

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)


class STLLoadError(Exception):
    """Ошибка при загрузке или разборе STL-файла."""


def detect_stl_format(filepath: str) -> Tuple[STLFormat, Optional[str]]:
    """Detect STL file format (binary vs ASCII).

    ASCII STL files start with 'solid' keyword followed by optional name.
    Binary STL files have 80-byte header (may contain 'solid' but not at start
    of valid ASCII content).

    Args:
        filepath: Path to STL file

    Returns:
        Tuple of (format, solid_name or None)

    Raises:
        STLLoadError: if file cannot be read
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(80)
    except FileNotFoundError:
        raise STLLoadError(f"Файл не найден: {filepath!r}")
    except Exception as exc:
        raise STLLoadError(f"Не удалось прочитать файл {filepath!r}: {exc}") from exc

    if len(header) < 80:
        # File too small for valid binary STL
        # Check if it's ASCII
        try:
            text = header.decode('ascii', errors='ignore').strip().lower()
            if text.startswith('solid'):
                solid_name = text[5:].strip().split('\n')[0].strip() or None
                return STLFormat.ASCII, solid_name
        except Exception:
            pass
        return STLFormat.UNKNOWN, None

    # Try to detect ASCII format
    try:
        # Read first ~1KB to check for ASCII patterns
        with open(filepath, 'r', encoding='ascii', errors='strict') as f:
            first_chunk = f.read(1024)
            first_line = first_chunk.strip().lower()

            if first_line.startswith('solid'):
                # Verify it's really ASCII by checking for 'facet' keyword
                if 'facet' in first_chunk.lower() or 'endsolid' in first_chunk.lower():
                    solid_name = first_line[5:].split('\n')[0].strip() or None
                    return STLFormat.ASCII, solid_name
    except (UnicodeDecodeError, Exception):
        pass

    # Default to binary
    # Try to extract solid name from binary header (first 80 bytes, null-terminated)
    try:
        solid_name = header.split(b'\x00')[0].decode('ascii', errors='ignore').strip()
        if solid_name and not solid_name.startswith('solid'):
            solid_name = None
        elif solid_name and solid_name.startswith('solid'):
            solid_name = solid_name[5:].strip() or None
    except Exception:
        solid_name = None

    return STLFormat.BINARY, solid_name


def load_stl(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузить STL-файл и вернуть уникальные вершины и грани.

    Автоматически определяет формат файла (бинарный или ASCII).
    Дедупликация вершин выполняется по точному совпадению координат,
    округлённых до 6 знаков после запятой.

    Args:
        filepath: Путь к STL-файлу (бинарный или ASCII).

    Returns:
        vertices: массив формы (N, 3), float64 — уникальные вершины.
        faces:    массив формы (M, 3), int32  — индексы вершин каждого треугольника.

    Raises:
        STLLoadError: если файл не найден, повреждён или содержит 0 треугольников.
    """
    # Detect format first
    stl_format, solid_name = detect_stl_format(filepath)
    file_size = os.path.getsize(filepath)

    logger.info("Загрузка STL: %s (формат: %s, размер: %.1f KB)",
                filepath, stl_format.value, file_size / 1024)
    if solid_name:
        logger.debug("Solid name: %s", solid_name)

    try:
        stl_mesh = mesh.Mesh.from_file(filepath)
    except FileNotFoundError:
        raise STLLoadError(f"Файл не найден: {filepath!r}")
    except Exception as exc:
        raise STLLoadError(f"Не удалось прочитать STL-файл {filepath!r}: {exc}") from exc

    if len(stl_mesh.vectors) == 0:
        raise STLLoadError(f"STL-файл {filepath!r} не содержит треугольников.")

    vertices: List[np.ndarray] = []
    faces: List[Tuple[int, int, int]] = []
    vertex_index: dict = {}

    for triangle in stl_mesh.vectors:
        face_indices = []
        for raw_vertex in triangle:
            # Округляем до 6 знаков, чтобы объединить вершины с небольшими
            # погрешностями численного представления из STL.
            key = tuple(round(float(c), 6) for c in raw_vertex)
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append(np.array(key, dtype=np.float64))
            face_indices.append(vertex_index[key])
        faces.append(tuple(face_indices))

    vertices_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int32)

    logger.info(
        "Загружено: %d уникальных вершин, %d граней.",
        len(vertices_arr),
        len(faces_arr),
    )
    return vertices_arr, faces_arr


def load_stl_with_info(filepath: str) -> Tuple[np.ndarray, np.ndarray, STLInfo]:
    """Load STL file and return vertices, faces, and metadata.

    Extended version of load_stl that also returns STLInfo with file metadata.

    Args:
        filepath: Path to STL file (binary or ASCII)

    Returns:
        vertices: array of shape (N, 3), float64 — unique vertices
        faces: array of shape (M, 3), int32 — vertex indices per triangle
        info: STLInfo with file metadata

    Raises:
        STLLoadError: if file not found, corrupted, or contains 0 triangles
    """
    stl_format, solid_name = detect_stl_format(filepath)
    file_size = os.path.getsize(filepath)

    vertices, faces = load_stl(filepath)

    info = STLInfo(
        filepath=filepath,
        format=stl_format,
        file_size_bytes=file_size,
        n_triangles=len(faces),
        n_unique_vertices=len(vertices),
        solid_name=solid_name,
    )

    return vertices, faces, info
