# STL-to-ESKD Drawing Generator

Автоматическая генерация ЕСКД-чертежей (SVG) из 3D-моделей (STL).

## Использование

```bash
python main.py <stl_file> [--output OUTPUT] [--format {A4,A3,A2,A1,A0}]
python main.py fuel.stl -o drawing.svg
python main.py Shveller_16P.stl --designation "АБВГ.123456.001" --part-name "Швеллер 16П"
```

**Аргументы CLI:**
- `--output` / `-o` — путь к выходному SVG
- `--format` — принудительный формат листа (по умолчанию автовыбор)
- `--designation` — обозначение документа (ячейка Гр.2 штампа)
- `--part-name` — наименование детали (ячейка Гр.1)
- `--org-name` — организация (ячейка Гр.8)
- `--surname` — фамилия (строка «Разраб.»)

## Зависимости

| Библиотека | Назначение |
|------------|------------|
| `numpy` | Массивы, линейная алгебра |
| `numpy-stl` | Чтение STL |
| `rtree` | Пространственный индекс (R-tree) |
| `svgwrite` | Генерация SVG |

---

## Пайплайн (6 шагов)

```
STL-файл
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 1: Загрузка STL                         │
│   stl_drawing/io/stl_loader.py               │
│   load_stl(path) → (vertices, faces)         │
│   Дедупликация вершин, numpy-массивы         │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 2: PCA-ориентация                       │
│   stl_drawing/orientation/pca.py             │
│   orient_model_by_normals(verts, faces)      │
│   Ковариационная матрица нормалей×площади,   │
│   snap к ближайшим осям ±X/Y/Z              │
│   Пропускается если OBB/AABB > 0.95         │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 3: Выбор лучшего фронтального вида      │
│   stl_drawing/orientation/view_scorer.py     │
│   select_best_front_and_reorient(...)        │
│   Скоринг 6 направлений: площадь × (1 +     │
│     0.3×силуэт + 0.5×нормали)               │
│   Переориентация: X=ширина, Y=высота,       │
│   Z=глубина, правая система координат        │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 4: Топология рёбер                      │
│   stl_drawing/topology/processor.py          │
│   TopologyProcessor(vertices, faces)         │
│   Классификация: острые (>10°), плавные,     │
│   граничные. Параллельная обработка          │
│   (threading)                                │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 4.5: Детекция цилиндров                 │
│   stl_symmetry_detector.py                   │
│   CylinderDetector().detect(mesh, offset)    │
│   → [{axis, center, radius, length}, ...]    │
│   Для осевых линий и автоматических размеров │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 5: Генерация 6 видов                    │
│   stl_drawing/projection/view_processor.py   │
│   ViewProcessor.process_view(view_type)      │
│                                              │
│   Для каждого из 6 ортогональных видов:      │
│   1. Проекция вершин (VIEW_MATRICES)         │
│   2. R-tree индекс на треугольниках          │
│   3. Детекция силуэтных рёбер                │
│   4. Определение видимости (adaptive         │
│      sampling + visibility cache)            │
│   5. Слияние коллинеарных (line_ops.py)      │
│   6. Приоритет стилей: видимые > скрытые     │
│   → (visible_lines, hidden_lines)            │
│                                              │
│   + Проекция осей цилиндров → centerlines    │
│     (centerline / crosshair)                 │
└──────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────┐
│ Шаг 6: Генерация ЕСКД-чертежа               │
│   stl_drawing/drawing/sheet.py               │
│   ESKDDrawingSheet.generate_drawing()        │
│                                              │
│   6a. Отбор видов (ГОСТ 2.305-2008)         │
│       view_selector.py: 5-шаговый алгоритм  │
│       → минимальный набор (обычно 3 вида)    │
│                                              │
│   6b. Выбор формата и масштаба               │
│       layout.py: scale-first, Tekla-style    │
│       Масштабы 1:1→1:1000, форматы A4→A0,   │
│       обе ориентации (landscape/portrait)    │
│                                              │
│   6c. Компоновка видов на листе              │
│       layout.py: arrange_six/three_views()   │
│       Метод первого угла (европейский)       │
│                                              │
│   6d. Параметры линий (ГОСТ 2.303-68)       │
│       gost_params.py: S, штрихи, пробелы    │
│                                              │
│   6e. Рендеринг линий в SVG                  │
│       svg_renderer.py: видимые, скрытые,     │
│       осевые (штрихпунктирная красная)       │
│                                              │
│   6f. Автоматическое оразмеривание           │
│       dimensions/extractor.py → candidates   │
│       dimensions/dedup.py → best view/dim    │
│       dimensions/placer.py → coordinates     │
│       dimensions/renderer.py → SVG elements  │
│                                              │
│   6g. Штамп и рамка (ГОСТ 2.104-2006)       │
│       title_block.py: Форма 1, 185×55 мм    │
│                                              │
│   → output.svg                               │
└──────────────────────────────────────────────┘
```

---

## Структура проекта

```
new_one/
├── main.py                          # Точка входа, CLI, пайплайн
├── stl_symmetry_detector.py         # Детекция цилиндров (внешний модуль)
├── stl_to_point_1.py                # Донорский скрипт (справочный)
│
├── stl_drawing/
│   ├── config.py                    # ВСЕ константы и ГОСТ-параметры
│   │
│   ├── io/
│   │   └── stl_loader.py            # load_stl() → (vertices, faces)
│   │
│   ├── orientation/
│   │   ├── pca.py                   # orient_model_by_normals()
│   │   └── view_scorer.py           # select_best_front_and_reorient()
│   │
│   ├── topology/
│   │   └── processor.py             # TopologyProcessor: sharp/smooth/boundary
│   │
│   ├── projection/
│   │   ├── view_processor.py        # ViewProcessor, VIEW_MATRICES (6 проекций)
│   │   └── line_ops.py              # merge_collinear, apply_style_priority
│   │
│   ├── geometry/
│   │   ├── visibility.py            # adaptive_sampling, VisibilityCache
│   │   ├── spatial_index.py         # build_rtree_index, query_rtree
│   │   └── triangle.py              # point_in_triangle, interpolate_z
│   │
│   └── drawing/
│       ├── sheet.py                 # ESKDDrawingSheet — оркестратор чертежа
│       ├── view_selector.py         # select_necessary_views() — ГОСТ 2.305
│       ├── layout.py                # select_format_and_scale(), arrange_views()
│       ├── gost_params.py           # calculate_line_parameters() — ГОСТ 2.303
│       ├── svg_renderer.py          # render_view_lines(), _render_centerlines()
│       ├── title_block.py           # add_title_block(), add_frame()
│       │
│       └── dimensions/              # Автоматическое оразмеривание
│           ├── __init__.py          # Реэкспорт публичных функций
│           ├── extractor.py         # extract_dimensions() — 3 источника
│           ├── dedup.py             # deduplicate_dimensions() — лучший вид
│           ├── placer.py            # place_dimensions() — AABB overlap
│           └── renderer.py          # render_dimensions() — SVG-элементы
```

---

## Ключевые модули: подробное описание

### main.py — Точка входа

**Функции:**
- `run_pipeline(stl_path, output_svg, metadata...)` — полный пайплайн STL→SVG
- `_compute_face_normals(vertices, faces)` — нормали граней
- `_detect_cylinders(vertices, faces)` — обёртка CylinderDetector
- `_compute_centerlines(cylinders, view_name)` — проекция осей цилиндров в 2D

**Особенности:**
- `sys.stdout.reconfigure(encoding='utf-8')` — поддержка emoji в логах CylinderDetector на Windows

### stl_drawing/config.py — Константы

Все магические числа собраны в одном файле:

| Группа | Примеры |
|--------|---------|
| Точность геометрии | `GRID_SIZE=1e-4`, `EPS_DEPTH=1e-6`, `EPS_SEGMENT=1e-4` |
| Классификация рёбер | `SHARP_ANGLE_DEGREES=10.0` |
| Отбор видов | `VIEW_SIMILARITY_THRESHOLD=0.85`, `VIEW_MIN_INFO_SCORE=2.0` |
| Форматы ГОСТ 2.301-68 | `GOST_FORMATS={'A4':(210,297), ...}` |
| Масштабы ГОСТ 2.302-68 | `GOST_REDUCTION_SCALES=[1.0, 0.5, 0.4, ...]` |
| Рамка ГОСТ 2.104-2006 | `MARGIN_LEFT=20`, `MARGIN_OTHER=5`, `TITLE_BLOCK_H=55` |
| Линии ГОСТ 2.303-68 | `S_THIN_DRAWING=0.5`, `S_THICK_DRAWING=0.7` |
| Размеры ГОСТ 2.307-2011 | `DIM_FIRST_OFFSET=10`, `DIM_NEXT_OFFSET=7`, `DIM_ARROW_LENGTH=2.5` |

### stl_drawing/drawing/view_selector.py — Отбор видов

5-шаговый алгоритм (ГОСТ 2.305-2008 п.5.1):
1. Исключить противоположные виды с меньшей информацией
2. Исключить «пустые» виды (только контурный прямоугольник)
3. Исключить зеркально-симметричные (Жаккар > 0.85)
4. Гарантировать покрытие осей X, Y, Z
5. Убрать информационно-избыточные

`info_score = n_visible + 0.3 × n_hidden`

### stl_drawing/drawing/layout.py — Формат и масштаб

**Стратегия (scale-first, Tekla-style):**
1. Перебор масштабов ГОСТ от 1:1 до 1:1000
2. Для каждого масштаба — вычислить размеры компоновки в мм
3. Перебор форматов от A4 до A0 в обеих ориентациях
4. Первая подходящая комбинация = максимальный масштаб + минимальный формат

**Компоновка видов (метод первого угла):**
```
         [bottom]            (вид снизу — СВЕРХУ)
[right]  [front]  [left]  [back]
         [top]              (вид сверху — СНИЗУ)
```

### stl_drawing/drawing/dimensions/ — Автоматическое оразмеривание

**Четыре модуля:**

1. **extractor.py** — извлечение кандидатов:
   - Габаритные (overall): из bbox каждого вида
   - Диаметры цилиндров: из CylinderDetector
   - Ступенчатые (step): кластеризация H/V рёбер
   - Каждый кандидат имеет `canonical_key` для дедупликации

2. **dedup.py** — выбор лучшего вида для каждого размера:
   - Группировка по `canonical_key`
   - Скоринг: front +2, true length +3, load balance +1
   - Результат: `{view_name → [candidates]}`

3. **placer.py** — размещение на листе:
   - Группировка по сторонам (top/bottom/left/right)
   - Сортировка: меньшие ближе к контуру
   - Ряды: 10мм первый, +7мм каждый следующий
   - **AABB collision detection**: проверка наложений между размерами и видами
   - Фильтр `DIM_MIN_DISPLAYABLE`: размеры < 4мм на бумаге не показываются

4. **renderer.py** — SVG-рендеринг:
   - Стрелки (path): треугольник 2.5×0.8мм
   - Размерная линия (line): тонкая сплошная
   - Выносные линии (line): тонкая сплошная
   - Текст (text): ISOCPEUR italic 3.5мм, baseline offset `y + size×0.35`
   - Фон текста (rect): белый прямоугольник для читаемости

### stl_drawing/drawing/svg_renderer.py — Рендеринг линий

- **Видимые линии**: сплошная основная (S), удлинение на S/2 для стыков
- **Скрытые линии**: ручная генерация штрихов (ЕСКД: начало/конец = полный штрих)
- **Осевые линии**: штрихпунктирная красная, вынос 3мм за контур
  - Crosshair (⊥ к оси): горизонталь + вертикаль, min 5мм полуплечо
  - Окружности Ø < 12мм: сплошная тонкая (без штрихпунктира)

### stl_drawing/projection/view_processor.py — Проекция видов

**VIEW_MATRICES** — 6 матриц 3×3 для ортогональных проекций:
- front: [X→right, Y→up, Z→depth]
- top: [X→right, Z→up, -Y→depth]
- right: [Z→right, Y→up, -X→depth]

**Алгоритм process_view():**
1. Проекция вершин через матрицу вида
2. R-tree индекс на 2D-треугольниках
3. Детекция силуэтных рёбер (смена знака dot(normal, view_dir))
4. Adaptive sampling каждого ребра → видимые/скрытые сегменты
5. Merge collinear + style priority (видимые «вырезают» скрытые)

---

## Применяемые ГОСТ-стандарты

| Стандарт | Название | Где используется |
|----------|----------|------------------|
| ГОСТ 2.104-2006 | Основная надпись | `title_block.py`: рамка, штамп 185×55мм, боковой штамп |
| ГОСТ 2.301-68 | Форматы листов | `config.py`, `layout.py`: A0–A4 |
| ГОСТ 2.302-68 | Масштабы | `config.py`, `layout.py`: 1:1 → 1:1000 |
| ГОСТ 2.303-68 | Линии чертежа | `gost_params.py`, `svg_renderer.py`: S, S/2, штрихи |
| ГОСТ 2.304-81 | Шрифты | `title_block.py`, `renderer.py`: ISOCPEUR тип Б italic |
| ГОСТ 2.305-2008 | Виды, проекции | `view_selector.py`, `layout.py`: метод 1-го угла |
| ГОСТ 2.307-2011 | Нанесение размеров | `dimensions/`: стрелки, выносные, текст, отступы |

---

## Ключевые структуры данных

### views_data (Dict[str, Dict])
```python
{
    'front': {
        'visible': [(pA, pB), ...],    # видимые отрезки (модельные коорд.)
        'hidden':  [(pA, pB), ...],     # скрытые отрезки
        'bbox': {'min_x', 'max_x', 'min_y', 'max_y', 'width', 'height'},
        'centerlines': [{'type': 'centerline'|'crosshair', ...}, ...],
    },
    'top': {...},
    ...
}
```

### layout (Dict[str, Dict])
```python
{
    'front': {
        'x': float, 'y': float,          # верхний-левый угол вида (мм)
        'width': float, 'height': float,  # размеры вида на листе (мм)
        'center_x': float, 'center_y': float,
        'offset_x': float, 'offset_y': float,  # смещение для пересчёта
    },
    ...
}
```

### DimensionCandidate (dataclass)
```python
DimensionCandidate:
    dim_type: str          # "linear_horizontal" | "linear_vertical" | "diameter"
    value_mm: float        # размер в мм модели
    canonical_key: str     # ключ дедупликации ("overall_X", "cylinder_diameter_0", ...)
    view_name: str
    anchor_a, anchor_b: Tuple[float, float]  # точки привязки (модельные)
    preferred_side: str    # "top" | "bottom" | "left" | "right"
    priority: int          # 0=габаритный, 1=диаметр, 2=ступенчатый
    center, radius: Optional  # для диаметров
```

### PlacedDimension (slots-class)
```python
PlacedDimension:
    dim_type, text_value, text_angle
    ext_line_a_start/end, ext_line_b_start/end  # выносные линии (мм на листе)
    dim_line_start/end                           # размерная линия
    arrow_a/b_pos, arrow_a/b_angle              # стрелки
    text_pos                                     # позиция текста
```

---

## Тестовые модели

| Файл | Описание | Особенности |
|------|----------|-------------|
| `fuel.stl` | Топливный бак | 230×140×410мм, 5 цилиндров, сложная геометрия |
| `Shveller_16P.stl` | Швеллер 16П | 751×160×64мм, 1 цилиндр, ступенчатый профиль |

---

## Известные ограничения

1. Определение видимости — O(n_edges × n_faces), медленно на больших моделях
2. Автоматические размеры не включают радиусы скруглений и фасок
3. Штрихпунктирная линия генерируется вручную (не через SVG stroke-dasharray для ЕСКД-совместимости)
4. Шрифт ISOCPEUR должен быть установлен в системе для корректного отображения
5. CylinderDetector (stl_symmetry_detector.py) использует emoji в логах — требует UTF-8 на Windows
