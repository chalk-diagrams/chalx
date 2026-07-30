# Chalk Development Guide

## Overview
Chalk is a Python library for declarative drawing and diagrams. It provides a functional API for creating 2D diagrams with multiple rendering backends (Cairo/PNG, SVG, Matplotlib).

## Quick Commands

| Task | Command |
|------|---------|
| Lint | `ruff check chalk/` |
| Format check | `ruff format --check chalk/` |
| Type check | `pyright` |
| Run tests | `pytest tests/` |
| Run hijax Path mock | `pytest tests/test_hijax_path.py` |
| Run example | `python3 examples/intro.py` |
| Pre-commit | `pre-commit run --all-files` or `make style` |

## Code Structure

- `chalk/` - Main library source code
  - `core.py` - Core diagram types and operations
  - `shapes.py` - Shape primitives (circle, square, rectangle, etc.)
  - `combinators.py` - Composition functions (hcat, vcat, beside, above)
  - `trail.py` - Path/trail construction
  - `path.py` - Pytree-batched Path (prefix batch dims on arrays)
  - `hijax_path.py` - Experimental opaque hijax Path mock
  - `transform.py` - Geometric transformations (V2, P2, Affine)
  - `style.py` - Styling (colors, line widths)
  - `backend/` - Rendering backends (cairo.py, svg.py, matplotlib.py)
- `examples/` - Example scripts demonstrating library usage
- `tests/` - Unit tests (note: some tests have outdated imports)
- `walkthrough/` - Tutorial/documentation notebooks

## Usage Pattern

```python
from colour import Color
from chalk import circle, square, hcat, vcat

# Create shapes
c = circle(1).fill_color(Color('#ff9700'))
s = square(2).fill_color(Color('#005FDB'))

# Compose
diagram = hcat([c, s])  # or c | s

# Render
diagram.render('output.png', height=256)
diagram.render_svg('output.svg', height=256)
```

## Hijax Path mock

`chalk/hijax_path.py` is an experimental opaque Path built on
[`jax.experimental.hijax`](https://docs.jax.dev/en/latest/hijax_types.html).
It is not wired into the main `chalk.path.Path` yet.

```bash
pytest tests/test_hijax_path.py
python examples/hijax_path.py
```

## Known Issues

- Tests in `tests/` have outdated imports (reference non-exported symbols)
- pyright config references `.venv12` which may not exist locally
- Some files need ruff formatting (use `ruff format chalk/` to fix)
