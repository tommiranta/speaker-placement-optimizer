# Room Mode Optimizer

Finds optimal speaker and listener positions in a room to minimize standing wave problems in the 20--200 Hz range.

The tool solves the 2D Helmholtz equation over your room's polygon using eigenmode decomposition with Neumann (rigid wall) boundary conditions, extends the modes vertically, and brute-force searches symmetric speaker/listener placements for the flattest combined frequency response.

## Installation

Requires Python 3.11+.

```bash
# macOS / Linux
uv tool install git+https://github.com/tranta/room-mode-optimizer.git

# Windows (PowerShell)
uv tool install git+https://github.com/tranta/room-mode-optimizer.git
```

Or install from a local clone:

```bash
git clone https://github.com/tranta/room-mode-optimizer.git
cd room-mode-optimizer
uv sync
```

## Usage

### Basic run (built-in default room)

```bash
room-optimize
```

Uses the L-shaped room hardcoded in `config.py`. Good for a quick test.

### From a vesalaasanen.com URL

```bash
room-optimize --url "https://www.vesalaasanen.com/tools/room-mode-calculator#poly,2.60,0.00,0.00,5.64,0.00,5.64,3.59,3.67,3.59,3.67,5.04,0.00,5.03|s,5.12,0.57,0.00,0.0,0.0,1,1|s,5.12,2.87,0.00,0.0,0.0,1,1|l,2.62,1.79,1.11|t21|a0.30"
```

Set up your room and speakers on [vesalaasanen.com/tools/room-mode-calculator](https://www.vesalaasanen.com/tools/room-mode-calculator), copy the URL, and pass it in. Results include URLs back to the calculator so you can visualize the suggested placements.

### Fix the listener, only move speakers

```bash
room-optimize --url "..." --fix-listener
```

Useful when your listening position is constrained (e.g., by a desk or couch).

### Other options

```bash
room-optimize --url "..." --absorption 0.20    # Wall absorption coefficient (default: 0.30)
room-optimize --url "..." --move-fraction 0.40  # Search radius as fraction of room size (default: 0.30)
room-optimize --url "..." --top 10              # Show more results (default: 5)
```

## How it works

1. **Mesh the room** -- discretize the polygon interior on a 5 cm grid.
2. **Solve eigenmodes** -- compute the 150 lowest eigenmodes of the 2D negative Laplacian (Neumann BCs) using `scipy.sparse.linalg.eigsh`.
3. **Extend to 3D** -- add vertical modes (5 cosine terms) weighted by speaker and listener heights.
4. **Modal decay** -- apply damping via Sabine's equation (absorption coefficient, room volume/surface).
5. **Frequency response** -- for each candidate placement, sum modal contributions across 20--200 Hz in 2 Hz steps to get the transfer function magnitude.
6. **Score** -- rank by response standard deviation in dB, with penalties for deep nulls (>12 dB) and tall peaks (>12 dB).
7. **Search** -- enumerate symmetric speaker pairs around the starting midpoint, combined with listener positions along the room axis. Enforce wall clearance and stereo triangle geometry constraints.

## Development

```bash
git clone https://github.com/tranta/room-mode-optimizer.git
cd room-mode-optimizer
uv sync

# Run tests
uv run pytest

# Run directly without installing
uv run room-optimize --help
```

### Project structure

```
src/room_mode_optimizer/
  cli.py        Command-line interface (Click)
  config.py     RoomConfig dataclass, URL parsing, defaults
  geometry.py   Polygon operations, wall distances, stereo triangle checks
  solver.py     Eigenmode solver, frequency response, scoring
  optimizer.py  Search loop: generate candidates, evaluate, rank
tests/
pyproject.toml
```
