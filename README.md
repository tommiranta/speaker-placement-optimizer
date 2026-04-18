# Speaker Placement Optimizer

Finds optimal speaker and listener positions in a room to minimize standing wave problems in the 20–200 Hz range.

The tool solves the 2D Helmholtz equation over your room's polygon using eigenmode decomposition with Neumann (rigid wall) boundary conditions, extends the modes vertically, and brute-force searches symmetric speaker/listener placements for the flattest combined frequency response.

## Installation

Requires Python 3.11+.

```bash
# macOS / Linux / Windows
uv tool install git+https://github.com/tommiranta/speaker-placement-optimizer.git
```

Or install from a local clone:

```bash
git clone https://github.com/tommiranta/speaker-placement-optimizer.git
cd speaker-placement-optimizer
uv sync
```

## Usage

### From a vesalaasanen.com URL

Set up your room and speakers on [vesalaasanen.com/tools/room-mode-calculator](https://www.vesalaasanen.com/tools/room-mode-calculator), copy the URL, and pass it in:

```bash
spo --url "https://www.vesalaasanen.com/tools/room-mode-calculator#poly,2.60,..."
```

Results include clickable links back to the calculator so you can visualize the suggested placements.

### Fix the listener, only move speakers

```bash
spo --url "..." --fix-listener
```

Useful when your listening position is constrained (e.g., by a sofa or desk).

### Other options

```bash
spo --url "..." --max-speaker-depth 80   # Max distance from front wall in cm
spo --url "..." --absorption 0.20        # Wall absorption coefficient (default: 0.30)
spo --url "..." --move-fraction 0.40     # Search radius as fraction of room size (default: 0.30)
spo --url "..." --freq-max 150           # Upper frequency limit in Hz (default: 200)
spo --url "..." --top 10                 # Show more results (default: 5)
spo --url "..." --open-browser           # Open best result in browser
spo --url "..." --no-reorigin            # Keep original coordinates from URL
```

## How it works

1. **Mesh the room** — discretize the polygon interior on a 5 cm grid.
2. **Solve eigenmodes** — compute the 150 lowest eigenmodes of the 2D negative Laplacian (Neumann BCs) using `scipy.sparse.linalg.eigsh`.
3. **Extend to 3D** — add vertical modes (5 cosine terms) weighted by speaker and listener heights.
4. **Modal decay** — apply damping via Sabine's equation (absorption coefficient, room volume/surface).
5. **Frequency response** — for each candidate placement, sum modal contributions across the frequency range in 2 Hz steps.
6. **Score** — rank by response standard deviation in dB, with penalties for deep nulls (>12 dB) and tall peaks (>12 dB).
7. **Search** — coarse-to-fine enumeration of symmetric speaker pairs, with listener positions freely optimized along the perpendicular bisector. Enforces wall clearance, room centering, and stereo symmetry.

## Development

```bash
git clone https://github.com/tommiranta/speaker-placement-optimizer.git
cd speaker-placement-optimizer
uv sync --all-groups

# Run tests
uv run pytest

# Run directly without installing
uv run spo --help
```

### Project structure

```
src/speaker_placement_optimizer/
  cli.py        Command-line interface (Click)
  config.py     RoomConfig dataclass, URL parsing, defaults
  geometry.py   Polygon operations, wall distances, stereo triangle checks
  solver.py     Eigenmode solver, frequency response, scoring
  optimizer.py  Search loop: generate candidates, evaluate, rank
tests/
pyproject.toml
```
