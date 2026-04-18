#!/usr/bin/env python3
"""
Room Mode Optimizer
Finds optimal speaker and listener placement to minimize room mode effects.

Uses eigenmode decomposition of the 2D Helmholtz equation with Neumann
boundary conditions (rigid walls), extended to 3D via vertical mode
superposition. Optimizes placement by scoring frequency response flatness
across the 20-200 Hz range.

Geometry constraint: speakers and listener must form an approximately
equilateral triangle. Speakers are searched as symmetric pairs about a
midpoint, and the listener is constrained to the perpendicular bisector.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import time

from room_config import (
    ROOM_VERTICES, ROOM_HEIGHT, ABSORPTION,
    SPEAKER_LEFT, SPEAKER_RIGHT, SPEAKER_Z,
    LISTENER_START, LISTENER_Z,
    SPEAKER_MOVE_FRACTION, SPEAKER_SEARCH_STEP,
    SPEAKER_MIN_WALL_DIST, LISTENER_MIN_WALL_DIST,
)

# === Physics ===
C = 343.0  # speed of sound (m/s)

# === Frequency range ===
FREQ_MIN = 20
FREQ_MAX = 200
FREQ_STEP = 2
FREQS = np.arange(FREQ_MIN, FREQ_MAX + 0.1, FREQ_STEP)

# === Solver parameters ===
GRID_DX = 0.05      # grid spacing (meters)
N_2D_MODES = 150    # number of 2D eigenmodes to compute
N_Z_MODES = 5       # vertical modes: 0, 1, 2, 3, 4

# === Stereo triangle constraints ===
BISECTOR_TOLERANCE = 0.15      # max listener offset from perpendicular bisector (m)
DISTANCE_RATIO_MIN = 0.7      # min listener_dist / speaker_dist ratio
DISTANCE_RATIO_MAX = 1.5      # max listener_dist / speaker_dist ratio
SPEAKER_CENTER_TOL = 0.15     # max speaker midpoint offset from room centerline (m)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def points_in_polygon(points, vertices):
    """Vectorized ray-casting point-in-polygon test."""
    n = len(vertices)
    px, py = points[:, 0], points[:, 1]
    crossings = np.zeros(len(points), dtype=int)
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        cond = (yi > py) != (yj > py)
        x_int = (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
        crossings += (cond & (px < x_int)).astype(int)
        j = i
    return (crossings % 2) == 1


def polygon_area_perimeter(vertices):
    """Compute area and perimeter of a polygon."""
    n = len(vertices)
    area = 0.0
    perim = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
        perim += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return abs(area) / 2, perim


def min_wall_distances(points, vertices):
    """Compute minimum distance from each point to any wall segment."""
    n_verts = len(vertices)
    min_dist = np.full(len(points), np.inf)
    for i in range(n_verts):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % n_verts]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            continue
        t = ((points[:, 0] - ax) * dx + (points[:, 1] - ay) * dy) / seg_len2
        t = np.clip(t, 0, 1)
        cx = ax + t * dx
        cy = ay + t * dy
        dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def bisector_filter(coords, spk_l, spk_r, tolerance):
    """Return boolean mask: True for points within tolerance of perpendicular bisector.
    The perpendicular bisector is the set of points equidistant from both speakers.
    """
    dl = np.sqrt((coords[:, 0] - spk_l[0]) ** 2 + (coords[:, 1] - spk_l[1]) ** 2)
    dr = np.sqrt((coords[:, 0] - spk_r[0]) ** 2 + (coords[:, 1] - spk_r[1]) ** 2)
    return np.abs(dl - dr) <= tolerance


def room_y_range_at_x(x, vertices):
    """Compute the min/max y of the room interior at a given x coordinate.
    Uses ray-casting along the y-axis to find wall intersections.
    Returns (y_min, y_max) or None if x is outside the room.
    """
    intersections = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # Skip horizontal edges
        if abs(x2 - x1) < 1e-10:
            continue
        # Parameter t where the edge crosses x
        t = (x - x1) / (x2 - x1)
        if 0 <= t <= 1:
            y_int = y1 + t * (y2 - y1)
            intersections.append(y_int)
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def equilateral_penalty(coords, spk_l, spk_r):
    """Soft penalty for deviation from equilateral triangle.
    Returns per-point penalty (lower = better).
    Only applied to points already on the bisector.
    """
    ss = np.sqrt((spk_l[0] - spk_r[0]) ** 2 + (spk_l[1] - spk_r[1]) ** 2)
    if ss < 0.01:
        return np.zeros(len(coords))
    dl = np.sqrt((coords[:, 0] - spk_l[0]) ** 2 + (coords[:, 1] - spk_l[1]) ** 2)
    dr = np.sqrt((coords[:, 0] - spk_r[0]) ** 2 + (coords[:, 1] - spk_r[1]) ** 2)
    avg_d = (dl + dr) / 2
    # How far from equilateral (ratio=1.0 is perfect)
    ratio_dev = np.abs(avg_d / ss - 1.0)
    return ratio_dev * 0.5  # moderate weight


# ---------------------------------------------------------------------------
# Grid and Laplacian
# ---------------------------------------------------------------------------

def build_domain(vertices, dx):
    """Create computational grid and identify interior points."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x_grid = np.arange(min(xs) + dx / 2, max(xs), dx)
    y_grid = np.arange(min(ys) + dx / 2, max(ys), dx)
    nx, ny = len(x_grid), len(y_grid)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    all_pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = points_in_polygon(all_pts, vertices)
    mask = inside.reshape(nx, ny)
    idx_map = np.full((nx, ny), -1, dtype=int)
    coords = []
    k = 0
    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                idx_map[i, j] = k
                coords.append((x_grid[i], y_grid[j]))
                k += 1
    return x_grid, y_grid, mask, idx_map, np.array(coords), k


def build_neg_laplacian(mask, idx_map, n_pts, dx):
    """Negative Laplacian with Neumann (rigid wall) boundary conditions."""
    nx, ny = mask.shape
    h2 = dx * dx
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            k = idx_map[i, j]
            diag = 0.0
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    rows.append(k)
                    cols.append(idx_map[ni, nj])
                    vals.append(-1.0 / h2)
                    diag += 1.0 / h2
            rows.append(k)
            cols.append(k)
            vals.append(diag)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_pts, n_pts))


# ---------------------------------------------------------------------------
# Acoustics
# ---------------------------------------------------------------------------

def compute_decay_rate(area, perimeter, height, absorption):
    """Compute modal decay rate from Sabine's equation."""
    volume = area * height
    surface = 2 * area + perimeter * height
    total_abs = absorption * surface
    T60 = 0.161 * volume / max(total_abs, 0.01)
    return 6.91 / T60, T60


def precompute_modal_kernel(evals, decay_rate):
    """Precompute the frequency-dependent modal denominator."""
    omega = 2 * np.pi * FREQS
    k2 = (omega / C) ** 2
    nz = np.arange(N_Z_MODES)
    kz2 = (nz * np.pi / ROOM_HEIGHT) ** 2
    z_w = np.cos(nz * np.pi * SPEAKER_Z / ROOM_HEIGHT) * \
          np.cos(nz * np.pi * LISTENER_Z / ROOM_HEIGHT)
    z_w[1:] *= 2.0
    k_eff2 = k2[:, None] - kz2[None, :]
    eta = 2 * decay_rate * omega / C ** 2
    denom = evals[None, None, :] - k_eff2[:, :, None] + 1j * eta[:, None, None]
    return 1.0 / denom, z_w


def compute_all_responses(speaker_idxs, evecs, inv_denom, z_w):
    """Compute frequency response from speakers to ALL listener positions.
    Returns: complex array [n_points, n_freq].
    """
    response = np.zeros((evecs.shape[0], len(FREQS)), dtype=complex)
    for si in speaker_idxs:
        s_vec = evecs[si, :]
        G_modal = s_vec[None, None, :] * inv_denom
        G_weighted = np.einsum("fnm,n->fm", G_modal, z_w)
        response += evecs @ G_weighted.T
    return response


def compute_speaker_contribution(speaker_idx, listener_idx, evecs, inv_denom, z_w):
    """Compute one speaker's contribution at one listener. Returns [n_freq] complex."""
    coupling = evecs[speaker_idx, :] * evecs[listener_idx, :]
    G_modal = coupling[None, None, :] * inv_denom
    return np.einsum("fnm,n->f", G_modal, z_w)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_responses(responses):
    """Score frequency response flatness. Lower = flatter = better."""
    if responses.ndim == 1:
        responses = responses[None, :]
    mag = np.abs(responses)
    mag = np.maximum(mag, 1e-30)
    db = 20 * np.log10(mag)
    db -= db.mean(axis=1, keepdims=True)
    std = db.std(axis=1)
    null_depth = -db.min(axis=1)
    peak_height = db.max(axis=1)
    null_pen = np.maximum(null_depth - 12, 0) * 0.5
    peak_pen = np.maximum(peak_height - 12, 0) * 0.3
    return std + null_pen + peak_pen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_idx(x, y, coords):
    """Index of nearest grid point to (x, y)."""
    return int(np.argmin((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2))


def generate_symmetric_speaker_pairs(spk_l, spk_r, step, coords, wall_dist,
                                     min_wall, vertices,
                                     max_move_x, max_move_y):
    """Generate candidate speaker pairs that are symmetric about a midpoint.

    Constraints enforced:
    - Speakers symmetric about midpoint (same orientation as current pair)
    - Each speaker within max_move_x / max_move_y of its original position
    - Each speaker at least min_wall from walls
    - Speaker pair midpoint centered between room side walls (within SPEAKER_CENTER_TOL)

    Returns list of (s1_idx, s2_idx, midpoint, distance) tuples.
    """
    mid = np.array([(spk_l[0] + spk_r[0]) / 2, (spk_l[1] + spk_r[1]) / 2])
    d_current = np.sqrt((spk_r[0] - spk_l[0]) ** 2 + (spk_r[1] - spk_l[1]) ** 2)
    direction = np.array([spk_r[0] - spk_l[0], spk_r[1] - spk_l[1]]) / d_current

    # Search ranges derived from max move
    midpoint_range_x = max_move_x
    midpoint_range_y = max_move_y
    distance_range = max_move_y  # spread is along speaker direction

    pairs = []
    seen = set()

    for dmx in np.arange(-midpoint_range_x, midpoint_range_x + step / 2, step):
        for dmy in np.arange(-midpoint_range_y, midpoint_range_y + step / 2, step):
            mx = mid[0] + dmx
            my = mid[1] + dmy

            # --- Room centering check ---
            y_range = room_y_range_at_x(mx, vertices)
            if y_range is None:
                continue
            y_min, y_max = y_range
            room_center_y = (y_min + y_max) / 2
            mid_along_spread = mx * direction[0] + my * direction[1]
            center_along_spread = mx * direction[0] + room_center_y * direction[1]
            if abs(mid_along_spread - center_along_spread) > SPEAKER_CENTER_TOL:
                continue

            for dd in np.arange(-distance_range,
                                distance_range + step / 2, step):
                d = d_current + dd
                if d < 0.5:
                    continue
                half = d / 2
                s1x = mx - half * direction[0]
                s1y = my - half * direction[1]
                s2x = mx + half * direction[0]
                s2y = my + half * direction[1]

                # Each speaker within max_move per axis of original
                if (abs(s1x - spk_l[0]) > max_move_x or
                    abs(s1y - spk_l[1]) > max_move_y or
                    abs(s2x - spk_r[0]) > max_move_x or
                    abs(s2y - spk_r[1]) > max_move_y):
                    continue

                idx1 = nearest_idx(s1x, s1y, coords)
                idx2 = nearest_idx(s2x, s2y, coords)

                snap1 = np.sqrt((coords[idx1][0] - s1x) ** 2 +
                                (coords[idx1][1] - s1y) ** 2)
                snap2 = np.sqrt((coords[idx2][0] - s2x) ** 2 +
                                (coords[idx2][1] - s2y) ** 2)
                if snap1 > step or snap2 > step:
                    continue

                if wall_dist[idx1] < min_wall or wall_dist[idx2] < min_wall:
                    continue

                key = (idx1, idx2)
                if key not in seen:
                    seen.add(key)
                    actual_mid = (coords[idx1] + coords[idx2]) / 2
                    actual_d = np.sqrt((coords[idx2][0] - coords[idx1][0]) ** 2 +
                                       (coords[idx2][1] - coords[idx1][1]) ** 2)
                    pairs.append((idx1, idx2, actual_mid, actual_d))

    return pairs


def generate_url(spk_l_xy, spk_r_xy, listener_xy):
    """Generate vesalaasanen.com URL for verification.
    Uses room-corner coordinates directly (origin at 0,0).
    """
    poly = f"poly,{ROOM_HEIGHT:.2f}," + ",".join(
        f"{v[0]:.2f},{v[1]:.2f}" for v in ROOM_VERTICES)
    s1 = f"s,{spk_l_xy[0]:.2f},{spk_l_xy[1]:.2f},0.00,0.0,0.0,1,1"
    s2 = f"s,{spk_r_xy[0]:.2f},{spk_r_xy[1]:.2f},0.00,0.0,0.0,1,1"
    lst = f"l,{listener_xy[0]:.2f},{listener_xy[1]:.2f},{LISTENER_Z:.2f}"
    return (f"https://www.vesalaasanen.com/tools/room-mode-calculator"
            f"#{poly}|{s1}|{s2}|{lst}|t21|a{ABSORPTION:.2f}")


def response_stats(response):
    """Return (std, peak, null) in dB for a complex frequency response."""
    mag = np.abs(response)
    mag = np.maximum(mag, 1e-30)
    db = 20 * np.log10(mag)
    db -= np.mean(db)
    return db.std(), db.max(), db.min()


def describe_position(xy, vertices, label="point"):
    """Return human-readable description of a position relative to walls.

    For the L-shaped room with speakers on the right wall:
    - 'front wall' = right wall (wall the speakers face from, max x)
    - 'left/right side walls' = bottom (y=0) and top (y=max) walls
    - 'rear wall' = left wall (x=0, behind the listener)

    Computes distances to the nearest wall in each cardinal direction
    by finding where rays from the point hit the polygon edges.
    """
    x, y = xy[0], xy[1]

    # Cast rays in 4 directions and find distances to polygon edges
    dirs = {
        "front wall (right)": (1, 0),    # +x toward right wall
        "rear wall (left)": (-1, 0),     # -x toward left wall
        "side wall (bottom)": (0, -1),   # -y toward bottom wall
        "side wall (top)": (0, 1),       # +y toward top wall
    }

    distances = {}
    n = len(vertices)
    for name, (dx, dy) in dirs.items():
        best = np.inf
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            # Ray: P + t*(dx,dy), Segment: A + s*(B-A)
            ex, ey = x2 - x1, y2 - y1
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-10:
                continue
            t = ((x1 - x) * ey - (y1 - y) * ex) / denom
            s = ((x1 - x) * dy - (y1 - y) * dx) / denom
            if t > 1e-6 and 0 <= s <= 1:
                best = min(best, t)
        if best < np.inf:
            distances[name] = best

    return distances


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Room Mode Optimizer")
    print("=" * 60)

    # --- Room properties ---
    area, perimeter = polygon_area_perimeter(ROOM_VERTICES)
    volume = area * ROOM_HEIGHT
    decay_rate, T60 = compute_decay_rate(area, perimeter, ROOM_HEIGHT, ABSORPTION)

    print(f"Room: {area:.1f} m² floor, {volume:.1f} m³, height {ROOM_HEIGHT} m")
    print(f"Perimeter: {perimeter:.1f} m, T60: {T60:.2f} s")
    print(f"Frequency range: {FREQ_MIN}–{FREQ_MAX} Hz ({len(FREQS)} points)")
    print(f"Coordinates: origin at bottom-left corner (0,0)")

    # --- Build grid ---
    print("\nBuilding computational grid...")
    xg, yg, mask, idx_map, coords, n_pts = build_domain(ROOM_VERTICES, GRID_DX)
    print(f"  {n_pts} interior points ({len(xg)}×{len(yg)} grid, dx={GRID_DX} m)")

    # Precompute wall distances for all grid points
    wall_dist = min_wall_distances(coords, ROOM_VERTICES)

    # --- Compute eigenmodes ---
    n_modes = min(N_2D_MODES, n_pts - 2)
    print(f"Computing {n_modes} eigenmodes...")
    L = build_neg_laplacian(mask, idx_map, n_pts, GRID_DX)
    evals, evecs = eigsh(L, k=n_modes, sigma=1e-4, which="LM")
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # Show first room modes
    print("\n2D room modes (lowest frequencies):")
    for i in range(min(10, len(evals))):
        if evals[i] > 0.01:
            f = C * np.sqrt(evals[i]) / (2 * np.pi)
            print(f"  Mode {i}: {f:.1f} Hz")

    # Precompute modal kernel (reused for all evaluations)
    inv_denom, z_w = precompute_modal_kernel(evals, decay_rate)

    # --- Current configuration ---
    sp_l_idx = nearest_idx(*SPEAKER_LEFT, coords)
    sp_r_idx = nearest_idx(*SPEAKER_RIGHT, coords)
    li_idx = nearest_idx(*LISTENER_START, coords)

    cur_sl = coords[sp_l_idx]
    cur_sr = coords[sp_r_idx]
    cur_li = coords[li_idx]
    cur_ss = np.sqrt((cur_sl[0] - cur_sr[0]) ** 2 + (cur_sl[1] - cur_sr[1]) ** 2)
    cur_dl = np.sqrt((cur_li[0] - cur_sl[0]) ** 2 + (cur_li[1] - cur_sl[1]) ** 2)
    cur_dr = np.sqrt((cur_li[0] - cur_sr[0]) ** 2 + (cur_li[1] - cur_sr[1]) ** 2)

    print(f"\nCurrent setup:")
    print(f"  Speaker L: ({cur_sl[0]:.2f}, {cur_sl[1]:.2f})")
    print(f"  Speaker R: ({cur_sr[0]:.2f}, {cur_sr[1]:.2f})")
    print(f"  Listener:  ({cur_li[0]:.2f}, {cur_li[1]:.2f})")
    print(f"  Triangle:  spkr↔spkr={cur_ss:.2f} m, "
          f"listen↔L={cur_dl:.2f} m, listen↔R={cur_dr:.2f} m")

    resp_orig = (compute_speaker_contribution(sp_l_idx, li_idx, evecs, inv_denom, z_w) +
                 compute_speaker_contribution(sp_r_idx, li_idx, evecs, inv_denom, z_w))
    score_orig = score_responses(resp_orig)[0]
    std_o, peak_o, null_o = response_stats(resp_orig)
    print(f"  Acoustic score: {score_orig:.2f}")
    print(f"  Response: std={std_o:.1f} dB, peak=+{peak_o:.1f} dB, null={null_o:.1f} dB")

    # Valid listener mask (wall distance)
    listener_wall_ok = wall_dist >= LISTENER_MIN_WALL_DIST

    # Compute move ranges from room dimensions
    xs = [v[0] for v in ROOM_VERTICES]
    ys = [v[1] for v in ROOM_VERTICES]
    room_depth = max(xs) - min(xs)
    cur_mid_x = (cur_sl[0] + cur_sr[0]) / 2
    y_range = room_y_range_at_x(cur_mid_x, ROOM_VERTICES)
    room_width_at_spk = (y_range[1] - y_range[0]) if y_range else (max(ys) - min(ys))

    max_move_x = room_depth * SPEAKER_MOVE_FRACTION
    max_move_y = room_width_at_spk * SPEAKER_MOVE_FRACTION

    if y_range:
        room_cy = (y_range[0] + y_range[1]) / 2
        cur_mid_y = (cur_sl[1] + cur_sr[1]) / 2
        print(f"  Room center at speaker x={cur_mid_x:.2f}: "
              f"y={room_cy:.2f} (speakers midpoint y={cur_mid_y:.2f}, "
              f"offset={cur_mid_y - room_cy:.2f} m)")

    print(f"  Move range: x ±{max_move_x:.2f} m ({SPEAKER_MOVE_FRACTION:.0%} of "
          f"{room_depth:.2f} m), y ±{max_move_y:.2f} m ({SPEAKER_MOVE_FRACTION:.0%} of "
          f"{room_width_at_spk:.2f} m)")

    def evaluate_speaker_pairs(pairs, label=""):
        """Evaluate all speaker pairs, return sorted (score, s1, s2, li) list."""
        configs = []
        for pi, (s1, s2, mid, d) in enumerate(pairs):
            if (pi + 1) % 500 == 0:
                print(f"    Progress: {pi + 1}/{len(pairs)}")

            sl_p = coords[s1]
            sr_p = coords[s2]

            on_bis = bisector_filter(coords, sl_p, sr_p, BISECTOR_TOLERANCE)
            d_l = np.sqrt((coords[:, 0] - sl_p[0]) ** 2 +
                          (coords[:, 1] - sl_p[1]) ** 2)
            d_r = np.sqrt((coords[:, 0] - sr_p[0]) ** 2 +
                          (coords[:, 1] - sr_p[1]) ** 2)
            avg_dist = (d_l + d_r) / 2
            ratio = avg_dist / max(d, 0.01)
            r_ok = ((ratio >= DISTANCE_RATIO_MIN) &
                    (ratio <= DISTANCE_RATIO_MAX))
            valid = on_bis & listener_wall_ok & r_ok

            if not valid.any():
                continue

            resp = compute_all_responses([s1, s2], evecs, inv_denom, z_w)
            scores = score_responses(resp)
            ep = equilateral_penalty(coords, sl_p, sr_p)
            total = scores + ep
            total[~valid] = np.inf

            best_li_local = int(np.argmin(total))
            if total[best_li_local] < np.inf:
                configs.append((total[best_li_local], s1, s2, best_li_local))

        configs.sort(key=lambda x: x[0])
        return configs

    # =======================================================================
    # Coarse pass: 10cm speaker step over full range
    # =======================================================================
    coarse_step = max(SPEAKER_SEARCH_STEP, 0.10)
    print(f"\nPass 1 — Coarse search (step {coarse_step:.2f} m)...")

    coarse_pairs = generate_symmetric_speaker_pairs(
        SPEAKER_LEFT, SPEAKER_RIGHT, coarse_step,
        coords, wall_dist, SPEAKER_MIN_WALL_DIST, ROOM_VERTICES,
        max_move_x, max_move_y)
    print(f"  {len(coarse_pairs)} speaker pairs")

    coarse_configs = evaluate_speaker_pairs(coarse_pairs)

    if not coarse_configs:
        print("\nNo valid configurations found! Try relaxing constraints.")
        return

    # Collect unique coarse winners to refine (top N distinct speaker regions)
    N_REFINE = 10
    coarse_winners = []
    seen_regions = set()
    for sc, s1, s2, li in coarse_configs:
        # Quantize to coarse grid to identify distinct regions
        region_key = (round(coords[s1][0], 1), round(coords[s1][1], 1),
                      round(coords[s2][0], 1), round(coords[s2][1], 1))
        if region_key not in seen_regions:
            seen_regions.add(region_key)
            coarse_winners.append((sc, s1, s2, li))
            if len(coarse_winners) >= N_REFINE:
                break

    print(f"  Top {len(coarse_winners)} distinct regions found "
          f"(best coarse score: {coarse_configs[0][0]:.2f})")

    # =======================================================================
    # Fine pass: refine around each coarse winner
    # =======================================================================
    fine_step = GRID_DX  # use the 5cm grid resolution
    refine_radius = coarse_step * 1.5  # search ±15cm around coarse winner

    print(f"\nPass 2 — Fine refinement (step {fine_step:.2f} m, "
          f"±{refine_radius:.2f} m around each winner)...")

    best_configs = []

    for wi, (coarse_sc, cs1, cs2, cli) in enumerate(coarse_winners):
        sl_c = coords[cs1]
        sr_c = coords[cs2]
        # Use the coarse winner's actual speaker positions as center for fine search
        fine_center_l = (sl_c[0], sl_c[1])
        fine_center_r = (sr_c[0], sr_c[1])
        fine_mid = ((fine_center_l[0] + fine_center_r[0]) / 2,
                    (fine_center_l[1] + fine_center_r[1]) / 2)

        fine_pairs = generate_symmetric_speaker_pairs(
            fine_center_l, fine_center_r, fine_step,
            coords, wall_dist, SPEAKER_MIN_WALL_DIST, ROOM_VERTICES,
            refine_radius, refine_radius)

        if fine_pairs:
            fine_configs = evaluate_speaker_pairs(fine_pairs)
            best_configs.extend(fine_configs)

    # Also include the coarse configs in case fine search didn't improve
    best_configs.extend(coarse_configs)
    best_configs.sort(key=lambda x: x[0])

    # =======================================================================
    # Results: show top 5 unique configurations
    # =======================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    def print_placement(sl_p, sr_p, lpos):
        """Print human-readable placement distances from walls."""
        sl_d = describe_position(sl_p, ROOM_VERTICES)
        sr_d = describe_position(sr_p, ROOM_VERTICES)
        li_d = describe_position(lpos, ROOM_VERTICES)
        print(f"    Speaker L:  {sl_d.get('front wall (right)', 0):.2f} m from front wall, "
              f"{sl_d.get('side wall (bottom)', 0):.2f} m from closest side wall")
        print(f"    Speaker R:  {sr_d.get('front wall (right)', 0):.2f} m from front wall, "
              f"{sr_d.get('side wall (top)', 0):.2f} m from closest side wall")
        print(f"    Listener:   {li_d.get('front wall (right)', 0):.2f} m from front wall, "
              f"{li_d.get('rear wall (left)', 0):.2f} m from rear wall")

    print(f"\nOriginal configuration:")
    print(f"  Speaker L: ({cur_sl[0]:.2f}, {cur_sl[1]:.2f})")
    print(f"  Speaker R: ({cur_sr[0]:.2f}, {cur_sr[1]:.2f})")
    print(f"  Listener:  ({cur_li[0]:.2f}, {cur_li[1]:.2f})")
    print_placement(cur_sl, cur_sr, cur_li)
    print(f"  Triangle:  spkr↔spkr={cur_ss:.2f} m, "
          f"listen↔L={cur_dl:.2f} m, listen↔R={cur_dr:.2f} m")
    print(f"  Response:  std={std_o:.1f} dB, peak=+{peak_o:.1f} dB, null={null_o:.1f} dB")
    print(f"  Acoustic score: {score_orig:.2f}")

    if not best_configs:
        print("\nNo valid configurations found! Try relaxing constraints.")
        return

    # Deduplicate: skip configs where speakers AND listener are same grid point
    shown = set()
    rank = 0
    for score, s1, s2, li in best_configs:
        key = (s1, s2, li)
        if key in shown:
            continue
        shown.add(key)
        rank += 1
        if rank > 5:
            break

        sl_p = coords[s1]
        sr_p = coords[s2]
        lpos = coords[li]

        ss = np.sqrt((sl_p[0] - sr_p[0]) ** 2 + (sl_p[1] - sr_p[1]) ** 2)
        d_l = np.sqrt((lpos[0] - sl_p[0]) ** 2 + (lpos[1] - sl_p[1]) ** 2)
        d_r = np.sqrt((lpos[0] - sr_p[0]) ** 2 + (lpos[1] - sr_p[1]) ** 2)

        resp = (compute_speaker_contribution(s1, li, evecs, inv_denom, z_w) +
                compute_speaker_contribution(s2, li, evecs, inv_denom, z_w))
        std_v, peak_v, null_v = response_stats(resp)

        url = generate_url(sl_p, sr_p, lpos)

        # Movement from original
        ml = np.sqrt((sl_p[0] - SPEAKER_LEFT[0]) ** 2 + (sl_p[1] - SPEAKER_LEFT[1]) ** 2)
        mr = np.sqrt((sr_p[0] - SPEAKER_RIGHT[0]) ** 2 + (sr_p[1] - SPEAKER_RIGHT[1]) ** 2)

        # Room centering
        spk_mid_x = (sl_p[0] + sr_p[0]) / 2
        spk_mid_y = (sl_p[1] + sr_p[1]) / 2
        yr = room_y_range_at_x(spk_mid_x, ROOM_VERTICES)
        center_offset = spk_mid_y - (yr[0] + yr[1]) / 2 if yr else 0

        print(f"\n{'─' * 60}")
        print(f"#{rank}  Acoustic score: {score:.2f}")
        print(f"    Speaker L:  ({sl_p[0]:.2f}, {sl_p[1]:.2f})  [moved {ml:.2f} m]")
        print(f"    Speaker R:  ({sr_p[0]:.2f}, {sr_p[1]:.2f})  [moved {mr:.2f} m]")
        print(f"    Listener:   ({lpos[0]:.2f}, {lpos[1]:.2f})")
        print_placement(sl_p, sr_p, lpos)
        print(f"    Triangle:   spkr↔spkr={ss:.2f} m, "
              f"listen↔L={d_l:.2f} m, listen↔R={d_r:.2f} m")
        print(f"    Centering:  speaker midpoint offset from room center: "
              f"{center_offset:+.2f} m")
        print(f"    Response:   std={std_v:.1f} dB, peak=+{peak_v:.1f} dB, "
              f"null={null_v:.1f} dB")
        print(f"    URL: {url}")

    best_score = best_configs[0][0]
    if score_orig > 0:
        improvement = (score_orig - best_score) / score_orig * 100
        print(f"\n{'=' * 60}")
        print(f"Best score: {best_score:.2f}  (was {score_orig:.2f}, "
              f"improvement {improvement:.1f}%)")

    print(f"Completed in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
