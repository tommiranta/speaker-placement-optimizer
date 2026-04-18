"""Geometry utilities for room polygons and point operations."""
import numpy as np


def dist2d(a, b):
    """Euclidean distance between two 2D points."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def points_in_polygon(points: np.ndarray, vertices: list) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test.

    Args:
        points: [N, 2] array of (x, y) coordinates.
        vertices: list of (x, y) tuples defining the polygon.

    Returns:
        [N] boolean array.
    """
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


def polygon_area_perimeter(vertices: list) -> tuple[float, float]:
    """Compute area and perimeter of a polygon (shoelace formula)."""
    n = len(vertices)
    area = 0.0
    perim = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
        perim += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return abs(area) / 2, perim


def min_wall_distances(points: np.ndarray, vertices: list) -> np.ndarray:
    """Minimum distance from each point to any wall segment.

    Args:
        points: [N, 2] array.
        vertices: polygon vertices.

    Returns:
        [N] array of distances.
    """
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


def room_y_range_at_x(x: float, vertices: list) -> tuple[float, float] | None:
    """Compute the min/max y of the room interior at a given x coordinate.

    Returns (y_min, y_max) or None if x is outside the room.
    """
    intersections = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if abs(x2 - x1) < 1e-10:
            continue
        t = (x - x1) / (x2 - x1)
        if 0 <= t <= 1:
            intersections.append(y1 + t * (y2 - y1))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def room_x_range_at_y(y: float, vertices: list) -> tuple[float, float] | None:
    """Compute the min/max x of the room interior at a given y coordinate.

    Returns (x_min, x_max) or None if y is outside the room.
    """
    intersections = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if abs(y2 - y1) < 1e-10:
            continue
        t = (y - y1) / (y2 - y1)
        if 0 <= t <= 1:
            intersections.append(x1 + t * (x2 - x1))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def bisector_filter(coords: np.ndarray, spk_l, spk_r, tolerance: float) -> np.ndarray:
    """Boolean mask for points within tolerance of the perpendicular bisector."""
    dl = np.sqrt((coords[:, 0] - spk_l[0]) ** 2 + (coords[:, 1] - spk_l[1]) ** 2)
    dr = np.sqrt((coords[:, 0] - spk_r[0]) ** 2 + (coords[:, 1] - spk_r[1]) ** 2)
    return np.abs(dl - dr) <= tolerance


def equilateral_penalty(coords: np.ndarray, spk_l, spk_r) -> np.ndarray:
    """Soft penalty for deviation from equilateral triangle (per-point)."""
    ss = dist2d(spk_l, spk_r)
    if ss < 0.01:
        return np.zeros(len(coords))
    dl = np.sqrt((coords[:, 0] - spk_l[0]) ** 2 + (coords[:, 1] - spk_l[1]) ** 2)
    dr = np.sqrt((coords[:, 0] - spk_r[0]) ** 2 + (coords[:, 1] - spk_r[1]) ** 2)
    avg_d = (dl + dr) / 2
    return np.abs(avg_d / ss - 1.0) * 0.5


def describe_position(xy, vertices: list, orientation: dict | None = None) -> dict[str, float]:
    """Distances from a point to walls in named directions via ray casting.

    If orientation is provided, uses the detected speaker/listener axes:
    - "front wall" = direction behind speakers (front_wall_dir)
    - "rear wall" = opposite of front wall
    - "side wall L" / "side wall R" = along and against spread axis

    Falls back to cardinal directions if no orientation given.
    """
    x, y = float(xy[0]), float(xy[1])

    if orientation:
        fw = orientation["front_wall_dir"]
        da = orientation["depth_axis"]
        sa = orientation["spread_axis"]
        dirs = {
            "front wall": fw,
            "rear wall": (da[0], da[1]),       # toward listener
            "side wall L": (-sa[0], -sa[1]),    # toward left speaker side
            "side wall R": (sa[0], sa[1]),      # toward right speaker side
        }
    else:
        dirs = {
            "front wall": (1, 0),
            "rear wall": (-1, 0),
            "side wall L": (0, -1),
            "side wall R": (0, 1),
        }
    distances = {}
    n = len(vertices)
    for name, (dx, dy) in dirs.items():
        best = np.inf
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
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
