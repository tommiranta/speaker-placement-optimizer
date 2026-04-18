"""Speaker/listener placement optimizer."""
import numpy as np

from .config import RoomConfig
from .geometry import (
    bisector_filter, dist2d, equilateral_penalty, min_wall_distances,
    polygon_area_perimeter, room_y_range_at_x,
)
from .solver import (
    SPEED_OF_SOUND, build_domain, compute_all_responses, compute_eigenmodes,
    compute_decay_rate, compute_speaker_contribution, make_freqs, nearest_idx,
    precompute_modal_kernel, score_responses,
)

# Stereo triangle constraints
BISECTOR_TOLERANCE = 0.06   # max |dist_L - dist_R| for listener (m) — ~1 grid cell
DISTANCE_RATIO_MIN = 0.7
DISTANCE_RATIO_MAX = 1.5
SPEAKER_CENTER_TOL = 0.06  # max speaker midpoint offset from room centerline (m)


def generate_symmetric_speaker_pairs(spk_l, spk_r, step, coords, wall_dist,
                                     min_wall, vertices,
                                     max_move_x, max_move_y):
    """Generate candidate speaker pairs symmetric about a midpoint.

    Constraints:
    - Symmetric about midpoint (same orientation as current pair)
    - Each speaker within max_move per axis of original
    - Each speaker at least min_wall from walls
    - Pair midpoint centered between room side walls

    Returns list of (s1_idx, s2_idx, midpoint, distance) tuples.
    """
    mid = np.array([(spk_l[0] + spk_r[0]) / 2, (spk_l[1] + spk_r[1]) / 2])
    d_current = dist2d(spk_l, spk_r)
    direction = np.array([spk_r[0] - spk_l[0], spk_r[1] - spk_l[1]]) / d_current

    pairs = []
    seen = set()

    for dmx in np.arange(-max_move_x, max_move_x + step / 2, step):
        for dmy in np.arange(-max_move_y, max_move_y + step / 2, step):
            mx, my = mid[0] + dmx, mid[1] + dmy

            # Room centering check
            y_range = room_y_range_at_x(mx, vertices)
            if y_range is None:
                continue
            room_center_y = (y_range[0] + y_range[1]) / 2
            mid_along = mx * direction[0] + my * direction[1]
            center_along = mx * direction[0] + room_center_y * direction[1]
            if abs(mid_along - center_along) > SPEAKER_CENTER_TOL:
                continue

            distance_range = max_move_y
            for dd in np.arange(-distance_range, distance_range + step / 2, step):
                d = d_current + dd
                if d < 0.5:
                    continue
                half = d / 2
                s1x = mx - half * direction[0]
                s1y = my - half * direction[1]
                s2x = mx + half * direction[0]
                s2y = my + half * direction[1]

                if (abs(s1x - spk_l[0]) > max_move_x or
                    abs(s1y - spk_l[1]) > max_move_y or
                    abs(s2x - spk_r[0]) > max_move_x or
                    abs(s2y - spk_r[1]) > max_move_y):
                    continue

                idx1 = nearest_idx(s1x, s1y, coords)
                idx2 = nearest_idx(s2x, s2y, coords)

                snap1 = dist2d(coords[idx1], (s1x, s1y))
                snap2 = dist2d(coords[idx2], (s2x, s2y))
                if snap1 > step or snap2 > step:
                    continue
                if wall_dist[idx1] < min_wall or wall_dist[idx2] < min_wall:
                    continue

                # Enforce exact symmetry after grid snap
                p1, p2 = coords[idx1], coords[idx2]

                # Speakers must be at same depth (same x-coordinate)
                if abs(p1[0] - p2[0]) > 1e-6:
                    continue

                # Speaker pair must be centered between side walls
                actual_mid_y = (p1[1] + p2[1]) / 2
                yr = room_y_range_at_x(p1[0], vertices)
                if yr is not None:
                    room_cy = (yr[0] + yr[1]) / 2
                    if abs(actual_mid_y - room_cy) > SPEAKER_CENTER_TOL:
                        continue

                key = (idx1, idx2)
                if key not in seen:
                    seen.add(key)
                    actual_mid = (p1 + p2) / 2
                    actual_d = dist2d(p1, p2)
                    pairs.append((idx1, idx2, actual_mid, actual_d))

    return pairs


def evaluate_speaker_pairs(pairs, coords, evecs, inv_denom, z_w, n_freqs,
                           listener_wall_ok, fixed_li=None):
    """Evaluate all speaker pairs with optional fixed listener.

    Returns sorted list of (score, s1_idx, s2_idx, li_idx) tuples.
    """
    configs = []
    for pi, (s1, s2, mid, d) in enumerate(pairs):
        if (pi + 1) % 500 == 0:
            print(f"    Progress: {pi + 1}/{len(pairs)}")

        sl_p, sr_p = coords[s1], coords[s2]

        if fixed_li is not None:
            li_coord = coords[fixed_li:fixed_li + 1]
            if not bisector_filter(li_coord, sl_p, sr_p, BISECTOR_TOLERANCE)[0]:
                continue
            d_l = dist2d(li_coord[0], sl_p)
            d_r = dist2d(li_coord[0], sr_p)
            ratio = (d_l + d_r) / 2 / max(d, 0.01)
            if not (DISTANCE_RATIO_MIN <= ratio <= DISTANCE_RATIO_MAX):
                continue
            resp = (compute_speaker_contribution(s1, fixed_li, evecs, inv_denom, z_w) +
                    compute_speaker_contribution(s2, fixed_li, evecs, inv_denom, z_w))
            sc = score_responses(resp)[0]
            ep = equilateral_penalty(li_coord, sl_p, sr_p)[0]
            configs.append((sc + ep, s1, s2, fixed_li))
        else:
            on_bis = bisector_filter(coords, sl_p, sr_p, BISECTOR_TOLERANCE)
            d_l = np.sqrt((coords[:, 0] - sl_p[0]) ** 2 + (coords[:, 1] - sl_p[1]) ** 2)
            d_r = np.sqrt((coords[:, 0] - sr_p[0]) ** 2 + (coords[:, 1] - sr_p[1]) ** 2)
            ratio = (d_l + d_r) / 2 / max(d, 0.01)
            r_ok = (ratio >= DISTANCE_RATIO_MIN) & (ratio <= DISTANCE_RATIO_MAX)
            valid = on_bis & listener_wall_ok & r_ok
            if not valid.any():
                continue
            resp = compute_all_responses([s1, s2], evecs, inv_denom, z_w, n_freqs)
            scores = score_responses(resp)
            ep = equilateral_penalty(coords, sl_p, sr_p)
            total = scores + ep
            total[~valid] = np.inf
            best_li = int(np.argmin(total))
            if total[best_li] < np.inf:
                configs.append((total[best_li], s1, s2, best_li))

    configs.sort(key=lambda x: x[0])
    return configs


def run_optimization(cfg: RoomConfig, fix_listener: bool = False):
    """Run the full optimization pipeline.

    Returns dict with keys: configs, coords, evecs, inv_denom, z_w,
    score_orig, resp_orig, sp_l_idx, sp_r_idx, li_idx, cfg.
    """
    area, perimeter = polygon_area_perimeter(cfg.vertices)
    decay_rate, T60 = compute_decay_rate(area, perimeter, cfg.height, cfg.absorption)

    print(f"Room: {area:.1f} m² floor, {area * cfg.height:.1f} m³, "
          f"height {cfg.height} m")
    freqs = make_freqs(cfg.freq_max)
    print(f"Perimeter: {perimeter:.1f} m, T60: {T60:.2f} s")
    print(f"Frequency range: 20–{cfg.freq_max:.0f} Hz ({len(freqs)} points)")
    print(f"Coordinates: origin at bottom-left corner (0,0)")

    # Build grid
    print("\nBuilding computational grid...")
    coords, n_pts, mask, idx_map = build_domain(cfg.vertices)
    print(f"  {n_pts} interior points (dx={0.05} m)")

    wall_dist = min_wall_distances(coords, cfg.vertices)

    # Eigenmodes
    print("Computing eigenmodes...")
    evals, evecs = compute_eigenmodes(mask, idx_map, n_pts)

    print("\n2D room modes:")
    for i in range(min(10, len(evals))):
        if evals[i] > 0.01:
            f = SPEED_OF_SOUND * np.sqrt(evals[i]) / (2 * np.pi)
            print(f"  Mode {i}: {f:.1f} Hz")

    inv_denom, z_w, freqs = precompute_modal_kernel(
        evals, decay_rate, cfg.height, cfg.speaker_z, cfg.listener_z, freqs)
    n_freqs = len(freqs)

    # Current positions
    sp_l_idx = nearest_idx(*cfg.speaker_left, coords)
    sp_r_idx = nearest_idx(*cfg.speaker_right, coords)
    li_idx = nearest_idx(*cfg.listener, coords)

    # Original score
    resp_orig = (
        compute_speaker_contribution(sp_l_idx, li_idx, evecs, inv_denom, z_w) +
        compute_speaker_contribution(sp_r_idx, li_idx, evecs, inv_denom, z_w))
    score_orig = score_responses(resp_orig)[0]

    # Compute move ranges
    xs = [v[0] for v in cfg.vertices]
    ys = [v[1] for v in cfg.vertices]
    room_depth = max(xs) - min(xs)
    cur_mid_x = (coords[sp_l_idx][0] + coords[sp_r_idx][0]) / 2
    y_range = room_y_range_at_x(cur_mid_x, cfg.vertices)
    room_width = (y_range[1] - y_range[0]) if y_range else (max(ys) - min(ys))
    max_move_x = room_depth * cfg.move_fraction
    max_move_y = room_width * cfg.move_fraction

    print(f"\n  Move range: x ±{max_move_x:.2f} m ({cfg.move_fraction:.0%} of "
          f"{room_depth:.2f} m), y ±{max_move_y:.2f} m ({cfg.move_fraction:.0%} of "
          f"{room_width:.2f} m)")

    listener_wall_ok = wall_dist >= cfg.listener_min_wall
    fixed_li = li_idx if fix_listener else None

    # === Coarse pass ===
    coarse_step = max(cfg.search_step, 0.10)
    print(f"\nPass 1 — Coarse search (step {coarse_step:.2f} m)...")
    coarse_pairs = generate_symmetric_speaker_pairs(
        cfg.speaker_left, cfg.speaker_right, coarse_step,
        coords, wall_dist, cfg.speaker_min_wall, cfg.vertices,
        max_move_x, max_move_y)
    print(f"  {len(coarse_pairs)} speaker pairs")

    coarse_configs = evaluate_speaker_pairs(
        coarse_pairs, coords, evecs, inv_denom, z_w, n_freqs,
        listener_wall_ok, fixed_li)

    if not coarse_configs:
        return None

    # Top N distinct regions for refinement
    N_REFINE = 10
    coarse_winners = []
    seen_regions = set()
    for sc, s1, s2, li in coarse_configs:
        key = (round(coords[s1][0], 1), round(coords[s1][1], 1),
               round(coords[s2][0], 1), round(coords[s2][1], 1))
        if key not in seen_regions:
            seen_regions.add(key)
            coarse_winners.append((sc, s1, s2, li))
            if len(coarse_winners) >= N_REFINE:
                break

    print(f"  Top {len(coarse_winners)} regions (best: {coarse_configs[0][0]:.2f})")

    # === Fine pass ===
    fine_step = 0.05
    refine_radius = coarse_step * 1.5
    print(f"\nPass 2 — Fine refinement (step {fine_step:.2f} m, "
          f"±{refine_radius:.2f} m)...")

    best_configs = []
    for _, cs1, cs2, cli in coarse_winners:
        sl_c, sr_c = coords[cs1], coords[cs2]
        fine_pairs = generate_symmetric_speaker_pairs(
            (sl_c[0], sl_c[1]), (sr_c[0], sr_c[1]), fine_step,
            coords, wall_dist, cfg.speaker_min_wall, cfg.vertices,
            refine_radius, refine_radius)
        if fine_pairs:
            fine_configs = evaluate_speaker_pairs(
                fine_pairs, coords, evecs, inv_denom, z_w, n_freqs,
                listener_wall_ok, fixed_li)
            best_configs.extend(fine_configs)

    best_configs.extend(coarse_configs)
    best_configs.sort(key=lambda x: x[0])

    return {
        "configs": best_configs,
        "coords": coords,
        "evecs": evecs,
        "inv_denom": inv_denom,
        "z_w": z_w,
        "score_orig": score_orig,
        "resp_orig": resp_orig,
        "sp_l_idx": sp_l_idx,
        "sp_r_idx": sp_r_idx,
        "li_idx": li_idx,
    }
