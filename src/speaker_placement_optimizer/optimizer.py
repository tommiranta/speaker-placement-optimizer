"""Speaker/listener placement optimizer."""
import numpy as np

from .config import RoomConfig
from .geometry import (
    bisector_filter, dist2d, equilateral_penalty, min_wall_distances,
    polygon_area_perimeter, room_x_range_at_y, room_y_range_at_x,
)
from .solver import (
    SPEED_OF_SOUND, build_domain, compute_all_responses, compute_eigenmodes,
    compute_decay_rate, compute_speaker_contribution, make_freqs, nearest_idx,
    precompute_modal_kernel, score_responses,
)

# Stereo triangle constraints
BISECTOR_TOLERANCE = 0.06   # max |dist_L - dist_R| for listener (m) — ~1 grid cell
SPEAKER_CENTER_TOL = 0.06  # max speaker midpoint offset from room centerline (m)


def _compute_locked_spread_pos(lock_dist, side, depth_coord, direction, vertices):
    """Compute the exact spread-axis position for a locked speaker.

    Args:
        lock_dist: fixed distance from side wall (meters).
        side: "L" or "R" — which speaker (determines which wall).
        depth_coord: the depth-axis coordinate (x or y depending on orientation).
        direction: spread direction unit vector (L→R).
        vertices: room polygon.

    Speaker L's side wall is in the -spread direction.
    Speaker R's side wall is in the +spread direction.
    The spread direction goes from L to R.

    Returns: (x, y) position or None if outside room.
    """
    if abs(direction[0]) > abs(direction[1]):
        rng = room_x_range_at_y(depth_coord, vertices)
        if rng is None:
            return None
        if (side == "L") == (direction[0] > 0):
            # L's wall is at rng[0] (min x) when spread goes +x
            # L's wall is at rng[1] (max x) when spread goes -x
            pos_spread = rng[0] + lock_dist
        else:
            pos_spread = rng[1] - lock_dist
        return (pos_spread, depth_coord)
    else:
        rng = room_y_range_at_x(depth_coord, vertices)
        if rng is None:
            return None
        if (side == "L") == (direction[1] > 0):
            pos_spread = rng[0] + lock_dist
        else:
            pos_spread = rng[1] - lock_dist
        return (depth_coord, pos_spread)


def generate_symmetric_speaker_pairs(spk_l, spk_r, step, coords, wall_dist,
                                     min_wall, vertices,
                                     max_move_x, max_move_y,
                                     lock_speaker_l=None, lock_speaker_r=None,
                                     max_spread=None,
                                     max_speaker_depth=None, orient=None):
    """Generate candidate speaker pairs.

    When speakers are locked, their spread position is computed from the
    room geometry — not searched. Only the depth axis is varied.

    Returns list of (s1_idx, s2_idx, midpoint, distance) tuples.
    """
    mid = np.array([(spk_l[0] + spk_r[0]) / 2, (spk_l[1] + spk_r[1]) / 2])
    d_current = dist2d(spk_l, spk_r)
    direction = np.array([spk_r[0] - spk_l[0], spk_r[1] - spk_l[1]]) / d_current
    asymmetric = (lock_speaker_l is not None or
                  lock_speaker_r is not None or
                  max_spread is not None)

    pairs = []
    seen = set()

    for dmx in np.arange(-max_move_x, max_move_x + step / 2, step):
        for dmy in np.arange(-max_move_y, max_move_y + step / 2, step):
            mx, my = mid[0] + dmx, mid[1] + dmy

            # Room centering check (skip in asymmetric mode)
            if not asymmetric:
                if abs(direction[0]) > abs(direction[1]):
                    rng = room_x_range_at_y(my, vertices)
                    if rng is None:
                        continue
                    room_center_spread = (rng[0] + rng[1]) / 2
                    mid_spread = mx
                else:
                    rng = room_y_range_at_x(mx, vertices)
                    if rng is None:
                        continue
                    room_center_spread = (rng[0] + rng[1]) / 2
                    mid_spread = my
                if abs(mid_spread - room_center_spread) > SPEAKER_CENTER_TOL:
                    continue

            # Determine spread positions
            if lock_speaker_l is not None and lock_speaker_r is not None:
                # Both locked: compute exact positions, no spread search
                depth_coord = my if abs(direction[0]) > abs(direction[1]) else mx
                pos_l = _compute_locked_spread_pos(
                    lock_speaker_l, "L", depth_coord, direction, vertices)
                pos_r = _compute_locked_spread_pos(
                    lock_speaker_r, "R", depth_coord, direction, vertices)
                if pos_l is None or pos_r is None:
                    continue
                spread_candidates = [(pos_l, pos_r)]
            elif lock_speaker_l is not None or lock_speaker_r is not None:
                # One locked: compute its position, search the other
                depth_coord = my if abs(direction[0]) > abs(direction[1]) else mx
                spread_candidates = []
                for dd in np.arange(-max_move_y, max_move_y + step / 2, step):
                    d = d_current + dd
                    if d < 0.5:
                        continue
                    if max_spread is not None and d > max_spread:
                        continue
                    half = d / 2
                    if lock_speaker_l is not None:
                        pos_l = _compute_locked_spread_pos(
                            lock_speaker_l, "L", depth_coord, direction, vertices)
                        if pos_l is None:
                            continue
                        # R is at lock_L_pos + d along spread
                        pos_r = (pos_l[0] + d * direction[0],
                                 pos_l[1] + d * direction[1])
                    else:
                        pos_r = _compute_locked_spread_pos(
                            lock_speaker_r, "R", depth_coord, direction, vertices)
                        if pos_r is None:
                            continue
                        pos_l = (pos_r[0] - d * direction[0],
                                 pos_r[1] - d * direction[1])
                    spread_candidates.append((pos_l, pos_r))
            else:
                # No locks: search midpoint + spread as before
                spread_candidates = []
                distance_range = max_move_y
                for dd in np.arange(-distance_range, distance_range + step / 2, step):
                    d = d_current + dd
                    if d < 0.5:
                        continue
                    if max_spread is not None and d > max_spread:
                        continue
                    half = d / 2
                    s1 = (mx - half * direction[0], my - half * direction[1])
                    s2 = (mx + half * direction[0], my + half * direction[1])
                    spread_candidates.append((s1, s2))

            for (s1_pos, s2_pos) in spread_candidates:
                s1x, s1y = s1_pos
                s2x, s2y = s2_pos

                # Max move check: depth always, spread only for unlocked speakers
                if abs(s1x - spk_l[0]) > max_move_x or abs(s2x - spk_r[0]) > max_move_x:
                    continue
                if lock_speaker_l is None and abs(s1y - spk_l[1]) > max_move_y:
                    continue
                if lock_speaker_r is None and abs(s2y - spk_r[1]) > max_move_y:
                    continue

                idx1 = nearest_idx(s1x, s1y, coords)
                idx2 = nearest_idx(s2x, s2y, coords)

                snap1 = dist2d(coords[idx1], (s1x, s1y))
                snap2 = dist2d(coords[idx2], (s2x, s2y))
                if snap1 > step or snap2 > step:
                    continue
                if wall_dist[idx1] < min_wall or wall_dist[idx2] < min_wall:
                    continue

                p1, p2 = coords[idx1], coords[idx2]

                # Absolute max depth from front wall
                if max_speaker_depth is not None and orient is not None:
                    from .geometry import describe_position as _desc
                    fw1 = _desc(p1, vertices, orient).get("front wall", 0)
                    fw2 = _desc(p2, vertices, orient).get("front wall", 0)
                    if fw1 > max_speaker_depth or fw2 > max_speaker_depth:
                        continue

                # Speakers must be at same depth along the depth axis
                depth1 = p1[0] * direction[1] - p1[1] * direction[0]
                depth2 = p2[0] * direction[1] - p2[1] * direction[0]
                if abs(depth1 - depth2) > 0.01:
                    continue

                # Post-snap centering check (skip in asymmetric mode)
                if not asymmetric:
                    if abs(direction[0]) > abs(direction[1]):
                        actual_mid_spread = (p1[0] + p2[0]) / 2
                        rng = room_x_range_at_y((p1[1] + p2[1]) / 2, vertices)
                    else:
                        actual_mid_spread = (p1[1] + p2[1]) / 2
                        rng = room_y_range_at_x((p1[0] + p2[0]) / 2, vertices)
                    if rng is not None:
                        room_c = (rng[0] + rng[1]) / 2
                        if abs(actual_mid_spread - room_c) > SPEAKER_CENTER_TOL:
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

        # Minimum listener depth: listener must be far enough from speakers
        # that the angle at the listening position is <= 90°.
        # This means: depth along listening axis >= spread / 2.
        spread = d
        min_depth = spread / 2
        spk_mid = (sl_p + sr_p) / 2

        if fixed_li is not None:
            li_coord = coords[fixed_li:fixed_li + 1]
            if not bisector_filter(li_coord, sl_p, sr_p, BISECTOR_TOLERANCE)[0]:
                continue
            # Check minimum depth
            li_depth = dist2d(li_coord[0], spk_mid)
            if li_depth < min_depth:
                continue
            resp = (compute_speaker_contribution(s1, fixed_li, evecs, inv_denom, z_w) +
                    compute_speaker_contribution(s2, fixed_li, evecs, inv_denom, z_w))
            sc = score_responses(resp)[0]
            ep = equilateral_penalty(li_coord, sl_p, sr_p)[0]
            configs.append((sc + ep, s1, s2, fixed_li))
        else:
            on_bis = bisector_filter(coords, sl_p, sr_p, BISECTOR_TOLERANCE)
            # Filter by minimum depth from speakers
            listener_depth = np.sqrt((coords[:, 0] - spk_mid[0]) ** 2 +
                                     (coords[:, 1] - spk_mid[1]) ** 2)
            depth_ok = listener_depth >= min_depth
            valid = on_bis & listener_wall_ok & depth_ok
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

    # Compute move ranges along depth and spread axes
    orient = cfg.detect_orientation()
    da = orient["depth_axis"]
    sa = orient["spread_axis"]

    xs = [v[0] for v in cfg.vertices]
    ys = [v[1] for v in cfg.vertices]
    room_depth = max(xs) - min(xs)
    cur_mid_x = (coords[sp_l_idx][0] + coords[sp_r_idx][0]) / 2
    y_range = room_y_range_at_x(cur_mid_x, cfg.vertices)
    room_width = (y_range[1] - y_range[0]) if y_range else (max(ys) - min(ys))
    max_move_x = room_depth * cfg.move_fraction
    max_move_y = room_width * cfg.move_fraction

    # Limit speaker depth from front wall if configured
    if cfg.max_speaker_depth is not None:
        # Find the front wall distance for current speaker position
        from .geometry import describe_position
        spk_wall = describe_position(coords[sp_l_idx], cfg.vertices, orient)
        current_depth = spk_wall.get("front wall", 0)
        # Clamp move range so speakers can't exceed max depth
        max_move_into_room = max(cfg.max_speaker_depth - current_depth, 0)
        max_move_x = min(max_move_x, current_depth + max_move_into_room)
        print(f"\n  Max speaker depth: {cfg.max_speaker_depth * 100:.0f} cm from front wall")

    print(f"  Move range: depth ±{max_move_x:.2f} m ({cfg.move_fraction:.0%} of "
          f"{room_depth:.2f} m), spread ±{max_move_y:.2f} m ({cfg.move_fraction:.0%} of "
          f"{room_width:.2f} m)")

    listener_wall_ok = wall_dist >= cfg.listener_min_wall
    fixed_li = li_idx if fix_listener else None

    # === Coarse pass ===
    coarse_step = max(cfg.search_step, 0.10)
    print(f"\nPass 1 — Coarse search (step {coarse_step:.2f} m)...")
    coarse_pairs = generate_symmetric_speaker_pairs(
        cfg.speaker_left, cfg.speaker_right, coarse_step,
        coords, wall_dist, cfg.speaker_min_wall, cfg.vertices,
        max_move_x, max_move_y,
        lock_speaker_l=cfg.lock_speaker_l, lock_speaker_r=cfg.lock_speaker_r,
        max_spread=cfg.max_spread,
        max_speaker_depth=cfg.max_speaker_depth, orient=orient)
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
            refine_radius, refine_radius,
            lock_speaker_l=cfg.lock_speaker_l, lock_speaker_r=cfg.lock_speaker_r,
            max_spread=cfg.max_spread,
            max_speaker_depth=cfg.max_speaker_depth, orient=orient)
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
