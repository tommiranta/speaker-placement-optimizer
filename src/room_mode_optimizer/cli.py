"""Command-line interface for room mode optimizer."""
import subprocess
import sys
import time

import click
import numpy as np

from .config import DEFAULT_CONFIG, RoomConfig
from .geometry import describe_position, dist2d, room_y_range_at_x
from .optimizer import run_optimization
from .solver import (
    compute_speaker_contribution, nearest_idx, response_stats, score_responses,
)


@click.command()
@click.option("--url", type=str, required=True,
              help="vesalaasanen.com room mode calculator URL to use as starting point.")
@click.option("--fix-listener", is_flag=True, default=False,
              help="Lock listener at starting position, only optimize speakers.")
@click.option("--absorption", type=float, default=None,
              help="Override absorption coefficient (0.0–1.0).")
@click.option("--move-fraction", type=float, default=None,
              help="Speaker move range as fraction of room dimensions (default: 0.30).")
@click.option("--top", type=int, default=5,
              help="Number of top results to show (default: 5).")
@click.option("--freq-max", type=float, default=None,
              help="Upper frequency limit in Hz (default: 200).")
@click.option("--reorigin/--no-reorigin", default=True, show_default=True,
              help="Shift coordinates so the bottom-left corner is at (0,0).")
@click.option("--open-browser", is_flag=True, default=False,
              help="Open the best result URL in the default web browser.")
def main(url, fix_listener, absorption, move_fraction, top, freq_max,
         reorigin, open_browser):
    """Optimize speaker and listener placement to minimize room mode effects.

    Reads room configuration from a vesalaasanen.com URL (--url) or uses the
    built-in default. Outputs the top placements with verification URLs.
    """
    t0 = time.time()

    click.echo("Room Mode Optimizer")
    if fix_listener:
        click.echo("Mode: fixed listener — optimizing speakers only")
    click.echo("=" * 60)

    # Load config
    click.echo("Parsing URL...")
    cfg = RoomConfig.from_url(url, reorigin=reorigin)

    # Apply CLI overrides
    if absorption is not None:
        cfg.absorption = absorption
    if move_fraction is not None:
        cfg.move_fraction = move_fraction
    if freq_max is not None:
        cfg.freq_max = freq_max

    # Run optimization
    result = run_optimization(cfg, fix_listener=fix_listener)

    if result is None:
        click.echo("\nNo valid configurations found! Try relaxing constraints.")
        raise SystemExit(1)

    coords = result["coords"]
    evecs = result["evecs"]
    inv_denom = result["inv_denom"]
    z_w = result["z_w"]

    # Print original configuration
    cur_sl = coords[result["sp_l_idx"]]
    cur_sr = coords[result["sp_r_idx"]]
    cur_li = coords[result["li_idx"]]

    click.echo(f"\n{'=' * 60}")
    click.echo("RESULTS")
    click.echo("=" * 60)

    std_o, peak_o, null_o = response_stats(result["resp_orig"])
    click.echo(f"\nOriginal configuration:")
    click.echo(f"  Speaker L: ({cur_sl[0]:.2f}, {cur_sl[1]:.2f})")
    click.echo(f"  Speaker R: ({cur_sr[0]:.2f}, {cur_sr[1]:.2f})")
    click.echo(f"  Listener:  ({cur_li[0]:.2f}, {cur_li[1]:.2f})")
    _print_placement(cur_sl, cur_sr, cur_li, cfg.vertices)
    click.echo(f"  Triangle:  spkr↔spkr={dist2d(cur_sl, cur_sr):.2f} m, "
               f"listen↔L={dist2d(cur_li, cur_sl):.2f} m, "
               f"listen↔R={dist2d(cur_li, cur_sr):.2f} m")
    click.echo(f"  Response:  std={std_o:.1f} dB, peak=+{peak_o:.1f} dB, "
               f"null={null_o:.1f} dB")
    click.echo(f"  Acoustic score: {result['score_orig']:.2f}")

    # Print top results
    shown = set()
    rank = 0
    best_url = None
    for score, s1, s2, li in result["configs"]:
        key = (s1, s2, li)
        if key in shown:
            continue
        shown.add(key)
        rank += 1
        if rank > top:
            break

        sl_p, sr_p, lpos = coords[s1], coords[s2], coords[li]
        ss = dist2d(sl_p, sr_p)
        d_l, d_r = dist2d(lpos, sl_p), dist2d(lpos, sr_p)

        resp = (compute_speaker_contribution(s1, li, evecs, inv_denom, z_w) +
                compute_speaker_contribution(s2, li, evecs, inv_denom, z_w))
        std_v, peak_v, null_v = response_stats(resp)
        url_out = cfg.generate_url(sl_p, sr_p, lpos)

        ml = dist2d(sl_p, cfg.speaker_left)
        mr = dist2d(sr_p, cfg.speaker_right)

        spk_mid_x = (sl_p[0] + sr_p[0]) / 2
        spk_mid_y = (sl_p[1] + sr_p[1]) / 2
        yr = room_y_range_at_x(spk_mid_x, cfg.vertices)
        center_off = spk_mid_y - (yr[0] + yr[1]) / 2 if yr else 0

        click.echo(f"\n{'─' * 60}")
        click.echo(f"#{rank}  Acoustic score: {score:.2f}")
        click.echo(f"    Speaker L:  ({sl_p[0]:.2f}, {sl_p[1]:.2f})  [moved {ml:.2f} m]")
        click.echo(f"    Speaker R:  ({sr_p[0]:.2f}, {sr_p[1]:.2f})  [moved {mr:.2f} m]")
        click.echo(f"    Listener:   ({lpos[0]:.2f}, {lpos[1]:.2f})")
        _print_placement(sl_p, sr_p, lpos, cfg.vertices)
        click.echo(f"    Triangle:   spkr↔spkr={ss:.2f} m, "
                   f"listen↔L={d_l:.2f} m, listen↔R={d_r:.2f} m")
        click.echo(f"    Centering:  offset from room center: {center_off:+.2f} m")
        click.echo(f"    Response:   std={std_v:.1f} dB, peak=+{peak_v:.1f} dB, "
                   f"null={null_v:.1f} dB")
        click.echo(f"    URL: {url_out}")
        if best_url is None:
            best_url = url_out

    best_score = result["configs"][0][0]
    score_orig = result["score_orig"]
    if score_orig > 0:
        improvement = (score_orig - best_score) / score_orig * 100
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Best score: {best_score:.2f}  (was {score_orig:.2f}, "
                   f"improvement {improvement:.1f}%)")

    click.echo(f"Completed in {time.time() - t0:.1f} s")

    if open_browser and best_url:
        click.echo(f"\nOpening best result in browser...")
        _open_url(best_url)


def _open_url(url: str):
    """Open a URL in the default browser without shell interpretation.

    Uses subprocess directly to avoid shell mangling of # and | characters.
    """
    if sys.platform == "darwin":
        subprocess.Popen(["open", url])
    elif sys.platform == "win32":
        subprocess.Popen(["start", "", url], shell=True)
    else:
        subprocess.Popen(["xdg-open", url])


def _print_placement(sl_p, sr_p, lpos, vertices):
    """Print human-readable wall distances."""
    sl_d = describe_position(sl_p, vertices)
    sr_d = describe_position(sr_p, vertices)
    li_d = describe_position(lpos, vertices)
    click.echo(f"    Speaker L:  {sl_d.get('front wall (right)', 0):.2f} m from front wall, "
               f"{sl_d.get('side wall (bottom)', 0):.2f} m from closest side wall")
    click.echo(f"    Speaker R:  {sr_d.get('front wall (right)', 0):.2f} m from front wall, "
               f"{sr_d.get('side wall (top)', 0):.2f} m from closest side wall")
    click.echo(f"    Listener:   {li_d.get('front wall (right)', 0):.2f} m from front wall, "
               f"{li_d.get('rear wall (left)', 0):.2f} m from rear wall")


if __name__ == "__main__":
    main()
