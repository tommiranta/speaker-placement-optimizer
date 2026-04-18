"""Room and optimization configuration."""
import math
from dataclasses import dataclass, field
from urllib.parse import urlparse, unquote


@dataclass
class RoomConfig:
    """Complete configuration for a room mode optimization run."""

    # Room geometry
    vertices: list[tuple[float, float]] = field(default_factory=list)
    height: float = 2.60
    absorption: float = 0.30

    # Speaker positions (x, y) and height
    speaker_left: tuple[float, float] = (0.0, 0.0)
    speaker_right: tuple[float, float] = (0.0, 0.0)
    speaker_z: float = 1.00

    # Listener position and ear height
    listener: tuple[float, float] = (0.0, 0.0)
    listener_z: float = 1.11

    # Optimization constraints
    move_fraction: float = 0.30
    speaker_min_wall: float = 0.20
    listener_min_wall: float = 0.30

    # Search parameters
    search_step: float = 0.05

    # Speaker depth constraint
    max_speaker_depth: float | None = None  # max distance from front wall (m), None=no limit

    # Asymmetric placement constraints (disable centering when set)
    lock_speaker_l: float | None = None  # lock L speaker distance from its side wall (m)
    lock_speaker_r: float | None = None  # lock R speaker distance from its side wall (m)
    max_spread: float | None = None      # max speaker-to-speaker distance (m)

    # Frequency range
    freq_max: float = 200.0


    def detect_orientation(self) -> dict:
        """Detect the speaker/listener orientation from positions.

        Returns dict with:
        - spread_axis: unit vector along speaker spread (L→R)
        - depth_axis: unit vector from speakers toward listener
        - front_wall_dir: direction toward the front wall (behind speakers)
        """
        sl, sr = self.speaker_left, self.speaker_right
        li = self.listener

        # Spread axis: L → R
        sx, sy = sr[0] - sl[0], sr[1] - sl[1]
        spread_len = math.sqrt(sx * sx + sy * sy)
        if spread_len < 0.01:
            return {"spread_axis": (0, 1), "depth_axis": (-1, 0),
                    "front_wall_dir": (1, 0)}
        spread_axis = (sx / spread_len, sy / spread_len)

        # Depth axis: perpendicular to spread, toward listener
        # Two candidates: rotate spread ±90°
        perp1 = (-spread_axis[1], spread_axis[0])
        perp2 = (spread_axis[1], -spread_axis[0])

        mid = ((sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2)
        to_li = (li[0] - mid[0], li[1] - mid[1])

        # Pick the perpendicular that points toward the listener
        dot1 = perp1[0] * to_li[0] + perp1[1] * to_li[1]
        depth_axis = perp1 if dot1 > 0 else perp2

        # Front wall is behind the speakers (opposite of depth)
        front_wall_dir = (-depth_axis[0], -depth_axis[1])

        return {
            "spread_axis": spread_axis,
            "depth_axis": depth_axis,
            "front_wall_dir": front_wall_dir,
        }

    @classmethod
    def from_url(cls, url: str, reorigin: bool = True) -> "RoomConfig":
        """Parse a vesalaasanen.com room mode calculator URL.

        Args:
            url: Full URL with hash fragment.
            reorigin: If True, shift all coordinates so bottom-left corner
                      is at (0, 0). Set to False to keep original coordinates.

        Expected hash format:
        #poly,HEIGHT,x1,y1,x2,y2,...|s,x,y,z,...|s,x,y,z,...|l,x,y,z|t..|a0.30
        """
        fragment = unquote(urlparse(url).fragment)
        if not fragment:
            raise ValueError("URL has no fragment (hash) — expected #poly,...|s,...|...")

        parts = fragment.split("|")
        cfg = cls()

        for part in parts:
            tokens = part.split(",")
            kind = tokens[0]

            if kind == "poly":
                cfg.height = float(tokens[1])
                coords = [float(x) for x in tokens[2:]]
                cfg.vertices = [(coords[i], coords[i + 1])
                                for i in range(0, len(coords), 2)]
            elif kind == "s":
                x, y = float(tokens[1]), float(tokens[2])
                if cfg.speaker_left == (0.0, 0.0):
                    cfg.speaker_left = (x, y)
                else:
                    cfg.speaker_right = (x, y)
            elif kind == "l":
                cfg.listener = (float(tokens[1]), float(tokens[2]))
                if len(tokens) > 3:
                    cfg.listener_z = float(tokens[3])
            elif kind.startswith("a"):
                cfg.absorption = float(kind[1:])

        if reorigin:
            cfg._normalize_origin()
        cfg._fix_lr_labels()
        return cfg

    def _fix_lr_labels(self):
        """Assign L/R labels based on listener's perspective.

        From the listener looking toward the speakers, L is the speaker
        on the listener's left side. Swaps if needed.
        """
        sl, sr, li = self.speaker_left, self.speaker_right, self.listener
        mid = ((sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2)
        dx, dy = mid[0] - li[0], mid[1] - li[1]
        d = math.sqrt(dx * dx + dy * dy)
        if d < 0.01:
            return
        # Listener's left = 90° CCW rotation of looking direction
        left = (-dy / d, dx / d)
        # Project both speakers onto "left" direction
        proj_l = sl[0] * left[0] + sl[1] * left[1]
        proj_r = sr[0] * left[0] + sr[1] * left[1]
        if proj_l < proj_r:
            # Current "L" is actually on the right → swap
            self.speaker_left, self.speaker_right = self.speaker_right, self.speaker_left

    def _normalize_origin(self):
        """Shift all coordinates so the bottom-left corner is at (0, 0)."""
        if not self.vertices:
            return
        x_min = min(v[0] for v in self.vertices)
        y_min = min(v[1] for v in self.vertices)
        if x_min == 0 and y_min == 0:
            return

        self.vertices = [(v[0] - x_min, v[1] - y_min) for v in self.vertices]
        self.speaker_left = (self.speaker_left[0] - x_min,
                             self.speaker_left[1] - y_min)
        self.speaker_right = (self.speaker_right[0] - x_min,
                              self.speaker_right[1] - y_min)
        self.listener = (self.listener[0] - x_min,
                         self.listener[1] - y_min)

    def symmetrize(self) -> list[str]:
        """Correct speaker and listener positions for stereo symmetry.

        Works in any orientation by projecting onto the detected depth
        and spread axes. Ensures:
        - Both speakers at the same depth (along depth axis).
        - Speaker pair centered between side walls.
        - Listener on the perpendicular bisector of the speaker pair.

        Returns list of correction messages (empty if no changes needed).
        """
        corrections = []
        orient = self.detect_orientation()
        da = orient["depth_axis"]
        sa = orient["spread_axis"]

        sl = self.speaker_left
        sr = self.speaker_right

        # 1. Speakers at same depth (project onto depth axis, average)
        depth_l = sl[0] * da[0] + sl[1] * da[1]
        depth_r = sr[0] * da[0] + sr[1] * da[1]
        if abs(depth_l - depth_r) > 0.01:
            avg_depth = (depth_l + depth_r) / 2
            shift_l = avg_depth - depth_l
            shift_r = avg_depth - depth_r
            sl = (sl[0] + shift_l * da[0], sl[1] + shift_l * da[1])
            sr = (sr[0] + shift_r * da[0], sr[1] + shift_r * da[1])
            corrections.append(f"Aligned speaker depth along listening axis")

        # 2. Center speaker pair between side walls (skip if asymmetric mode)
        asymmetric = (self.lock_speaker_l is not None or
                      self.lock_speaker_r is not None or
                      self.max_spread is not None)
        if self.vertices and not asymmetric:
            from .geometry import room_y_range_at_x, room_x_range_at_y
            mid = ((sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2)
            spread_pos = mid[0] * sa[0] + mid[1] * sa[1]

            if abs(sa[0]) > abs(sa[1]):
                xr = room_x_range_at_y(mid[1], self.vertices)
                if xr:
                    room_center_spread = ((xr[0] + xr[1]) / 2) * sa[0]
                else:
                    room_center_spread = spread_pos
            else:
                yr = room_y_range_at_x(mid[0], self.vertices)
                if yr:
                    room_center_spread = ((yr[0] + yr[1]) / 2) * sa[1]
                else:
                    room_center_spread = spread_pos

            offset = spread_pos - room_center_spread
            if abs(offset) > 0.01:
                sl = (sl[0] - offset * sa[0], sl[1] - offset * sa[1])
                sr = (sr[0] - offset * sa[0], sr[1] - offset * sa[1])
                corrections.append(
                    f"Centered speakers between side walls "
                    f"(shifted {-offset * 100:+.0f} cm)")
        elif asymmetric:
            corrections.append("Asymmetric mode: skipping room centering")

        self.speaker_left = sl
        self.speaker_right = sr

        # 3. Center listener on perpendicular bisector
        spk_mid = ((sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2)
        # Listener's spread-axis component should match speaker midpoint
        li_spread = self.listener[0] * sa[0] + self.listener[1] * sa[1]
        mid_spread = spk_mid[0] * sa[0] + spk_mid[1] * sa[1]
        spread_offset = li_spread - mid_spread
        if abs(spread_offset) > 0.01:
            new_li = (self.listener[0] - spread_offset * sa[0],
                      self.listener[1] - spread_offset * sa[1])
            corrections.append(f"Centered listener between speakers")
            self.listener = new_li

        # 4. If listener is at origin, compute equilateral default
        if abs(self.listener[0]) < 0.01 and abs(self.listener[1]) < 0.01:
            spread = math.sqrt((sr[0] - sl[0]) ** 2 + (sr[1] - sl[1]) ** 2)
            eq_depth = spread * math.sqrt(3) / 2
            default = (spk_mid[0] + eq_depth * da[0],
                       spk_mid[1] + eq_depth * da[1])
            corrections.append(
                f"Computed listener position (equilateral, "
                f"{eq_depth:.2f} m from speakers)")
            self.listener = default

        return corrections

    def generate_url(self, spk_l, spk_r, listener_xy) -> str:
        """Generate a vesalaasanen.com URL for a given placement."""
        poly = f"poly,{self.height:.2f}," + ",".join(
            f"{v[0]:.2f},{v[1]:.2f}" for v in self.vertices)
        s1 = f"s,{spk_l[0]:.2f},{spk_l[1]:.2f},0.00,0.0,0.0,1,1"
        s2 = f"s,{spk_r[0]:.2f},{spk_r[1]:.2f},0.00,0.0,0.0,1,1"
        lst = f"l,{listener_xy[0]:.2f},{listener_xy[1]:.2f},{self.listener_z:.2f}"
        return (f"https://www.vesalaasanen.com/tools/room-mode-calculator"
                f"#{poly}|{s1}|{s2}|{lst}|t21|a{self.absorption:.2f}")


# Default configuration matching room_config.py
DEFAULT_CONFIG = RoomConfig(
    vertices=[
        (0.00, 0.00), (5.64, 0.00), (5.64, 3.59),
        (3.67, 3.59), (3.67, 5.04), (0.00, 5.03),
    ],
    height=2.60,
    absorption=0.30,
    speaker_left=(5.12, 0.57),
    speaker_right=(5.12, 2.87),
    speaker_z=1.00,
    listener=(2.62, 1.79),
    listener_z=1.11,
)
