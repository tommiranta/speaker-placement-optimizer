"""Room and optimization configuration."""
from dataclasses import dataclass, field
from urllib.parse import urlparse, unquote
import re


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

    # Frequency range
    freq_max: float = 200.0

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
        return cfg

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

        Ensures:
        - Both speakers at the same depth (averaged x).
        - Speaker pair centered between side walls at their depth.
        - Listener y centered between speakers (on perpendicular bisector).
        - If listener position looks unset, compute a default (equilateral).

        Returns list of correction messages (empty if no changes needed).
        """
        import math
        corrections = []
        sl = self.speaker_left
        sr = self.speaker_right

        # 1. Speakers at same depth (same x)
        if abs(sl[0] - sr[0]) > 0.01:
            avg_x = (sl[0] + sr[0]) / 2
            corrections.append(
                f"Aligned speaker depth: both moved to x={avg_x:.2f} "
                f"(was L={sl[0]:.2f}, R={sr[0]:.2f})")
            sl = (avg_x, sl[1])
            sr = (avg_x, sr[1])

        # 2. Center speaker pair between side walls
        if self.vertices:
            from .geometry import room_y_range_at_x
            yr = room_y_range_at_x(sl[0], self.vertices)
            if yr is not None:
                room_center_y = (yr[0] + yr[1]) / 2
                pair_mid_y = (sl[1] + sr[1]) / 2
                offset = pair_mid_y - room_center_y
                if abs(offset) > 0.01:
                    sl = (sl[0], sl[1] - offset)
                    sr = (sr[0], sr[1] - offset)
                    corrections.append(
                        f"Centered speakers between side walls: "
                        f"shifted {-offset:+.2f} m to room center y={room_center_y:.2f}")

        self.speaker_left = sl
        self.speaker_right = sr

        # 3. Center listener y between speakers
        spk_mid_y = (sl[1] + sr[1]) / 2
        if abs(self.listener[1] - spk_mid_y) > 0.01:
            corrections.append(
                f"Centered listener: y {self.listener[1]:.2f} → {spk_mid_y:.2f} "
                f"(midpoint of speakers)")
            self.listener = (self.listener[0], spk_mid_y)

        # 4. If listener x is unset (0,0), compute equilateral default
        if self.listener == (0.0, spk_mid_y) or self.listener[0] == 0.0:
            spread = abs(sr[1] - sl[1])
            eq_depth = spread * math.sqrt(3) / 2
            default_x = sl[0] - eq_depth
            corrections.append(
                f"Computed listener depth: x={default_x:.2f} "
                f"(equilateral triangle, {eq_depth:.2f} m from speakers)")
            self.listener = (default_x, spk_mid_y)

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
