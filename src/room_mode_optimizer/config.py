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

    @classmethod
    def from_url(cls, url: str) -> "RoomConfig":
        """Parse a vesalaasanen.com room mode calculator URL.

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

        # Normalize so origin is at bottom-left corner
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
