"""
Room configuration for the mode optimizer.
Edit this file to match your room dimensions and setup.

All coordinates use origin at bottom-left corner of the room.
Y-axis points "up" (toward the far wall from the speakers).
X-axis points "right" when facing the speakers.
"""

# Room polygon vertices (x, y) in meters, counter-clockwise from origin.
# This L-shaped room has a wider bottom section and a narrower upper extension.
#
#  (0,5.03)───────────(3.67,5.04)
#     │                    │
#     │   narrower upper   │
#     │     section        │
#     │              (3.67,3.59)──────(5.64,3.59)
#     │                                   │
#     │        wider bottom section        │
#     │                                    │
#  (0,0)──────────────────────────────(5.64,0)
#
# For a simple rectangular room, use 4 vertices:
#   ROOM_VERTICES = [(0,0), (5.0,0), (5.0,4.0), (0,4.0)]

ROOM_VERTICES = [
    (0.00, 0.00),     # bottom-left corner (origin)
    (5.64, 0.00),     # bottom-right
    (5.64, 3.59),     # right wall top
    (3.67, 3.59),     # inner step corner
    (3.67, 5.04),     # upper section right
    (0.00, 5.03),     # upper-left
]

ROOM_HEIGHT = 2.60    # ceiling height in meters
ABSORPTION = 0.30     # average absorption coefficient (0=reflective, 1=absorptive)

# --- Speaker positions (x, y) in room coordinates ---
# Both speakers are near the right wall, facing left toward the listener.
SPEAKER_LEFT = (5.12, 0.57)    # "Left" channel
SPEAKER_RIGHT = (5.12, 2.87)   # "Right" channel
SPEAKER_Z = 1.00               # tweeter height above floor (meters)

# How far speakers may be moved during optimization
# Expressed as fraction of room depth (x) and width (y) at speaker position.
# 0.30 = speakers can move up to 30% of the room dimension along each axis.
SPEAKER_MOVE_FRACTION = 0.30   # fraction of room depth/width
SPEAKER_SEARCH_STEP = 0.05    # fine search step (meters); coarse pass always 10cm
SPEAKER_MIN_WALL_DIST = 0.20   # minimum distance from any wall (meters)
LISTENER_MIN_WALL_DIST = 0.30  # minimum listener distance from walls (meters)

# --- Listener position ---
LISTENER_START = (2.62, 1.79)  # starting / current position (x, y)
LISTENER_Z = 1.11              # ear height when seated (meters)
