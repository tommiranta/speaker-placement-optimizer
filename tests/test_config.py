"""Tests for speaker_placement_optimizer.config."""
import pytest
from speaker_placement_optimizer.config import RoomConfig


# A minimal vesalaasanen-style URL for testing
SAMPLE_URL = (
    "https://www.vesalaasanen.com/tools/room-mode-calculator"
    "#poly,2.60,1.00,1.00,6.00,1.00,6.00,4.00,1.00,4.00"
    "|s,5.50,1.50,0.00,0.0,0.0,1,1"
    "|s,5.50,3.50,0.00,0.0,0.0,1,1"
    "|l,3.00,2.50,1.11"
    "|t21"
    "|a0.25"
)


class TestFromUrl:
    def test_parses_height(self):
        cfg = RoomConfig.from_url(SAMPLE_URL)
        assert cfg.height == 2.60

    def test_parses_absorption(self):
        cfg = RoomConfig.from_url(SAMPLE_URL)
        assert cfg.absorption == 0.25

    def test_parses_vertices_count(self):
        cfg = RoomConfig.from_url(SAMPLE_URL)
        assert len(cfg.vertices) == 4

    def test_parses_listener(self):
        cfg = RoomConfig.from_url(SAMPLE_URL)
        # After normalization (shift by -1, -1), listener should be (2.0, 1.5)
        assert cfg.listener == pytest.approx((2.0, 1.5), abs=1e-6)

    def test_parses_speakers(self):
        cfg = RoomConfig.from_url(SAMPLE_URL)
        # After normalization: speaker_left = (4.5, 0.5), speaker_right = (4.5, 2.5)
        assert cfg.speaker_left == pytest.approx((4.5, 0.5), abs=1e-6)
        assert cfg.speaker_right == pytest.approx((4.5, 2.5), abs=1e-6)

    def test_no_fragment_raises(self):
        with pytest.raises(ValueError, match="no fragment"):
            RoomConfig.from_url("https://example.com/page")


    def test_no_reorigin_keeps_original_coords(self):
        cfg = RoomConfig.from_url(SAMPLE_URL, reorigin=False)
        # Vertices should still start at (1,1), not shifted to (0,0)
        assert cfg.vertices[0] == pytest.approx((1.0, 1.0), abs=1e-6)
        assert cfg.speaker_left == pytest.approx((5.5, 1.5), abs=1e-6)
        assert cfg.listener == pytest.approx((3.0, 2.5), abs=1e-6)

    def test_reorigin_true_shifts_to_zero(self):
        cfg = RoomConfig.from_url(SAMPLE_URL, reorigin=True)
        assert cfg.vertices[0] == pytest.approx((0.0, 0.0), abs=1e-6)
        assert cfg.speaker_left == pytest.approx((4.5, 0.5), abs=1e-6)


class TestNormalizeOrigin:
    def test_shifts_to_zero(self):
        cfg = RoomConfig()
        cfg.vertices = [(2.0, 3.0), (5.0, 3.0), (5.0, 6.0), (2.0, 6.0)]
        cfg.speaker_left = (4.0, 4.0)
        cfg.speaker_right = (4.0, 5.0)
        cfg.listener = (3.0, 4.5)
        cfg._normalize_origin()

        assert cfg.vertices[0] == pytest.approx((0.0, 0.0))
        assert cfg.speaker_left == pytest.approx((2.0, 1.0))
        assert cfg.listener == pytest.approx((1.0, 1.5))

    def test_already_at_origin_is_noop(self):
        cfg = RoomConfig()
        cfg.vertices = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
        cfg.speaker_left = (1.0, 1.0)
        original_speaker = cfg.speaker_left
        cfg._normalize_origin()
        assert cfg.speaker_left == original_speaker

    def test_empty_vertices_is_noop(self):
        cfg = RoomConfig()
        cfg._normalize_origin()  # should not raise


class TestGenerateUrl:
    def test_roundtrip_format(self):
        cfg = RoomConfig(
            vertices=[(0.0, 0.0), (5.0, 0.0), (5.0, 3.0), (0.0, 3.0)],
            height=2.50,
            absorption=0.30,
            listener_z=1.10,
        )
        url = cfg.generate_url(
            spk_l=(4.0, 0.5), spk_r=(4.0, 2.5), listener_xy=(2.0, 1.5)
        )
        assert url.startswith("https://www.vesalaasanen.com/tools/room-mode-calculator#")
        assert "poly,2.50," in url
        assert "a0.30" in url
        assert "l,2.00,1.50,1.10" in url

    def test_contains_both_speakers(self):
        cfg = RoomConfig(
            vertices=[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)],
            height=2.60,
        )
        url = cfg.generate_url(
            spk_l=(1.0, 1.0), spk_r=(1.0, 2.0), listener_xy=(2.0, 1.5)
        )
        # Two speaker sections separated by |
        assert url.count("|s,") == 2
