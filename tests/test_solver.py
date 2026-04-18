"""Tests for speaker_placement_optimizer.solver."""
import numpy as np
import pytest
from speaker_placement_optimizer.solver import (
    build_domain,
    compute_eigenmodes,
    score_responses,
)

# Simple 4x3 rectangle
RECT = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]


class TestBuildDomain:
    def test_interior_point_count(self):
        coords, n_pts, mask, idx_map = build_domain(RECT, dx=0.5)
        # Grid starts at dx from wall and steps by dx.
        # x: 0.5, 1.0, ..., 3.5 -> 7 points
        # y: 0.5, 1.0, ..., 2.5 -> 5 points
        # All interior to the rectangle, so 7 * 5 = 35
        assert n_pts == 35
        assert coords.shape == (35, 2)

    def test_coords_inside_room(self):
        coords, n_pts, mask, idx_map = build_domain(RECT, dx=0.5)
        assert coords[:, 0].min() >= 0.0
        assert coords[:, 0].max() <= 4.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 3.0

    def test_idx_map_consistency(self):
        coords, n_pts, mask, idx_map = build_domain(RECT, dx=0.5)
        # Number of non-negative entries in idx_map should equal n_pts
        assert (idx_map >= 0).sum() == n_pts

    def test_l_shape_fewer_points(self):
        l_shape = [
            (0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
            (2.0, 2.0), (2.0, 3.0), (0.0, 3.0),
        ]
        coords_l, n_l, _, _ = build_domain(l_shape, dx=0.5)
        coords_r, n_r, _, _ = build_domain(RECT, dx=0.5)
        assert n_l < n_r


class TestComputeEigenmodes:
    @pytest.fixture
    def rect_domain(self):
        coords, n_pts, mask, idx_map = build_domain(RECT, dx=0.5)
        return coords, n_pts, mask, idx_map

    def test_first_eigenvalue_near_zero(self, rect_domain):
        coords, n_pts, mask, idx_map = rect_domain
        evals, evecs = compute_eigenmodes(mask, idx_map, n_pts, dx=0.5, n_modes=10)
        # First mode (constant/DC) should have eigenvalue ~0
        assert evals[0] == pytest.approx(0.0, abs=0.1)

    def test_eigenvalues_sorted(self, rect_domain):
        coords, n_pts, mask, idx_map = rect_domain
        evals, evecs = compute_eigenmodes(mask, idx_map, n_pts, dx=0.5, n_modes=10)
        assert np.all(np.diff(evals) >= -1e-10)

    def test_eigenvector_shapes(self, rect_domain):
        coords, n_pts, mask, idx_map = rect_domain
        n_modes = 10
        evals, evecs = compute_eigenmodes(mask, idx_map, n_pts, dx=0.5, n_modes=n_modes)
        assert evals.shape == (n_modes,)
        assert evecs.shape == (n_pts, n_modes)


class TestScoreResponses:
    def test_flat_scores_lower_than_peaked(self):
        n_freq = 91  # matches FREQS length (20 to 200 step 2)
        # Flat response: uniform magnitude
        flat = np.ones(n_freq, dtype=complex)
        # Peaked response: one big peak
        peaked = np.ones(n_freq, dtype=complex)
        peaked[20] = 100.0

        score_flat = score_responses(flat)[0]
        score_peaked = score_responses(peaked)[0]
        assert score_flat < score_peaked

    def test_batch_vs_single(self):
        n_freq = 91
        r1 = np.random.randn(n_freq) + 1j * np.random.randn(n_freq)
        r2 = np.random.randn(n_freq) + 1j * np.random.randn(n_freq)

        batch = score_responses(np.stack([r1, r2]))
        s1 = score_responses(r1)[0]
        s2 = score_responses(r2)[0]
        assert batch[0] == pytest.approx(s1)
        assert batch[1] == pytest.approx(s2)

    def test_constant_response_scores_zero(self):
        n_freq = 91
        flat = np.ones(n_freq, dtype=complex) * 5.0
        score = score_responses(flat)[0]
        assert score == pytest.approx(0.0, abs=1e-6)
