"""Eigenmode solver and acoustic response computation."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .geometry import points_in_polygon

# Physics
SPEED_OF_SOUND = 343.0  # m/s

# Solver defaults
GRID_DX = 0.05
N_2D_MODES = 150
N_Z_MODES = 5
FREQ_MIN = 20
FREQ_MAX = 200
FREQ_STEP = 2


def make_freqs(freq_max=FREQ_MAX):
    """Generate frequency array from FREQ_MIN to freq_max in FREQ_STEP increments."""
    return np.arange(FREQ_MIN, freq_max + 0.1, FREQ_STEP)


def build_domain(vertices, dx=GRID_DX):
    """Create computational grid and identify interior points.

    Returns:
        coords: [n_pts, 2] array of interior point coordinates.
        n_pts: number of interior points.
        mask: [nx, ny] boolean grid.
        idx_map: [nx, ny] integer grid mapping to point indices (-1 = outside).
    """
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x_grid = np.arange(min(xs) + dx / 2, max(xs), dx)
    y_grid = np.arange(min(ys) + dx / 2, max(ys), dx)
    nx, ny = len(x_grid), len(y_grid)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    all_pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = points_in_polygon(all_pts, vertices)
    mask = inside.reshape(nx, ny)
    idx_map = np.full((nx, ny), -1, dtype=int)
    coords = []
    k = 0
    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                idx_map[i, j] = k
                coords.append((x_grid[i], y_grid[j]))
                k += 1
    return np.array(coords), k, mask, idx_map


def build_neg_laplacian(mask, idx_map, n_pts, dx=GRID_DX):
    """Negative Laplacian with Neumann (rigid wall) boundary conditions."""
    nx, ny = mask.shape
    h2 = dx * dx
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                continue
            k = idx_map[i, j]
            diag = 0.0
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    rows.append(k)
                    cols.append(idx_map[ni, nj])
                    vals.append(-1.0 / h2)
                    diag += 1.0 / h2
            rows.append(k)
            cols.append(k)
            vals.append(diag)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_pts, n_pts))


def compute_eigenmodes(mask, idx_map, n_pts, dx=GRID_DX, n_modes=N_2D_MODES):
    """Compute the lowest eigenmodes of the negative Laplacian.

    Returns:
        evals: [n_modes] eigenvalues (sorted ascending).
        evecs: [n_pts, n_modes] eigenvectors.
    """
    n_modes = min(n_modes, n_pts - 2)
    L = build_neg_laplacian(mask, idx_map, n_pts, dx)
    evals, evecs = eigsh(L, k=n_modes, sigma=1e-4, which="LM")
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def compute_decay_rate(area, perimeter, height, absorption):
    """Modal decay rate from Sabine's equation. Returns (decay_rate, T60)."""
    volume = area * height
    surface = 2 * area + perimeter * height
    total_abs = absorption * surface
    T60 = 0.161 * volume / max(total_abs, 0.01)
    return 6.91 / T60, T60


def precompute_modal_kernel(evals, decay_rate, height, speaker_z, listener_z,
                            freqs=None):
    """Precompute frequency-dependent modal denominator.

    Returns:
        inv_denom: [n_freq, n_z, n_modes] complex array.
        z_weights: [n_z] array.
        freqs: the frequency array used.
    """
    if freqs is None:
        freqs = make_freqs()
    c = SPEED_OF_SOUND
    omega = 2 * np.pi * freqs
    k2 = (omega / c) ** 2
    nz = np.arange(N_Z_MODES)
    kz2 = (nz * np.pi / height) ** 2
    z_w = (np.cos(nz * np.pi * speaker_z / height) *
           np.cos(nz * np.pi * listener_z / height))
    z_w[1:] *= 2.0
    k_eff2 = k2[:, None] - kz2[None, :]
    eta = 2 * decay_rate * omega / c ** 2
    denom = evals[None, None, :] - k_eff2[:, :, None] + 1j * eta[:, None, None]
    return 1.0 / denom, z_w, freqs


def compute_all_responses(speaker_idxs, evecs, inv_denom, z_w, n_freqs):
    """Frequency response from speakers to ALL listener positions.

    Returns: [n_points, n_freq] complex array.
    """
    response = np.zeros((evecs.shape[0], n_freqs), dtype=complex)
    for si in speaker_idxs:
        s_vec = evecs[si, :]
        G_modal = s_vec[None, None, :] * inv_denom
        G_weighted = np.einsum("fnm,n->fm", G_modal, z_w)
        response += evecs @ G_weighted.T
    return response


def compute_speaker_contribution(speaker_idx, listener_idx, evecs, inv_denom, z_w):
    """One speaker's contribution at one listener. Returns [n_freq] complex."""
    coupling = evecs[speaker_idx, :] * evecs[listener_idx, :]
    G_modal = coupling[None, None, :] * inv_denom
    return np.einsum("fnm,n->f", G_modal, z_w)


def nearest_idx(x, y, coords):
    """Index of nearest grid point to (x, y)."""
    return int(np.argmin((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2))


def score_responses(responses):
    """Score frequency response flatness. Lower = flatter = better.

    Args:
        responses: [n, n_freq] or [n_freq] complex array.
    Returns:
        [n] float array of scores.
    """
    if responses.ndim == 1:
        responses = responses[None, :]
    mag = np.abs(responses)
    mag = np.maximum(mag, 1e-30)
    db = 20 * np.log10(mag)
    db -= db.mean(axis=1, keepdims=True)
    std = db.std(axis=1)
    null_depth = -db.min(axis=1)
    peak_height = db.max(axis=1)
    null_pen = np.maximum(null_depth - 12, 0) * 0.5
    peak_pen = np.maximum(peak_height - 12, 0) * 0.3
    return std + null_pen + peak_pen


def response_stats(response):
    """Return (std, peak, null) in dB for a complex frequency response."""
    mag = np.abs(response)
    mag = np.maximum(mag, 1e-30)
    db = 20 * np.log10(mag)
    db -= np.mean(db)
    return db.std(), db.max(), db.min()
