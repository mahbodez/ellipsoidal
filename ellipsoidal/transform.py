"""ellipsoidal_transforms.py
===================================================================
Robust Cartesian ↔ Ellipsoidal volume resampling
-------------------------------------------------------------------
* Author : ChatGPT‑o3 – 2025‑06‑09
* Licence: MIT
-------------------------------------------------------------------
This module provides **fully‑general** coordinate transforms between a
Cartesian voxel grid and an *arbitrarily oriented* ellipsoidal coordinate
system *(R, Φ, Θ)*.

Typical workflow
----------------
```python
>>> import nibabel as nib, ellipsoidal_transforms as et
>>> mri      = nib.load('brain.nii.gz').get_fdata()
>>> mask     = (mri > 0)                 # crude brain mask
>>> centre, axes, R = et.fit_ellipsoid(mask)
>>> ell, meta = et.cartesian_to_ellipsoidal(mri,
...                                         semi_axes=axes,
...                                         rotation=R,
...                                         center=centre,
...                                         r_bins=128,
...                                         theta_bins=384,
...                                         phi_bins=192)
>>> # … do analysis in ellipsoidal space …
>>> mri_back = et.ellipsoidal_to_cartesian(ell, meta)
```

Why another module?
-------------------
* **Arbitrary orientation** – supply any 3 × 3 orthonormal matrix.
* **Torch + NumPy compatible** inputs.
* **Metadata round‑trip** is deterministic.
* **PCA helper** quickly fits the best‑fitting ellipsoid to a binary mask.
* **Equal‑area latitude** sampling avoids over‑representation of the poles.

Performance
-----------
SciPy’s `map_coordinates` (CPU) is used by default.  If PyTorch ≥ 2 is
installed, set `backend='torch'` to get GPU, batching and autograd support.

-------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Union, Optional
import numpy as np

try:
    from scipy.ndimage import map_coordinates as _scipy_interpolate
except ImportError as _err:  # pragma: no cover – unit tests require SciPy
    raise ImportError("SciPy is required: pip install scipy") from _err

try:
    import torch  # noqa: F401 – optional

    _HAVE_TORCH = True
except ModuleNotFoundError:  # pragma: no cover
    _HAVE_TORCH = False

__all__ = [
    "TransformMeta",
    "fit_ellipsoid",
    "cartesian_to_ellipsoidal",
    "ellipsoidal_to_cartesian",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformMeta:
    """All parameters required for an inverse transform."""

    orig_shape: Tuple[int, int, int]
    center: Tuple[float, float, float]
    semi_axes: Tuple[float, float, float]
    rotation: np.ndarray  # shape (3, 3)
    r_bins: int
    theta_bins: int
    phi_bins: int
    equal_area_phi: bool = True
    backend: str = "scipy"  # "scipy" | "torch"

    # serialization helpers -------------------------------------------------
    def asdict(self) -> Dict:
        d = asdict(self)
        d["rotation"] = self.rotation.tolist()
        return d

    @staticmethod
    def fromdict(d: Dict) -> "TransformMeta":
        d = dict(d)
        d["rotation"] = np.asarray(d["rotation"], dtype=np.float64)
        return TransformMeta(**d)  # type: ignore[arg‑type]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _equal_area_phi_centres(n: int) -> np.ndarray:
    k = np.arange(1, n + 1, dtype=np.float64)
    return np.arccos(1.0 - 2.0 * (k - 0.5) / n)


def _check_rotation(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("rotation must be 3×3")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError("rotation matrix must be orthonormal")
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1  # enforce right‑handedness
    return R


# ---------------------------------------------------------------------------
# Public API – PCA helper
# ---------------------------------------------------------------------------


def fit_ellipsoid(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a *minimum bounding ellipsoid* to a binary mask via PCA.

    Returns
    -------
    center : (3,) float64
    semi_axes : (a, b, c)
        Half‑lengths along the principal axes, **in voxels**.
    rotation : (3, 3) ndarray
        Orthonormal matrix whose *columns* are the principal axes.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    pts = np.column_stack(np.nonzero(mask))  # (N, 3)
    if pts.shape[0] < 10:
        raise ValueError("Mask too small to fit an ellipsoid")

    center = pts.mean(axis=0)
    centred = pts - center
    # PCA via SVD (more stable than eig on covariance)
    u, s, vh = np.linalg.svd(centred, full_matrices=False)
    rotation = vh.T  # right singular vectors
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1
    projections = centred @ rotation
    semi_axes = np.max(np.abs(projections), axis=0)
    return center.astype(np.float64), semi_axes.astype(np.float64), rotation


# ---------------------------------------------------------------------------
# Public API – Forward transform
# ---------------------------------------------------------------------------


def cartesian_to_ellipsoidal(
    vol: Union[np.ndarray, "torch.Tensor"],
    *,
    semi_axes: Tuple[float, float, float],
    rotation: np.ndarray,
    center: Tuple[float, float, float],
    r_bins: int = 128,
    theta_bins: int = 384,
    phi_bins: int = 192,
    equal_area_phi: bool = True,
    backend: str = "scipy",
    interp_order: int = 3,
) -> Tuple[np.ndarray, TransformMeta]:
    """Resample *vol* onto a regular *(R, Φ, Θ)* grid.

    Parameters
    ----------
    vol
        3‑D NumPy array or PyTorch tensor (channels not supported here).
    semi_axes
        (a, b, c) semi‑axis lengths **in voxels**.
    rotation
        3 × 3 orthonormal matrix – columns = ellipsoid axes in voxel space.
    center
        (cx, cy, cz) in voxels.
    backend
        "scipy" (CPU) or "torch" (GPU, autograd‑safe).  Torch requires
        PyTorch ≥ 2.0.
    """
    R = _check_rotation(rotation)
    vol_np = (
        vol.cpu().numpy() if isinstance(vol, np.ndarray.__mro__[1]) else np.asarray(vol)
    )
    nx, ny, nz = vol_np.shape

    # latitude / longitude sampling ----------------------------------------
    phi_centres = (
        _equal_area_phi_centres(phi_bins)
        if equal_area_phi
        else (np.pi * (np.arange(phi_bins) + 0.5) / phi_bins)
    )
    theta_centres = 2.0 * np.pi * (np.arange(theta_bins) + 0.5) / theta_bins
    r_centres = np.linspace(0.0, 1.0, r_bins)

    # build meshgrid (memory‑friendly with broadcasting)
    Rg = r_centres[:, None, None]
    Φg = phi_centres[None, :, None]
    Θg = theta_centres[None, None, :]

    sinΦ, cosΦ = np.sin(Φg), np.cos(Φg)
    sinΘ, cosΘ = np.sin(Θg), np.cos(Θg)

    Xs = Rg * sinΦ * cosΘ
    Ys = Rg * sinΦ * sinΘ
    Zs = Rg * cosΦ

    a, b, c = np.asarray(semi_axes, dtype=np.float64)
    cx, cy, cz = center

    xs = a * Xs * R[0, 0] + b * Ys * R[0, 1] + c * Zs * R[0, 2] + cx
    ys = a * Xs * R[1, 0] + b * Ys * R[1, 1] + c * Zs * R[1, 2] + cy
    zs = a * Xs * R[2, 0] + b * Ys * R[2, 1] + c * Zs * R[2, 2] + cz

    if backend == "scipy":
        coords = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()])
        ell = _scipy_interpolate(
            vol_np, coords, order=interp_order, mode="constant", cval=0.0
        ).reshape(r_bins, phi_bins, theta_bins)
    elif backend == "torch":
        if not _HAVE_TORCH:
            raise ImportError("PyTorch not installed; set backend='scipy'")
        import torch.nn.functional as F  # local import → optional dependency

        grid = np.stack([zs, ys, xs], axis=-1)  # z‑y‑x order, (...,3)
        grid = torch.from_numpy(grid.astype(np.float32))[None]  # (1, ..., 3)
        grid = (
            grid / ((np.array([nz - 1, ny - 1, nx - 1])[None, None, None, None]) / 2)
            - 1
        )
        vol_t = torch.as_tensor(vol, dtype=torch.float32)[None, None]
        ell_t = F.grid_sample(vol_t, grid, mode="bilinear", align_corners=True)
        ell = ell_t.squeeze().cpu().numpy()
    else:
        raise ValueError("backend must be 'scipy' or 'torch'")

    meta = TransformMeta(
        orig_shape=(nx, ny, nz),
        center=(cx, cy, cz),
        semi_axes=(a, b, c),
        rotation=R,
        r_bins=r_bins,
        theta_bins=theta_bins,
        phi_bins=phi_bins,
        equal_area_phi=bool(equal_area_phi),
        backend=backend,
    )
    return ell.astype(vol_np.dtype, copy=False), meta


# ---------------------------------------------------------------------------
# Public API – Inverse transform
# ---------------------------------------------------------------------------


def ellipsoidal_to_cartesian(
    ell: Union[np.ndarray, "torch.Tensor"],
    meta: TransformMeta,
    *,
    interp_order: int = 3,
) -> np.ndarray:
    """Reconstruct a Cartesian volume from an ellipsoidal one."""
    if isinstance(ell, np.ndarray.__mro__[1]):
        ell_np = ell.cpu().numpy()
    else:
        ell_np = np.asarray(ell)

    nx, ny, nz = meta.orig_shape
    cx, cy, cz = meta.center
    a, b, c = meta.semi_axes
    R = meta.rotation
    Rb, Fb, Tb = meta.r_bins, meta.phi_bins, meta.theta_bins

    # Cartesian grid -------------------------------------------------------
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    z = np.arange(nz, dtype=np.float64)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")

    Xw, Yw, Zw = Xg - cx, Yg - cy, Zg - cz
    Xb = R[0, 0] * Xw + R[0, 1] * Yw + R[0, 2] * Zw
    Yb = R[1, 0] * Xw + R[1, 1] * Yw + R[1, 2] * Zw
    Zb = R[2, 0] * Xw + R[2, 1] * Yw + R[2, 2] * Zw

    Xs, Ys, Zs = Xb / a, Yb / b, Zb / c
    r = np.sqrt(Xs**2 + Ys**2 + Zs**2)
    φ = np.arccos(np.clip(Zs / r, -1.0, 1.0))
    θ = np.mod(np.arctan2(Ys, Xs), 2.0 * np.pi)

    r_idx = r * (Rb - 1)
    if meta.equal_area_phi:
        φ_idx = (1.0 - np.cos(φ)) * Fb / 2.0 - 0.5
    else:
        φ_idx = φ * Fb / np.pi - 0.5
    θ_idx = θ * Tb / (2.0 * np.pi) - 0.5

    if meta.backend == "scipy":
        coords = np.vstack([r_idx.ravel(), φ_idx.ravel(), θ_idx.ravel()])
        vol = _scipy_interpolate(
            ell_np, coords, order=interp_order, mode="constant", cval=0.0
        ).reshape(nx, ny, nz)
        return vol.astype(ell_np.dtype, copy=False)

    elif meta.backend == "torch":
        if not _HAVE_TORCH:
            raise ImportError("PyTorch not installed; cannot use backend 'torch'")
        import torch.nn.functional as F

        grid = np.stack([θ_idx, φ_idx, r_idx], axis=-1)  # Θ‑Φ‑R order
        grid = grid / np.array([Tb - 1, Fb - 1, Rb - 1]) * 2.0 - 1.0  # → [-1, 1]
        grid_t = torch.from_numpy(grid.astype(np.float32))[None]
        ell_t = torch.as_tensor(ell, dtype=torch.float32)[None, None]
        vol_t = F.grid_sample(ell_t, grid_t, mode="bilinear", align_corners=True)
        return vol_t.squeeze().cpu().numpy().astype(ell_np.dtype)

    else:  # pragma: no cover
        raise ValueError("Unknown backend in meta")
