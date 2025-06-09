"""mask_enhance.py
=================================
Utility functions to *refine* binary brain masks.

Typical usage
-------------
>>> from mask_enhance import enhance_mask
>>> refined = enhance_mask(raw_mask, dilate_mm=2, smooth_mm=3)

API
---
* ``dilate(mask, radius_vox=1)``
* ``erode(mask, radius_vox=1)``
* ``gaussian_smooth(mask, sigma_vox=1.5)``  – keeps boolean output.
* ``largest_cc(mask)``  – retain biggest connected component.
* ``fill_holes_3d(mask)``
* ``enhance_mask(mask, dilate_mm=2, smooth_mm=2, spacing=(1,1,1))``

All functions are CPU‑only and rely solely on ``numpy`` + ``scipy``.

Licence: MIT – © 2025 ChatGPT‑o3
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Structuring element helpers
# ---------------------------------------------------------------------------


def _ball(radius: float, spacing: Tuple[float, float, float]) -> np.ndarray:
    """Return a 3‑D bool structuring element approximating a metric *ball*."""
    rz, ry, rx = (radius / s for s in spacing)
    z, y, x = np.ogrid[
        -rz : rz + 1,
        -ry : ry + 1,
        -rx : rx + 1,
    ]
    return (x**2 + y**2 + z**2) <= 1.0


# ---------------------------------------------------------------------------
# Basic morphological ops
# ---------------------------------------------------------------------------


def dilate(mask: np.ndarray, *, radius_vox: int = 1) -> np.ndarray:
    strel = ndi.generate_binary_structure(3, 1)
    if radius_vox > 1:
        strel = ndi.iterate_structure(strel, radius_vox)
    return ndi.binary_dilation(mask, structure=strel)


def erode(mask: np.ndarray, *, radius_vox: int = 1) -> np.ndarray:
    strel = ndi.generate_binary_structure(3, 1)
    if radius_vox > 1:
        strel = ndi.iterate_structure(strel, radius_vox)
    return ndi.binary_erosion(mask, structure=strel)


# ---------------------------------------------------------------------------
# Connected components & hole‑fill
# ---------------------------------------------------------------------------


def largest_cc(mask: np.ndarray) -> np.ndarray:
    label, num = ndi.label(mask)
    if num == 0:
        return mask.astype(bool)
    areas = ndi.sum(mask, label, index=np.arange(1, num + 1))
    return label == (np.argmax(areas) + 1)


def fill_holes_3d(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def gaussian_smooth(
    mask: np.ndarray, *, sigma_vox: float = 1.5, thresh: float = 0.5
) -> np.ndarray:
    """Gaussian‑blur a mask then re‑threshold to boolean.

    Parameters
    ----------
    sigma_vox : float
        Gaussian sigma in *voxels* (isotropic).
    thresh : float
        Threshold on blurred mask.  0.5 retains original volume; lower → grow.
    """
    blurred = ndi.gaussian_filter(mask.astype(float), sigma=sigma_vox)
    return blurred > thresh


# ---------------------------------------------------------------------------
# High‑level convenience
# ---------------------------------------------------------------------------


def enhance_mask(
    mask: np.ndarray,
    *,
    dilate_mm: float = 3.0,
    smooth_mm: float = 3.0,
    spacing: Tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Expand *mask* outward then smooth jagged edges.

    Parameters
    ----------
    mask : ndarray bool
    dilate_mm : float
        Radius of dilation in **millimetres** – compensates skull‑stripping
        under‑segmentation.  Set to 0 to disable.
    smooth_mm : float
        Gaussian sigma in millimetres applied after dilation + cleanup.
    spacing : (Z, Y, X) voxel spacing in mm.  If *None* assumes isotropic 1 mm.
    """
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    mask_bool = mask.astype(bool, copy=False)

    # 1. Keep only largest connected component
    mask_proc = largest_cc(mask_bool)

    # 2. Optional dilation
    if dilate_mm > 0:
        strel = _ball(dilate_mm, spacing)
        mask_proc = ndi.binary_dilation(mask_proc, structure=strel)

    # 3. Hole‑fill then smooth & re‑threshold
    mask_proc = fill_holes_3d(mask_proc)
    if smooth_mm > 0:
        sigma_vox = tuple(smooth_mm / sp for sp in spacing)
        mask_proc = gaussian_smooth(mask_proc, sigma_vox=float(np.mean(sigma_vox)))

    return mask_proc.astype(bool)
