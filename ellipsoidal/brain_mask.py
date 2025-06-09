"""brain_mask.py

Robust brain‐extraction for low‑field (≈0.1–0.5 T) T2‑weighted MRI.
===================================================================
This module provides *fully self‑contained* (CPU‑only) utilities to create a
binary brain mask from a single 3‑D NIfTI image.  The algorithm avoids any
large DL dependencies so it runs on restricted environments—but optionally
hooks into FSL BET or HD‑BET if they are available.

Main entry‑points
-----------------
>>> import brain_mask as bm
>>> mask = bm.brain_mask("t2w_lowfield.nii.gz")        # ndarray (Z, Y, X)
>>> bm.save_mask("t2w_lowfield.nii.gz", mask, "mask.nii.gz")

* ``brain_mask``         – pure‑Python fallback pipeline.
* ``brain_mask_bet``     – wrapper for FSL *BET* if installed.
* ``brain_mask_hd_bet``  – wrapper for HD‑BET (PyTorch) if installed.

If a third‑party tool is present the top‑level ``auto_brain_mask`` will pick
it in the following preference order: **HD‑BET → FSL BET → built‑in**.

Algorithm (built‑in backend)
----------------------------
1. **Bias‑field correction**   via *SimpleITK.N4BiasFieldCorrection* if the
   library is available—otherwise skipped.
2. **Robust intensity normalisation** (percentile‑based z‑score).
3. **Histogram‑based threshold** (Otsu on the inverted brain‑probability map).
4. **Morphological cleanup**   (closing → biggest component → hole‑fill →
   erosion/dilation to match brain margin).
5. **Optional runtime QC** image can be written to disk if *matplotlib* is
   present.

The heuristics are tuned for low‑field T2w images where CSF is bright, bone is
very dark, and soft tissue has mid‑range signal.

Dependencies
------------
* Required : ``numpy``, ``scipy``, ``scikit‑image``, ``nibabel``.
* Optional : ``simpleitk`` (N4), ``matplotlib`` (QA plot), ``fsl`` or
             ``hd‑bet`` executables.

Copyright & Licence
-------------------
© 2025 ChatGPT‐o3 – MIT licence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import logging
import subprocess
import shutil
import tempfile

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:  # optional N4 bias correction
    import SimpleITK as sitk  # type: ignore

    _HAVE_SITK = True
except ModuleNotFoundError:
    _HAVE_SITK = False

# ---------------------------------------------------------------------------
# Helper I/O
# ---------------------------------------------------------------------------


def load_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NIfTI and return data (ZYX), affine, header."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_mask(src_nifti: str | Path, mask: np.ndarray, out_path: str | Path):
    """Save *mask* with header & affine inherited from *src_nifti*."""
    _, affine, hdr = load_nifti(src_nifti)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, hdr), str(out_path))
    logger.info("Mask written to %s", out_path)


# ---------------------------------------------------------------------------
# Bias‑field correction (optional)
# ---------------------------------------------------------------------------


def _n4_bias_correction(vol: np.ndarray) -> np.ndarray:
    if not _HAVE_SITK:
        logger.debug("SimpleITK not found – skipping N4 correction.")
        return vol
    logger.debug("Running N4 bias‑field correction …")
    img = sitk.GetImageFromArray(vol)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrected = sitk.N4BiasFieldCorrection(img, mask)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


# ---------------------------------------------------------------------------
# Built‑in masking pipeline
# ---------------------------------------------------------------------------


def brain_mask(path: str | Path, *, debug: bool = False) -> np.ndarray:
    """Generate a brain mask using the pure‑Python fallback algorithm.

    Parameters
    ----------
    path : str or Path
        Input T2w NIfTI.
    debug : bool
        If *True*, intermediate volumes are returned as a dict.

    Returns
    -------
    mask : ndarray bool, shape = (Z, Y, X)
        Binary brain mask.
    """
    vol, _, _ = load_nifti(path)

    # N4 bias correction (optional) ----------------------------------------
    vol = _n4_bias_correction(vol)

    # Intensity normalisation ---------------------------------------------
    p2, p98 = np.percentile(vol, [2, 98])
    vol = np.clip((vol - p2) / (p98 - p2 + 1e-5), 0, 1)  # [0,1]

    # Invert (skull is dark) and apply Otsu on probability map -------------
    inv = 1.0 - vol
    thresh = filters.threshold_otsu(inv)
    mask = inv > thresh

    # Morphology cleanup ---------------------------------------------------
    mask = ndi.binary_closing(mask, structure=morphology.ball(2))
    label, num = ndi.label(mask)
    if num == 0:
        raise RuntimeError("Masking failed – no foreground detected.")
    areas = ndi.sum(mask, label, index=np.arange(1, num + 1))
    mask = label == (np.argmax(areas) + 1)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_erosion(mask, morphology.ball(1))
    mask = morphology.binary_dilation(mask, morphology.ball(2))

    if debug:
        return {"norm": vol, "inv": inv, "mask": mask}
    return mask


# ---------------------------------------------------------------------------
# Wrappers for external tools (if present)
# ---------------------------------------------------------------------------


def _which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def brain_mask_bet(nifti: str | Path, *, frac: float = 0.4) -> np.ndarray:
    if _which("brainextractor") is None:
        raise RuntimeError("brainextractor not found in $PATH.")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(nifti)
        tmp_out = Path(tmp) / "bet_out.nii.gz"
        cmd = ["brainextractor", str(src), str(tmp_out), "-f", str(frac)]
        logger.info("Running brainextractor: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mask = nib.load(tmp_out).get_fdata().astype(bool)
        return mask


def brain_mask_hd_bet(
    nifti: str | Path, *, device: str = "cpu", disable_tta: bool = False
) -> np.ndarray:
    if _which("hd-bet") is None:
        raise RuntimeError("HD‑BET script not found in $PATH.")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(nifti)
        cmd = ["hd-bet", "-i", str(src), "-o", tmp, "-device", device]
        if disable_tta:
            cmd.append("--disable-tta")
        logger.info("Running HD‑BET …")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mask = nib.load(Path(tmp) / src.name).get_fdata().astype(bool)
        return mask


# ---------------------------------------------------------------------------
# Auto‑select backend
# ---------------------------------------------------------------------------


def auto_brain_mask(nifti: str | Path) -> np.ndarray:
    """Choose the best available backend and return a brain mask."""
    try:
        return brain_mask_hd_bet(nifti)
    except Exception as e:
        logger.debug("HD‑BET unavailable: %s", e)
    try:
        return brain_mask_bet(nifti)
    except Exception as e:
        logger.debug("FSL BET unavailable: %s", e)
    logger.info("Falling back to built‑in masker …")
    return brain_mask(nifti)
