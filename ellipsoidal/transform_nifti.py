import os
from pathlib import Path
from typing import Optional, Tuple
import nibabel as nib
import numpy as np

from .transform import (
    cartesian_to_ellipsoidal,
    ellipsoidal_to_cartesian,
    fit_ellipsoid,
    TransformMeta,
)
from .brain_mask import brain_mask_bet
from .mask_enhance import enhance_mask

NiftiLike = nib.Nifti1Image | str | Path
Nifti = nib.Nifti1Image


def _load_niftilike(nii: NiftiLike) -> Nifti:
    if isinstance(nii, str):
        img = nib.load(nii)
    elif isinstance(nii, Path):
        img = nib.load(str(nii))
    elif isinstance(nii, Nifti):
        img = nii
    return img


def _nib_from_np(arr: np.ndarray, affine: Optional[np.ndarray] = None) -> Nifti:
    return Nifti(arr, np.eye(4) if affine is None else affine)


def nifti_to_ellipsoidal(
    img: NiftiLike,
    *,
    r_bins: int = 128,
    theta_bins: int = 128,
    phi_bins: int = 128,
    equal_area_phi: bool = True,
    is_mask: bool = False,
    meta: Optional[TransformMeta | dict] = None,
    bet_kwargs: dict = {},
    enhance_kwargs: dict = {},
    **kwargs,
) -> dict:
    if is_mask == True and meta is None:
        raise ValueError("When `is_mask` is True, `meta` MUST be passed!")

    nii = _load_niftilike(img)
    vol = nii.get_fdata(dtype=np.float32)
    affine = nii.affine

    if not is_mask:  # a volume is passed
        # extract brain mask using BET
        brain_mask = brain_mask_bet(img, **bet_kwargs)
        # Enhance the mask using morphological operators
        brain_mask = enhance_mask(brain_mask, **enhance_kwargs)
        # Fit an ellipsoid to the mask using PCA
        fit_results = fit_ellipsoid(brain_mask)
        # Transform the coord system from Cartesian to Ellipsoidal
        ellip, meta = cartesian_to_ellipsoidal(
            vol=vol,
            **fit_results,
            r_bins=r_bins,
            theta_bins=theta_bins,
            phi_bins=phi_bins,
            equal_area_phi=equal_area_phi,
            interp_order=3,
        )
        meta.affine = affine
    else:
        if isinstance(meta, dict):
            meta = TransformMeta.fromdict(meta)
        brain_mask = None
        # Input is a mask, so a meta object is required
        ellip, _ = cartesian_to_ellipsoidal(
            vol=vol.astype(np.uint8),
            semi_axes=meta.semi_axes,
            rotation=meta.rotation,
            center=meta.center,
            r_bins=r_bins,
            theta_bins=theta_bins,
            phi_bins=phi_bins,
            equal_area_phi=equal_area_phi,
            interp_order=0,
        )
    return {"nii": _nib_from_np(ellip), "meta": meta.asdict(), "brain_mask": brain_mask}


def ellipsoidal_to_nifti(
    img: NiftiLike,
    meta: TransformMeta | dict,
    *,
    is_mask: bool = False,
    **kwargs,
) -> Nifti | Tuple[Nifti, dict]:
    nii = _load_niftilike(img)
    ellip = nii.get_fdata(dtype=np.float32)

    if isinstance(meta, dict):
        meta = TransformMeta.fromdict(meta)

    vol = ellipsoidal_to_cartesian(
        ell=ellip, meta=meta, interp_order=0 if is_mask else 3
    )

    vol = vol.astype(np.uint8 if is_mask else np.float32)

    return {"nii": _nib_from_np(vol, meta.affine), "meta": meta.asdict()}
