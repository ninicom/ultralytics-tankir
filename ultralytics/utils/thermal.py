# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""TankIR thermal input encodings (single-channel TIR -> informative 3-channel input, no network change).

phys3      : [I, CLAHE(I), |Sobel(I)|] - intensity, local contrast and hull/barrel edges instead of [I, I, I].
temporal3  : [I(t-2) -> t, I(t-1) -> t, I(t)] - previous frames warped onto frame t with an ECC affine (camera motion
             removed), so moving vehicles differ across channels while static hot rocks do not.
"""

from __future__ import annotations

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 view of a gray or BGR image."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def phys3(img: np.ndarray) -> np.ndarray:
    """Encode a thermal image as [I, CLAHE(I), Sobel magnitude] (uint8, HxWx3)."""
    g = to_gray(img)
    c = _CLAHE.apply(g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    m = np.clip(m / (np.percentile(m, 99.5) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([g, c, m])


def align_to(src: np.ndarray, dst: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Warp gray `src` onto gray `dst` with an ECC affine estimated at reduced resolution."""
    s = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.resize(dst, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            d, s, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4), None, 5
        )
    except cv2.error:
        pass
    warp[:, 2] /= scale
    return cv2.warpAffine(
        src,
        warp,
        (dst.shape[1], dst.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def temporal3(frames: list[np.ndarray]) -> np.ndarray:
    """Encode [I(t-2), I(t-1), I(t)] as one aligned 3-channel uint8 image."""
    g = [to_gray(f) for f in frames]
    t = g[-1]
    return np.dstack([align_to(g[0], t), align_to(g[1], t), t])


def encode(img: np.ndarray, mode: str | None) -> np.ndarray:
    """Apply the single-frame encoding named by `mode` ('' / None = unchanged)."""
    if not mode:
        return img
    if mode == "phys3":
        return phys3(img)
    raise ValueError(f"unknown ir_input '{mode}' (temporal3 is built offline from sequences)")
