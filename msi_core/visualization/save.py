from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_render(
    image: np.ndarray,
    path: str | Path,
    *,
    jpeg_quality: int = 95,
) -> Path:
    """Save an 8-bit grayscale or RGB render as PNG/JPEG.

    ``image`` must already be a rendered ``uint8`` array with shape ``(y, x)``
    or ``(y, x, 3)``. Scientific arrays should first be converted with
    :func:`render_grayscale` or :func:`render_rgb`.
    """

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("save_render expects a uint8 rendered image.")
    if array.ndim == 2:
        mode = "L"
    elif array.ndim == 3 and array.shape[-1] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Rendered image must have shape (y, x) or (y, x, 3); got {array.shape}.")

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    image_obj = Image.fromarray(array, mode=mode)
    if suffix in {".jpg", ".jpeg"}:
        if not (1 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be in 1..100.")
        image_obj.save(output, format="JPEG", quality=jpeg_quality)
    elif suffix == ".png":
        image_obj.save(output, format="PNG")
    else:
        raise ValueError("Render output path must end in .png, .jpg, or .jpeg.")
    return output
