from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal

import numpy as np
import tifffile
from PIL import Image

from ..result import Result
from ..stack import Stack, _json_safe
from ..visualization import render_grayscale

RenderFormat = Literal["png", "jpeg"]


def save_result(
    result: Result,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
    save_render: bool = True,
    render_format: RenderFormat = "png",
    render_stretch: Literal["minmax", "percentile"] = "minmax",
    render_percentiles: tuple[float, float] = (1.0, 99.0),
    jpeg_quality: int = 95,
) -> Path:
    """Persist scientific result data and optional 8-bit publication-ready renders.

    Scientific arrays are always written unchanged as TIFF files under ``data/``.
    When ``save_render`` is true, each output band is also rendered to 8-bit PNG
    or JPEG under ``renders/``. Rendering changes only the display representation;
    it never changes the scientific TIFF values.
    """

    if render_format not in {"png", "jpeg"}:
        raise ValueError("render_format must be 'png' or 'jpeg'.")
    if not (1 <= jpeg_quality <= 100):
        raise ValueError("jpeg_quality must be in 1..100.")

    output_path = Path(output_dir).expanduser().resolve()
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    render_path = output_path / "renders"
    if save_render:
        render_path.mkdir(parents=True, exist_ok=True)

    base_prefix = _safe_name(prefix or result.operation_id)
    scientific_files: list[str] = []
    render_files: list[str] = []
    render_mappings: list[dict[str, object]] = []

    for index in range(result.output.n_bands):
        band = result.output[index]
        label = _safe_name(band.name or f"band_{index:03d}")
        stem = f"{base_prefix}_{index:03d}_{label}"

        scientific_filename = f"{stem}.tif"
        tifffile.imwrite(
            data_path / scientific_filename,
            band.data,
            photometric="minisblack",
        )
        scientific_files.append(f"data/{scientific_filename}")

        if save_render:
            rendered, mapping = render_grayscale(
                band.data,
                stretch=render_stretch,
                percentiles=render_percentiles,
            )
            suffix = ".png" if render_format == "png" else ".jpg"
            render_filename = f"{stem}{suffix}"
            image = Image.fromarray(rendered, mode="L")
            if render_format == "jpeg":
                image.save(render_path / render_filename, format="JPEG", quality=jpeg_quality)
            else:
                image.save(render_path / render_filename, format="PNG")
            render_files.append(f"renders/{render_filename}")
            render_mappings.append(
                {
                    "file": f"renders/{render_filename}",
                    **mapping.to_dict(),
                }
            )

    payload = result.to_metadata_dict()
    payload["schema_version"] = "1.1"
    payload["export"] = {
        "scientific_files": scientific_files,
        "scientific_format": "tiff",
        "scientific_values_preserved": True,
        "renders": {
            "enabled": save_render,
            "format": render_format if save_render else None,
            "bit_depth": 8 if save_render else None,
            "stretch": render_stretch if save_render else None,
            "percentiles": list(render_percentiles) if save_render and render_stretch == "percentile" else None,
            "jpeg_quality": jpeg_quality if save_render and render_format == "jpeg" else None,
            "files": render_files,
            "per_image_mapping": render_mappings,
        },
    }
    (output_path / "metadata.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def save_stack(
    stack: Stack,
    output_dir: str | Path,
    *,
    prefix: str = "stack",
    save_render: bool = True,
    render_format: RenderFormat = "png",
    render_stretch: Literal["minmax", "percentile"] = "minmax",
    render_percentiles: tuple[float, float] = (1.0, 99.0),
    jpeg_quality: int = 95,
) -> Path:
    result = Result(
        output=stack,
        operation_id="export_stack",
        operation_version="1.0.0",
        parameters={},
    )
    return save_result(
        result,
        output_dir,
        prefix=prefix,
        save_render=save_render,
        render_format=render_format,
        render_stretch=render_stretch,
        render_percentiles=render_percentiles,
        jpeg_quality=jpeg_quality,
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "image"
