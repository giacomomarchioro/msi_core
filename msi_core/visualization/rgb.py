from __future__ import annotations

from typing import Literal

import numpy as np

from ..io.manifest import RGBRepresentation
from ..stack import Stack
from .grayscale import RenderMapping, StretchMode, render_grayscale

ChannelScaling = Literal["independent", "shared"]


def render_rgb(
    stack: Stack,
    representation: RGBRepresentation,
    *,
    stretch: StretchMode = "minmax",
    percentiles: tuple[float, float] = (1.0, 99.0),
    channel_scaling: ChannelScaling = "independent",
) -> tuple[np.ndarray, dict[str, object]]:
    """Render an RGB representation declared by a :class:`DatasetManifest`.

    Logical band ids are read from ``BandMetadata.extra['band_id']``. The
    returned image is an 8-bit RGB array suitable for PNG/JPEG export or web
    display. The scientific stack is left unchanged.
    """

    lookup = {
        band.extra.get("band_id"): stack.data[index]
        for index, band in enumerate(stack.bands)
        if band.extra.get("band_id") is not None
    }
    ids = [representation.red_band, representation.green_band, representation.blue_band]
    missing = [band_id for band_id in ids if band_id not in lookup]
    if missing:
        raise KeyError(f"RGB representation references bands not present in stack: {missing}.")

    arrays = [np.asarray(lookup[band_id]) for band_id in ids]
    if channel_scaling == "independent":
        rendered_channels: list[np.ndarray] = []
        mappings: list[RenderMapping] = []
        for array in arrays:
            rendered, mapping = render_grayscale(
                array,
                stretch=stretch,
                percentiles=percentiles,
            )
            rendered_channels.append(rendered)
            mappings.append(mapping)
    elif channel_scaling == "shared":
        cube = np.stack(arrays, axis=0).astype(np.float64, copy=False)
        finite = np.isfinite(cube)
        if finite.any():
            values = cube[finite]
            if stretch == "minmax":
                black = float(values.min())
                white = float(values.max())
                low_pct = high_pct = None
            else:
                low_pct, high_pct = percentiles
                if not (0 <= low_pct < high_pct <= 100):
                    raise ValueError("percentiles must satisfy 0 <= low < high <= 100.")
                black, white = [float(v) for v in np.percentile(values, [low_pct, high_pct])]
        else:
            black = white = None
            low_pct = high_pct = None

        rendered_channels = []
        mappings = []
        for array in arrays:
            work = array.astype(np.float64, copy=False)
            if black is None or white is None or white <= black:
                normalized = np.zeros(work.shape, dtype=np.float64)
            else:
                normalized = np.clip((work - black) / (white - black), 0.0, 1.0)
                normalized[~np.isfinite(work)] = 0.0
            rendered_channels.append(np.rint(normalized * 255.0).astype(np.uint8))
            mappings.append(
                RenderMapping(
                    method=stretch,
                    source_min=black,
                    source_max=white,
                    low_percentile=low_pct,
                    high_percentile=high_pct,
                )
            )
    else:
        raise ValueError("channel_scaling must be 'independent' or 'shared'.")

    rgb = np.stack(rendered_channels, axis=-1)
    metadata = {
        "representation_id": representation.id,
        "label": representation.label,
        "role": representation.role,
        "band_ids": {"red": ids[0], "green": ids[1], "blue": ids[2]},
        "stretch": stretch,
        "channel_scaling": channel_scaling,
        "channels": {
            "red": mappings[0].to_dict(),
            "green": mappings[1].to_dict(),
            "blue": mappings[2].to_dict(),
        },
    }
    return rgb, metadata
