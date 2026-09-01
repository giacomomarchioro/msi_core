from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

StretchMode = Literal["minmax", "percentile"]


@dataclass(frozen=True, slots=True)
class RenderMapping:
    method: StretchMode
    source_min: float | None
    source_max: float | None
    output_min: int = 0
    output_max: int = 255
    low_percentile: float | None = None
    high_percentile: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "method": self.method,
            "source_min": self.source_min,
            "source_max": self.source_max,
            "output_min": self.output_min,
            "output_max": self.output_max,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
        }


def render_grayscale(
    image: np.ndarray,
    *,
    stretch: StretchMode = "minmax",
    percentiles: tuple[float, float] = (1.0, 99.0),
    invert: bool = False,
) -> tuple[np.ndarray, RenderMapping]:
    """Render a scientific 2-D array as display-ready 8-bit grayscale.

    Rendering never mutates the scientific input array. ``minmax`` maps the
    finite input minimum and maximum linearly to 0..255. ``percentile`` uses
    percentile-derived black/white points and clips values outside that range.
    """

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"render_grayscale expects a 2-D array, received shape {values.shape}.")

    work = values.astype(np.float64, copy=False)
    finite = np.isfinite(work)
    if not finite.any():
        rendered = np.zeros(work.shape, dtype=np.uint8)
        mapping = RenderMapping(method=stretch, source_min=None, source_max=None)
        return rendered, mapping

    finite_values = work[finite]
    if stretch == "minmax":
        black = float(finite_values.min())
        white = float(finite_values.max())
        low_pct = high_pct = None
    elif stretch == "percentile":
        low_pct, high_pct = percentiles
        if not (0 <= low_pct < high_pct <= 100):
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100.")
        black, white = [float(value) for value in np.percentile(finite_values, [low_pct, high_pct])]
    else:  # pragma: no cover - protected by Literal in typed callers
        raise ValueError(f"Unknown stretch mode: {stretch}")

    if white <= black:
        normalized = np.zeros(work.shape, dtype=np.float64)
    else:
        normalized = (work - black) / (white - black)
        normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite] = 0.0
    if invert:
        normalized = 1.0 - normalized
    rendered = np.rint(normalized * 255.0).astype(np.uint8)
    mapping = RenderMapping(
        method=stretch,
        source_min=black,
        source_max=white,
        low_percentile=low_pct,
        high_percentile=high_pct,
    )
    return rendered, mapping
