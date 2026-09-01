from __future__ import annotations

from dataclasses import dataclass

from .errors import RegionError


@dataclass(frozen=True, slots=True)
class Region:
    """A half-open rectangular image region: ``[x0, x1) x [y0, y1)``."""

    x0: int
    x1: int
    y0: int
    y1: int

    def __post_init__(self) -> None:
        values = (self.x0, self.x1, self.y0, self.y1)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise RegionError("Region coordinates must be integers.")
        if self.x0 < 0 or self.y0 < 0:
            raise RegionError("Region start coordinates must be non-negative.")
        if self.x1 <= self.x0:
            raise RegionError("x1 must be greater than x0.")
        if self.y1 <= self.y0:
            raise RegionError("y1 must be greater than y0.")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def validate_within(self, *, width: int, height: int) -> None:
        if self.x1 > width or self.y1 > height:
            raise RegionError(
                "Region is outside image bounds: "
                f"region=({self.x0}, {self.x1}, {self.y0}, {self.y1}), "
                f"image_width={width}, image_height={height}."
            )

    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1}

    @classmethod
    def from_xywh(cls, *, x: int, y: int, width: int, height: int) -> "Region":
        if width <= 0 or height <= 0:
            raise RegionError("Region width and height must be positive.")
        return cls(x0=x, x1=x + width, y0=y, y1=y + height)
