import numpy as np
import pytest

from msi_core import BandMetadata, Region, RegionError, Stack


def test_region_properties_and_slices():
    region = Region(x0=2, x1=7, y0=3, y1=9)
    assert region.width == 5
    assert region.height == 6
    assert region.slices == (slice(3, 9), slice(2, 7))


def test_region_rejects_invalid_coordinates():
    with pytest.raises(RegionError):
        Region(x0=4, x1=4, y0=0, y1=2)
    with pytest.raises(RegionError):
        Region(x0=-1, x1=4, y0=0, y1=2)


def test_region_bounds_validation():
    region = Region(x0=0, x1=11, y0=0, y1=5)
    with pytest.raises(RegionError):
        region.validate_within(width=10, height=5)


def test_stack_requires_contiguous_metadata_indices():
    data = np.zeros((2, 3, 4), dtype=np.uint16)
    with pytest.raises(ValueError):
        Stack(data=data, bands=[BandMetadata(index=0), BandMetadata(index=2)])


def test_stack_select_reindexes_and_preserves_source_indices(small_stack):
    selected = small_stack.select([2, 0])
    assert selected.shape == (2, 6, 8)
    assert [band.index for band in selected.bands] == [0, 1]
    assert [band.source_index for band in selected.bands] == [2, 0]
    assert [band.wavelength_nm for band in selected.bands] == [650, 365]
