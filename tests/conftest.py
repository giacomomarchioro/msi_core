from __future__ import annotations

import numpy as np
import pytest

from msi_core import BandMetadata, Stack


@pytest.fixture
def small_stack() -> Stack:
    y, x = np.mgrid[0:6, 0:8]
    data = np.stack(
        [
            x + y,
            2 * x + y,
            x + 3 * y,
        ],
        axis=0,
    ).astype(np.uint16)
    bands = [
        BandMetadata(index=0, source_index=0, name="365 nm", wavelength_nm=365),
        BandMetadata(index=1, source_index=1, name="450 nm", wavelength_nm=450),
        BandMetadata(index=2, source_index=2, name="650 nm", wavelength_nm=650),
    ]
    return Stack(data=data, bands=bands, metadata={"fixture": True})
