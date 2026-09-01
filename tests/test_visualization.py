import numpy as np
from PIL import Image

from msi_core import RGBRepresentation, render_grayscale, render_rgb


def test_render_grayscale_maps_negative_float_data_to_uint8():
    image = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)
    rendered, mapping = render_grayscale(image)
    assert rendered.dtype == np.uint8
    assert rendered.tolist() == [[0, 128, 255]]
    assert mapping.source_min == -10.0
    assert mapping.source_max == 10.0


def test_render_grayscale_percentile_clips_outliers():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    image[0, 0] = -1000
    image[-1, -1] = 1000
    rendered, mapping = render_grayscale(image, stretch="percentile", percentiles=(1, 99))
    assert rendered.dtype == np.uint8
    assert rendered.min() == 0
    assert rendered.max() == 255
    assert mapping.low_percentile == 1
    assert mapping.high_percentile == 99


def test_render_rgb_uses_manifest_band_ids():
    from msi_core import BandMetadata, Stack

    data = np.stack(
        [
            np.full((4, 5), 100, dtype=np.uint16),
            np.full((4, 5), 200, dtype=np.uint16),
            np.full((4, 5), 300, dtype=np.uint16),
        ]
    )
    stack = Stack(
        data=data,
        bands=[
            BandMetadata(index=0, name="R", extra={"band_id": "vis_r"}),
            BandMetadata(index=1, name="G", extra={"band_id": "vis_g"}),
            BandMetadata(index=2, name="B", extra={"band_id": "vis_b"}),
        ],
    )
    rep = RGBRepresentation(
        id="visible_rgb",
        label="Visible",
        role="visible",
        red_band="vis_r",
        green_band="vis_g",
        blue_band="vis_b",
    )
    rendered, metadata = render_rgb(stack, rep, channel_scaling="shared")
    assert rendered.shape == (4, 5, 3)
    assert rendered.dtype == np.uint8
    assert np.all(rendered[..., 0] == 0)
    assert np.all(rendered[..., 1] == 128)
    assert np.all(rendered[..., 2] == 255)
    assert metadata["representation_id"] == "visible_rgb"


def test_save_render_writes_rgb_png(tmp_path):
    from msi_core import save_render

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 255
    output = save_render(image, tmp_path / "rgb.png")
    loaded = np.asarray(Image.open(output))
    assert loaded.shape == (4, 5, 3)
    assert np.all(loaded[..., 0] == 255)
