from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from msi_core import BandOrderError, Region, TiffFormatError, load_tiff_folder


def _write(path, value, shape=(10, 12)):
    tifffile.imwrite(path, np.full(shape, value, dtype=np.uint16), photometric="minisblack")


def test_orders_tiffs_by_wavelength_and_reads_roi(tmp_path):
    _write(tmp_path / "capture_650nm.tif", 650)
    _write(tmp_path / "capture_365nm.tif", 365)
    _write(tmp_path / "capture_450nm.tif", 450)

    stack = load_tiff_folder(
        tmp_path,
        region=Region(x0=2, x1=7, y0=3, y1=9),
    )

    assert stack.shape == (3, 6, 5)
    assert [band.wavelength_nm for band in stack.bands] == [365, 450, 650]
    assert stack.data[:, 0, 0].tolist() == [365, 450, 650]
    assert stack.metadata["band_order_source"] == "filename_wavelength"


def test_orders_tiffs_by_explicit_band_index(tmp_path):
    _write(tmp_path / "sample_band_010.tif", 10)
    _write(tmp_path / "sample_band_002.tif", 2)
    stack = load_tiff_folder(tmp_path)
    assert stack.data[:, 0, 0].tolist() == [2, 10]
    assert stack.bands[0].source_index == 2
    assert stack.bands[1].source_index == 10


def test_ambiguous_band_names_raise(tmp_path):
    _write(tmp_path / "blue.tif", 1)
    _write(tmp_path / "red.tif", 2)
    with pytest.raises(BandOrderError):
        load_tiff_folder(tmp_path)


def test_partial_wavelength_names_raise(tmp_path):
    _write(tmp_path / "365nm.tif", 1)
    _write(tmp_path / "unknown.tif", 2)
    with pytest.raises(BandOrderError):
        load_tiff_folder(tmp_path)


def test_manifest_controls_order_and_metadata(tmp_path):
    _write(tmp_path / "uv.tif", 4)
    _write(tmp_path / "red.tif", 8)
    manifest = {
        "schema_version": "1.0",
        "bands": [
            {"index": 0, "file": "red.tif", "name": "Red", "wavelength_nm": 650},
            {"index": 1, "file": "uv.tif", "name": "UV", "wavelength_nm": 365},
        ],
    }
    (tmp_path / "bands.json").write_text(json.dumps(manifest), encoding="utf-8")
    stack = load_tiff_folder(tmp_path)
    assert stack.data[:, 0, 0].tolist() == [8, 4]
    assert [band.name for band in stack.bands] == ["Red", "UV"]
    assert stack.metadata["band_order_source"] == "manifest"


def test_manifest_must_list_all_tiffs(tmp_path):
    _write(tmp_path / "a.tif", 1)
    _write(tmp_path / "b.tif", 2)
    manifest = {
        "schema_version": "1.0",
        "bands": [{"index": 0, "file": "a.tif"}],
    }
    (tmp_path / "bands.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BandOrderError):
        load_tiff_folder(tmp_path)



def test_explicit_relative_manifest_is_resolved_inside_folder(tmp_path):
    _write(tmp_path / "first.tif", 1)
    _write(tmp_path / "second.tif", 2)
    manifest = {
        "schema_version": "1.0",
        "bands": [
            {"index": 0, "file": "second.tif", "name": "Second"},
            {"index": 1, "file": "first.tif", "name": "First"},
        ],
    }
    (tmp_path / "custom.json").write_text(json.dumps(manifest), encoding="utf-8")
    stack = load_tiff_folder(tmp_path, manifest="custom.json")
    assert stack.data[:, 0, 0].tolist() == [2, 1]

def test_splits_rgb_tiff_into_three_bands_in_rgb_order(tmp_path):
    rgb = np.zeros((5, 6, 3), dtype=np.uint16)
    rgb[..., 0] = 100
    rgb[..., 1] = 200
    rgb[..., 2] = 300
    tifffile.imwrite(tmp_path / "band_001.tif", rgb, photometric="rgb")

    stack = load_tiff_folder(tmp_path)

    assert stack.shape == (3, 5, 6)
    assert stack.data[:, 0, 0].tolist() == [100, 200, 300]
    assert [band.channel for band in stack.bands] == ["R", "G", "B"]
    assert [band.source_file for band in stack.bands] == ["band_001.tif"] * 3
    assert [band.source_index for band in stack.bands] == [1, 1, 1]


def test_rgb_roi_is_applied_before_channel_stack_materialization(tmp_path):
    y, x = np.indices((8, 10))
    rgb = np.stack([x, y, x + y], axis=-1).astype(np.uint16)
    tifffile.imwrite(tmp_path / "band_001.tif", rgb, photometric="rgb")

    stack = load_tiff_folder(
        tmp_path,
        region=Region(x0=2, x1=7, y0=3, y1=6),
    )

    assert stack.shape == (3, 3, 5)
    assert np.array_equal(stack.data[0], rgb[3:6, 2:7, 0])
    assert np.array_equal(stack.data[1], rgb[3:6, 2:7, 1])
    assert np.array_equal(stack.data[2], rgb[3:6, 2:7, 2])


def test_rgb_file_stays_consecutive_with_other_ordered_sources(tmp_path):
    _write(tmp_path / "band_001.tif", 10, shape=(5, 6))
    rgb = np.zeros((5, 6, 3), dtype=np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 20, 30, 40
    tifffile.imwrite(tmp_path / "band_002.tif", rgb, photometric="rgb")
    _write(tmp_path / "band_003.tif", 50, shape=(5, 6))

    stack = load_tiff_folder(tmp_path)

    assert stack.data[:, 0, 0].tolist() == [10, 20, 30, 40, 50]
    assert [band.channel for band in stack.bands] == [None, "R", "G", "B", None]


def test_rejects_non_rgb_multichannel_tiff(tmp_path):
    tifffile.imwrite(tmp_path / "band_001.tif", np.zeros((5, 6, 4), dtype=np.uint8))
    with pytest.raises(TiffFormatError):
        load_tiff_folder(tmp_path)


def test_band_subset_preserves_source_metadata(tmp_path):
    for wavelength in (365, 450, 650):
        _write(tmp_path / f"capture_{wavelength}nm.tif", wavelength)
    stack = load_tiff_folder(tmp_path, bands=[2, 0])
    assert stack.shape[0] == 2
    assert [band.source_index for band in stack.bands] == [2, 0]
    assert [band.wavelength_nm for band in stack.bands] == [650, 365]


def test_inspection_handles_numeric_prefixes_and_named_rgb_sources(tmp_path):
    from msi_core import inspect_tiff_folder

    for wavelength, name in ((365, "UV"), (475, "Blue"), (535, "Green"), (630, "Red")):
        _write(tmp_path / f"{wavelength} {name}.tif", wavelength, shape=(5, 6))

    rgb = np.zeros((5, 6, 3), dtype=np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    tifffile.imwrite(tmp_path / "VIS.tif", rgb, photometric="rgb")

    inspection = inspect_tiff_folder(tmp_path)

    assert [source.file for source in inspection.sources] == [
        "365 UV.tif", "475 Blue.tif", "535 Green.tif", "630 Red.tif", "VIS.tif"
    ]
    assert [source.wavelength_nm for source in inspection.sources[:4]] == [365, 475, 535, 630]
    assert inspection.sources[-1].detected_type == "rgb"
    assert {rep.id for rep in inspection.suggested_representations} == {
        "visible_rgb", "visible_spectral"
    }


def test_create_manifest_expands_rgb_and_builds_visible_representations(tmp_path):
    from msi_core import create_manifest, inspect_tiff_folder

    for wavelength, name in ((475, "Blue"), (535, "Green"), (630, "Red")):
        _write(tmp_path / f"{wavelength} {name}.tif", wavelength, shape=(5, 6))
    rgb = np.zeros((5, 6, 3), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "VIS.tif", rgb, photometric="rgb")

    manifest = create_manifest(inspect_tiff_folder(tmp_path))

    assert [source.type for source in manifest.sources] == [
        "single_band", "single_band", "single_band", "rgb"
    ]
    vis = next(source for source in manifest.sources if source.file == "VIS.tif")
    assert [band.channel for band in vis.bands] == ["R", "G", "B"]
    assert [band.id for band in vis.bands] == ["vis_r", "vis_g", "vis_b"]

    visible_rgb = next(rep for rep in manifest.representations if rep.id == "visible_rgb")
    assert (visible_rgb.red_band, visible_rgb.green_band, visible_rgb.blue_band) == (
        "vis_r", "vis_g", "vis_b"
    )

    visible_spectral = next(rep for rep in manifest.representations if rep.id == "visible_spectral")
    assert (visible_spectral.red_band, visible_spectral.green_band, visible_spectral.blue_band) == (
        "wl_630", "wl_535", "wl_475"
    )


def test_loader_accepts_in_memory_dataset_manifest(tmp_path):
    from msi_core import create_manifest, inspect_tiff_folder

    _write(tmp_path / "365 UV.tif", 365, shape=(5, 6))
    rgb = np.zeros((5, 6, 3), dtype=np.uint16)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 1, 2, 3
    tifffile.imwrite(tmp_path / "VIS.tif", rgb, photometric="rgb")

    manifest = create_manifest(inspect_tiff_folder(tmp_path))
    stack = load_tiff_folder(tmp_path, manifest=manifest)

    assert stack.shape == (4, 5, 6)
    assert stack.data[:, 0, 0].tolist() == [365, 1, 2, 3]
    assert [band.extra.get("band_id") for band in stack.bands] == [
        "wl_365", "vis_r", "vis_g", "vis_b"
    ]
    assert stack.metadata["manifest"] == "in_memory"


def test_v1_manifest_is_upgraded_for_backward_compatibility(tmp_path):
    _write(tmp_path / "red.tif", 8)
    _write(tmp_path / "uv.tif", 4)
    manifest = {
        "schema_version": "1.0",
        "bands": [
            {"index": 0, "file": "red.tif", "name": "Red", "wavelength_nm": 650},
            {"index": 1, "file": "uv.tif", "name": "UV", "wavelength_nm": 365},
        ],
    }
    (tmp_path / "bands.json").write_text(json.dumps(manifest), encoding="utf-8")
    stack = load_tiff_folder(tmp_path)
    assert stack.data[:, 0, 0].tolist() == [8, 4]
    assert [band.wavelength_nm for band in stack.bands] == [650, 365]


def test_float16_conversion_rejects_uint16_values_outside_finite_range(tmp_path):
    _write(tmp_path / "365nm.tif", 65535, shape=(4, 5))
    with pytest.raises(ValueError, match="float32"):
        load_tiff_folder(tmp_path, dtype="float16")


def test_float32_conversion_accepts_full_uint16_range(tmp_path):
    _write(tmp_path / "365nm.tif", 65535, shape=(4, 5))
    stack = load_tiff_folder(tmp_path, dtype="float32")
    assert stack.dtype == np.dtype("float32")
    assert np.isfinite(stack.data).all()
    assert stack.data.max() == 65535
