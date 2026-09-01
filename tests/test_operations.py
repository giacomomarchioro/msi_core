import numpy as np
import pytest
from pydantic import ValidationError

from msi_core import apply_pairwise
from msi_core.errors import ParameterConflictError
from msi_core.operations import (
    denoise,
    division,
    hot_pixel,
    ica,
    kmeans,
    log_ratio,
    normalize,
    pca,
    rotate,
    subtraction,
)


def test_normalize_kwargs_and_params_are_equivalent(small_stack):
    first = normalize.run(small_stack, low_percentile=0, high_percentile=100)
    params = normalize.Params(low_percentile=0, high_percentile=100)
    second = normalize.run(small_stack, params=params)
    assert np.allclose(first.output.data, second.output.data)
    assert first.output.data.min() == 0
    assert first.output.data.max() == 1


def test_params_and_kwargs_cannot_be_mixed(small_stack):
    with pytest.raises(ParameterConflictError):
        normalize.run(small_stack, params=normalize.Params(), mode="global")


def test_invalid_percentile_order_is_rejected():
    with pytest.raises(ValidationError):
        normalize.Params(low_percentile=90, high_percentile=10)


def test_spectral_pca_returns_score_images(small_stack):
    result = pca.run(small_stack, n_components=2, mode="spectral")
    assert result.output.shape == (2, 6, 8)
    assert result.metadata["scientific_convention"] == "pixels_as_samples_bands_as_features"
    assert len(result.additional_data["spectral_loadings"][0]) == small_stack.n_bands


def test_image_space_pca_returns_component_images(small_stack):
    result = pca.run(small_stack, n_components=2, mode="image_space")
    assert result.output.shape == (2, 6, 8)
    assert result.metadata["scientific_convention"] == "bands_as_samples_pixels_as_features"
    assert "band_scores" in result.additional_data


def test_ica_returns_requested_components(small_stack):
    result = ica.run(small_stack, n_components=2, random_state=0, max_iter=1000)
    assert result.output.shape == (2, 6, 8)
    assert len(result.additional_data["components"]) == 2


def test_kmeans_returns_label_image(small_stack):
    result = kmeans.run(small_stack, n_clusters=3, random_state=0, batch_size=32)
    assert result.output.shape == (1, 6, 8)
    assert result.output.data.dtype == np.int32
    assert set(np.unique(result.output.data)).issubset({0, 1, 2})


def test_pairwise_arithmetic(small_stack):
    sub = subtraction.run(small_stack[1], small_stack[0])
    div = division.run(small_stack[1], small_stack[0], epsilon=1.0)
    log = log_ratio.run(small_stack[1], small_stack[0], offset=1.0)
    assert sub.output.shape == (1, 6, 8)
    assert np.isfinite(div.output.data).all()
    assert np.isfinite(log.output.data).all()


def test_apply_pairwise_counts_combinations_and_ordered_pairs(small_stack):
    combinations = apply_pairwise(subtraction.run, small_stack)
    ordered = apply_pairwise(subtraction.run, small_stack, ordered=True)
    assert len(combinations) == 3
    assert len(ordered) == 6


def test_rotate_and_denoise(small_stack):
    rotated = rotate.run(small_stack, quarter_turns=1)
    assert rotated.output.shape == (3, 8, 6)
    filtered = denoise.run(small_stack, size=3)
    assert filtered.output.shape == small_stack.shape


def test_hot_pixel_replaces_isolated_outlier(small_stack):
    modified = small_stack.data.copy()
    modified[0, 3, 4] = 65535
    stack = small_stack.with_data(modified)
    result = hot_pixel.run(stack, window_size=3, threshold_sigma=3, minimum_deviation=100)
    assert result.output.data[0, 3, 4] != 65535
    assert result.additional_data["replaced_pixels_per_band"][0] >= 1


def test_pca_standardization_large_image_space_feature_path():
    from msi_core import BandMetadata, Stack

    rng = np.random.default_rng(42)
    data = rng.integers(0, 65536, size=(8, 40, 40), dtype=np.uint16).astype(np.float32)
    stack = Stack(
        data=data,
        bands=[BandMetadata(index=i, name=f"B{i}") for i in range(8)],
    )
    result = pca.run(stack, n_components=4, mode="image_space", standardize=True)
    assert result.output.shape == (4, 40, 40)
    assert np.isfinite(result.output.data).all()
    assert result.additional_data["standardization"] == {
        "enabled": True,
        "vectors_stored": False,
    }


def test_pca_handles_full_uint16_range_when_loaded_as_float32():
    from msi_core import BandMetadata, Stack

    data = np.array(
        [
            [[0, 65535], [65534, 1]],
            [[65535, 0], [2, 65533]],
            [[100, 200], [300, 400]],
        ],
        dtype=np.uint16,
    ).astype(np.float32)
    stack = Stack(data=data, bands=[BandMetadata(index=i) for i in range(3)])
    result = pca.run(stack, n_components=2, mode="spectral")
    assert np.isfinite(result.output.data).all()
