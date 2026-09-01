import json

import numpy as np
import tifffile

from types import ModuleType

from pydantic import BaseModel

from msi_core import (
    AlgorithmAttribution,
    OperationInfo,
    OperationRegistry,
    Person,
    Result,
    create_default_registry,
    registry,
    save_result,
)
from msi_core.operations import pca


def test_registry_exposes_schema_and_information():
    description = registry.describe("pca")
    assert description["operation_id"] == "pca"
    assert "n_components" in description["parameter_schema"]["properties"]
    assert description["input_kind"] == "stack"


def test_registry_lists_operations_in_stable_order():
    identifiers = [item.info.operation_id for item in registry.list_available()]
    assert identifiers == sorted(identifiers)
    assert "normalize" in identifiers
    assert "subtraction" in identifiers


def test_save_result_writes_scientific_tiffs_and_png_renders(tmp_path, small_stack):
    result = pca.run(small_stack, n_components=2)
    output = save_result(result, tmp_path / "export")
    tiffs = sorted((output / "data").glob("*.tif"))
    renders = sorted((output / "renders").glob("*.png"))
    assert len(tiffs) == 2
    assert len(renders) == 2
    image = tifffile.imread(tiffs[0])
    assert np.issubdtype(image.dtype, np.floating)
    from PIL import Image
    rendered = np.asarray(Image.open(renders[0]))
    assert rendered.dtype == np.uint8
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["operation_id"] == "pca"
    assert metadata["export"]["scientific_values_preserved"] is True
    assert metadata["export"]["renders"]["bit_depth"] == 8
    mapping = metadata["export"]["renders"]["per_image_mapping"][0]
    assert mapping["method"] == "minmax"
    assert mapping["source_min"] is not None
    assert mapping["source_max"] is not None
    assert mapping["output_min"] == 0
    assert mapping["output_max"] == 255


def test_save_result_can_disable_renders(tmp_path, small_stack):
    result = pca.run(small_stack, n_components=1)
    output = save_result(result, tmp_path / "raw", save_render=False)
    assert (output / "data").is_dir()
    assert not (output / "renders").exists()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["export"]["renders"]["enabled"] is False


def test_save_result_can_write_jpeg_renders(tmp_path, small_stack):
    result = pca.run(small_stack, n_components=1)
    output = save_result(result, tmp_path / "jpeg", render_format="jpeg", jpeg_quality=90)
    assert len(list((output / "renders").glob("*.jpg"))) == 1


def test_default_registry_factory_returns_isolated_registry():
    first = create_default_registry()
    second = create_default_registry()
    assert first is not second
    assert [item.info.operation_id for item in first.list_available()] == [
        item.info.operation_id for item in second.list_available()
    ]


def test_registry_preserves_scientific_method_attribution(small_stack):
    module = ModuleType("author_method")

    class Params(BaseModel):
        gain: float = 1.0

    def run(stack, params=None, **kwargs):
        return Result(
            output=stack,
            operation_id="author_method",
            operation_version="1.0.0",
            parameters={"gain": 1.0},
        )

    module.INFO = OperationInfo(
        operation_id="author_method",
        operation_version="1.0.0",
        display_name="Author Method",
        description="An attributed scientific method.",
        category="enhancement",
        input_kind="stack",
        attribution=AlgorithmAttribution(
            method_name="Author Method",
            authors=[Person(name="Jane Doe", affiliation="Example University")],
            status="experimental",
        ),
    )
    module.Params = Params
    module.run = run

    local_registry = OperationRegistry()
    local_registry.register_module(module)
    description = local_registry.describe("author_method")
    assert description["attribution"]["authors"][0]["name"] == "Jane Doe"


def test_saved_metadata_includes_scientific_attribution(tmp_path, small_stack):
    info = OperationInfo(
        operation_id="author_method",
        operation_version="1.0.0",
        display_name="Author Method",
        description="An attributed scientific method.",
        category="enhancement",
        input_kind="stack",
        attribution=AlgorithmAttribution(
            method_name="Author Method",
            authors=[Person(name="Jane Doe", affiliation="Example University")],
            status="experimental",
        ),
    )
    result = Result(
        output=small_stack,
        operation_id=info.operation_id,
        operation_version=info.operation_version,
        operation_info=info,
        parameters={},
    )
    output = save_result(result, tmp_path / "attributed")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["operation_info"]["attribution"]["authors"][0]["name"] == "Jane Doe"
