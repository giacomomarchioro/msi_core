"""Template showing scientific-method attribution for a contributed algorithm."""

from typing import Any

from pydantic import BaseModel, Field

from msi_core import AlgorithmAttribution, OperationInfo, Person, Result, Stack
from msi_core.parameters import resolve_params

INFO = OperationInfo(
    operation_id="example_author_method",
    operation_version="1.0.0",
    display_name="Example Author Method",
    description="A minimal example of an author-attributed operation.",
    category="enhancement",
    input_kind="stack",
    attribution=AlgorithmAttribution(
        method_name="Example Author Method",
        authors=[
            Person(
                name="Jane Doe",
                affiliation="Example University",
                orcid="0000-0000-0000-0000",
            )
        ],
        status="experimental",
        notes="Replace this example with the real method reference.",
    ),
)


class Params(BaseModel):
    gain: float = Field(default=1.0, gt=0)


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    config = resolve_params(Params, params, kwargs)
    output = stack.with_data(stack.data.astype("float32") * config.gain)
    return Result(
        output=output,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
    )
