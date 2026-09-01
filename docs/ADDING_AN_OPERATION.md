# Adding an operation

Create one file under `msi_core/operations/`.

```python
from typing import Any

from pydantic import BaseModel, Field

from msi_core.attribution import AlgorithmAttribution, OperationInfo, Person
from msi_core.parameters import resolve_params
from msi_core.result import Result
from msi_core.stack import Stack


INFO = OperationInfo(
    operation_id="my_method",
    operation_version="1.0.0",
    display_name="My Method",
    description="Describe what the method calculates.",
    category="enhancement",
    input_kind="stack",
    attribution=AlgorithmAttribution(
        method_name="My Method",
        authors=[Person(name="Researcher Name")],
        status="experimental",
    ),
)


class Params(BaseModel):
    strength: float = Field(
        default=1.0,
        gt=0,
        description="Multiplicative strength applied to every pixel.",
    )


def run(
    stack: Stack,
    params: Params | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    config = resolve_params(Params, params, kwargs)

    output_data = stack.data.astype("float32", copy=True) * config.strength
    output = stack.with_data(output_data)

    return Result(
        output=output,
        operation_id=INFO.operation_id,
        operation_version=INFO.operation_version,
        operation_info=INFO,
        parameters=config.model_dump(mode="json"),
    )
```

Then add the module explicitly to `BUILTIN_OPERATION_MODULES` in
`msi_core/operations/__init__.py`.

## Requirements

- Keep numerical computation inside `run`.
- Do not write files or access a database from the operation.
- Do not mutate the input stack.
- Give the operation a stable ID and semantic version.
- Treat `Params` as the single source of truth for parameter types, defaults, constraints, and descriptions.
- Give every public `Params` field a non-empty Pydantic `description`.
- Keep `run()` docstrings focused on the method, inputs/outputs, scientific notes, examples, and references; do not duplicate every parameter description there.
- Document the input/output interpretation and numerical range.
- Add tests using small synthetic arrays.
- Use `AlgorithmAttribution` only for scientific-method authorship; implementation
  contributors belong in version control and contributor records.
