"""CPU ONNX export with an adjacent feature-schema contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def export_policy(
    policy,
    output_path: str | Path,
    *,
    feature_names: Sequence[str],
    opset_version: int = 17,
) -> tuple[Path, Path]:
    """Export a Torch module and record the ordered input/output tensor contract."""
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or not all(names):
        raise ValueError("feature_names must be a non-empty sequence of unique names")
    if opset_version < 11:
        raise ValueError("opset_version must be at least 11")
    try:
        import torch
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX export requires the SB3 extra and `pip install onnx`") from error
    if not isinstance(policy, torch.nn.Module):
        raise TypeError("policy must be a torch.nn.Module")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema_path = destination.with_suffix(destination.suffix + ".schema.json")
    sample = torch.zeros((1, len(names)), dtype=torch.float32, device="cpu")
    policy = policy.to("cpu").eval()
    torch.onnx.export(
        policy,
        sample,
        destination,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=opset_version,
    )
    model = onnx.load(str(destination))
    onnx.checker.check_model(model)
    inputs = model.graph.input
    if len(inputs) != 1 or inputs[0].type.tensor_type.shape.dim[-1].dim_value != len(names):
        destination.unlink(missing_ok=True)
        raise ValueError("exported ONNX input does not match the feature contract")
    schema = {
        "version": 1,
        "input_name": "observations",
        "output_name": "actions",
        "dtype": "float32",
        "feature_names": list(names),
        "feature_count": len(names),
        "opset_version": opset_version,
    }
    schema_path.write_text(
        json.dumps(schema, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination, schema_path
