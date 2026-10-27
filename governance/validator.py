"""Configuration-only L1-L3 gates; no simulation or training execution.

This bounded gatekeeper checks schema, chronology, EMA parameters, and local
file identity. It does not certify the full Data & Feature Contract, market
calendar coverage, warm-up sufficiency, or trading performance.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

LOGGER = logging.getLogger(__name__)
Schema = Dict[str, Union[type, Tuple[type, ...], "Schema"]]


class ValidationState(str, Enum):
    """Readiness states, independent of research or deployment outcomes."""

    DRAFT_INCOMPLETE = "DRAFT_INCOMPLETE"
    INVALID = "INVALID"
    EVIDENCE_UNCHECKED = "EVIDENCE_UNCHECKED"
    READY_FOR_OFFLINE_EXECUTION = "READY_FOR_OFFLINE_EXECUTION"


class ConfigValidationError(ValueError):
    """A configuration or evidence constraint was violated."""


_FEE_SCHEMA: Schema = {
    "value": (int, float),
    "unit": str,
    "application_rule": str,
}
_SCHEMA: Schema = {
    "dataset": {
        "path": str,
        "sha256": str,
        "format": str,
        "source_description": str,
        "symbol": str,
        "timeframe": str,
        "timezone": str,
        "timestamp_meaning": str,
        "price_basis": str,
        "column_mapping": dict,
        "missing_bar_policy": str,
    },
    "strategy": {
        "name": str,
        "ema": {
            "fast_period": int,
            "slow_period": int,
            "initialization_rule": str,
        },
        "signal_rule": str,
        "position_between_crossovers": str,
        "shorting_allowed": bool,
    },
    "costs": {
        "spread": _FEE_SCHEMA,
        "commission": _FEE_SCHEMA,
        "slippage": _FEE_SCHEMA,
        "swap": {
            "long_value": (int, float),
            "short_value": (int, float),
            "unit": str,
            "rollover_timezone": str,
            "rollover_time": str,
            "day_count_and_multiplier_rule": str,
        },
    },
    "execution": {
        "decision_timing": str,
        "fill_timing": str,
        "fill_price_rule": str,
        "final_bar_entry_policy": str,
        "terminal_liquidation_price_rule": str,
        "terminal_liquidation_cost_rule": str,
    },
    "splits": {
        "boundary_convention": str,
        "train": {"start": str, "end": str},
        "validation": {"start": str, "end": str},
        "test": {"start": str, "end": str},
    },
    "frozen": bool,
}
# The six required roots use the existing template's complete nested fields.
# Optional template sections are type-checked, but not used to infer defaults.
_OPTIONAL_SCHEMA: Schema = {
    "template_version": int,
    "specification_status": str,
    "experiment_id": str,
    "milestone": str,
    "proposals_not_confirmed": dict,
    "instrument": dict,
    "policy": dict,
    "features": dict,
    "training": dict,
    "evaluation": dict,
    "safety": dict,
    "orchestration": dict,
    "deployment": dict,
    "optimization": dict,
}


class ConfigValidator:
    """Validate an explicit configuration without mutating it.

    Args:
        config: JSON-compatible configuration mapping. Relative dataset paths
            resolve against ``base_directory``.
        base_directory: Explicit path base; None uses the process working
            directory. No data or trading settings are defaulted.
        logger: Logger receiving exact failure reasons and draft diagnostics.

    Attributes:
        errors: Blocking reasons from the latest validation.
        unresolved_fields: Missing/null settings from the latest validation.
        evidence: Independent check statuses, never inferred from frozen.
        actual_dataset_sha256: Computed file digest, or None if not computed.
    """

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        base_directory: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.base_directory = (
            Path.cwd() if base_directory is None else Path(base_directory)
        )
        self.logger = logger if logger is not None else LOGGER
        self.errors: List[str] = []
        self.unresolved_fields: List[str] = []
        self.evidence: Dict[str, str] = {}
        self.actual_dataset_sha256: Optional[str] = None

    def validate(self, *, check_evidence: bool = True) -> ValidationState:
        """Run the L1-L3 checks, converting every exception to INVALID.

        Args:
            check_evidence: False explicitly defers physical file inspection.
                This can never yield READY_FOR_OFFLINE_EXECUTION.

        Returns:
            The derived readiness state. READY applies only to this module's
            bounded checks and is not full orchestrator execution authorization.
        """
        self.errors = []
        self.unresolved_fields = []
        self.evidence = {"dataset_readability": "UNCHECKED", "dataset_sha256": "UNCHECKED"}
        self.actual_dataset_sha256 = None
        try:
            if type(check_evidence) is not bool:
                raise ConfigValidationError("check_evidence must be a boolean")
            self.validate_layer1()
            self.validate_layer2()
            if check_evidence:
                dataset = self._section("dataset")
                if dataset.get("path") is not None and dataset.get("sha256") is not None:
                    self.validate_layer3(dataset["path"])
            if self.unresolved_fields or self.config["frozen"] is not True:
                self.logger.info(
                    "DRAFT_INCOMPLETE: unresolved=%s; frozen=%r",
                    self.unresolved_fields,
                    self.config["frozen"],
                )
                return ValidationState.DRAFT_INCOMPLETE
            if any(status != "PASS" for status in self.evidence.values()):
                return ValidationState.EVIDENCE_UNCHECKED
            return ValidationState.READY_FOR_OFFLINE_EXECUTION
        except Exception as exc:
            # This public boundary intentionally converts unexpected failures
            # as well as expected constraint violations into a logged refusal.
            reason = f"{type(exc).__name__}: {exc}"
            self.errors.append(reason)
            self.logger.exception("Configuration validation failed: %s", reason)
            return ValidationState.INVALID

    def validate_layer1(self) -> None:
        """Check required roots, nested structure, types, and frozen nulls.

        Raises:
            ConfigValidationError: A required root is absent or a supplied
                value is malformed. Missing nested settings are allowed only
                in an unfrozen draft.
        """
        if not isinstance(self.config, Mapping):
            raise ConfigValidationError("config must be a mapping")
        for key in _SCHEMA:
            if key not in self.config:
                raise ConfigValidationError(f"missing required configuration key: {key}")
        frozen = self.config["frozen"]
        if frozen is not None and type(frozen) is not bool:
            raise ConfigValidationError("frozen must be a boolean or draft null")
        unknown = set(self.config) - _SCHEMA.keys() - _OPTIONAL_SCHEMA.keys()
        if unknown:
            raise ConfigValidationError(f"unknown configuration keys: {sorted(unknown, key=str)}")
        self._check_object(self.config, _SCHEMA, "", frozen is True)
        for key, expected in _OPTIONAL_SCHEMA.items():
            if key in self.config:
                self._check_value(self.config[key], expected, key, frozen is True)
        self._check_json(self.config, "config", frozen is True)

    def validate_layer2(self) -> None:
        """Check half-open split ordering, explicit timestamps, and EMA bounds.

        Raises:
            ConfigValidationError: Chronology or a supplied EMA constraint fails.
        """
        splits = self._section("splits")
        convention = splits.get("boundary_convention")
        if convention is not None and convention != "start_inclusive_end_exclusive":
            raise ConfigValidationError("splits.boundary_convention must be start_inclusive_end_exclusive")
        boundaries: Dict[str, datetime] = {}
        for name in ("train", "validation", "test"):
            partition = splits.get(name)
            if partition is None:
                continue
            if not isinstance(partition, Mapping):
                raise ConfigValidationError(f"splits.{name} must be an object")
            for edge in ("start", "end"):
                value = partition.get(edge)
                if value is not None:
                    boundaries[f"{name}.{edge}"] = self._timestamp(value, f"splits.{name}.{edge}")
        for left, right, strict in (
            ("train.start", "train.end", True),
            ("validation.start", "validation.end", True),
            ("test.start", "test.end", True),
            ("train.end", "validation.start", False),
            ("validation.end", "test.start", False),
            ("train.end", "test.start", False),
        ):
            if left in boundaries and right in boundaries:
                if boundaries[left] > boundaries[right] or (
                    strict and boundaries[left] == boundaries[right]
                ):
                    operator = "<" if strict else "<="
                    raise ConfigValidationError(f"splits.{left} must be {operator} splits.{right}")
        strategy = self._section("strategy")
        ema = strategy.get("ema")
        if isinstance(ema, Mapping):
            for name in ("fast_period", "slow_period"):
                value = ema.get(name)
                if value is not None and (type(value) is not int or value <= 0):
                    raise ConfigValidationError(f"strategy.ema.{name} must be a positive integer")
            fast, slow = ema.get("fast_period"), ema.get("slow_period")
            if isinstance(fast, int) and isinstance(slow, int) and fast >= slow:
                raise ConfigValidationError("strategy.ema.fast_period must be < strategy.ema.slow_period")
        dataset = self._section("dataset")
        declared_hash = dataset.get("sha256")
        if declared_hash is not None and (
            not isinstance(declared_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_hash) is None
        ):
            raise ConfigValidationError("dataset.sha256 must be 64 lowercase hexadecimal characters")

    def validate_layer3(self, dataset_path: object) -> None:
        """Verify a declared physical file's readability and raw-byte SHA-256.

        Args:
            dataset_path: Exact dataset.path value from the configuration.

        Raises:
            ConfigValidationError: File identity differs, is not a local file,
                or the file changed during hashing.
            OSError: Opening or reading the file fails.
        """
        dataset = self._section("dataset")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ConfigValidationError("dataset.path must be a nonempty string")
        if dataset_path != dataset.get("path"):
            raise ConfigValidationError("dataset_path differs from configured dataset.path")
        if dataset_path.startswith(("\\\\", "//")) or "://" in dataset_path:
            raise ConfigValidationError("dataset.path must be local, not a network path or URL")
        path = Path(dataset_path)
        if not path.is_absolute():
            path = self.base_directory / path
        self.evidence["dataset_readability"] = "FAIL"
        if not path.is_file():
            raise ConfigValidationError(f"dataset.path is not a readable regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ConfigValidationError(f"dataset changed during hashing: {path}")
        current = path.stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise ConfigValidationError(f"dataset path changed during hashing: {path}")
        self.evidence["dataset_readability"] = "PASS"
        self.actual_dataset_sha256 = digest.hexdigest()
        self.evidence["dataset_sha256"] = "FAIL"
        if self.actual_dataset_sha256 != dataset.get("sha256"):
            raise ConfigValidationError(
                f"dataset.sha256 mismatch: expected {dataset.get('sha256')!r}, "
                f"actual {self.actual_dataset_sha256}"
            )
        self.evidence["dataset_sha256"] = "PASS"

    def _section(self, name: str) -> Mapping[str, object]:
        value = self.config.get(name)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigValidationError(f"{name} must be an object")
        return value

    def _check_object(
        self, value: Mapping[str, object], schema: Schema, path: str, frozen: bool
    ) -> None:
        if path:
            unknown = set(value) - schema.keys()
            if unknown:
                raise ConfigValidationError(f"{path}: unknown keys {sorted(unknown, key=str)}")
        for name, expected in schema.items():
            field = f"{path}.{name}" if path else name
            self._check_value(value.get(name), expected, field, frozen)

    def _check_value(
        self,
        value: object,
        expected: Union[type, Tuple[type, ...], Schema],
        path: str,
        frozen: bool,
    ) -> None:
        if value is None:
            if frozen:
                raise ConfigValidationError(f"{path} is required and cannot be null when frozen")
            self.unresolved_fields.append(path)
        elif isinstance(expected, dict):
            if not isinstance(value, Mapping):
                raise ConfigValidationError(f"{path} must be an object")
            self._check_object(value, expected, path, frozen)
        else:
            types = expected if isinstance(expected, tuple) else (expected,)
            if type(value) not in types:
                raise ConfigValidationError(f"{path} has invalid type {type(value).__name__}")
            if isinstance(value, str) and not value.strip():
                raise ConfigValidationError(f"{path} cannot be an empty string")

    def _check_json(self, value: object, path: str, frozen: bool) -> None:
        if value is None:
            if frozen:
                raise ConfigValidationError(f"{path} cannot be null when frozen")
        elif isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ConfigValidationError(f"{path} keys must be strings")
                self._check_json(child, f"{path}.{key}", frozen)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self._check_json(child, f"{path}[{index}]", frozen)
        elif type(value) not in (str, int, float, bool):
            raise ConfigValidationError(f"{path} is not JSON-compatible")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ConfigValidationError(f"{path} must be finite")

    @staticmethod
    def _timestamp(value: object, path: str) -> datetime:
        if not isinstance(value, str):
            raise ConfigValidationError(f"{path} must be an ISO-8601 string")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigValidationError(f"{path} is not a valid ISO-8601 timestamp: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ConfigValidationError(f"{path} must include an explicit UTC offset")
        return parsed
