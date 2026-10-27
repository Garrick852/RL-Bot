"""Deterministic, marker-restricted MQL5 parameter patching and compilation."""

from __future__ import annotations

import os
import math
import re
import subprocess
import uuid
import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CompileResult:
    source: Path
    log_path: Path
    errors: int
    warnings: int

    @property
    def passed(self) -> bool:
        return self.errors == 0 and self.warnings == 0


def patch_source_text(source: str, parameters: Mapping[str, object]) -> str:
    """Replace exactly one declaration per supplied key carrying its OPTIMIZED tag."""
    patched = source
    for name, value in parameters.items():
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValueError(f"invalid MQL5 parameter name: {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"unsupported value for {name}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite source value for {name}")
        replacement = str(value)
        if isinstance(value, str) and not re.fullmatch(
            r"[A-Za-z_]\w*|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
            replacement,
        ):
            raise ValueError(f"unsafe source value for {name}")
        pattern = re.compile(
            rf"(?P<prefix>\binput\s+\w+\s+{re.escape(name)}\s*=\s*)"
            rf"(?P<value>[^;\r\n]+)"
            rf"(?P<suffix>;\s*//\s*OPTIMIZED:{re.escape(name)}\b)"
        )
        matches = list(pattern.finditer(patched))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one OPTIMIZED:{name} declaration; found {len(matches)}"
            )
        patched = pattern.sub(
            lambda match: match.group("prefix") + replacement + match.group("suffix"),
            patched,
            count=1,
        )
    return patched


def _parse_compile_log(text: str) -> tuple[int, int, bool]:
    summary = re.search(r"(\d+)\s+errors?\s*,\s*(\d+)\s+warnings?", text, re.IGNORECASE)
    if summary:
        return int(summary.group(1)), int(summary.group(2)), True
    errors = len(re.findall(r"\berror\s*:", text, re.IGNORECASE))
    warnings = len(re.findall(r"\bwarning\s*:", text, re.IGNORECASE))
    return errors, warnings, False


def _decode_source(content: bytes) -> tuple[str, str]:
    if content.startswith(codecs.BOM_UTF16_LE):
        return content.decode("utf-16"), "utf-16"
    if content.startswith(codecs.BOM_UTF16_BE):
        return content.decode("utf-16"), "utf-16"
    if content.startswith(codecs.BOM_UTF8):
        return content.decode("utf-8-sig"), "utf-8-sig"
    try:
        return content.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return content.decode("cp1252"), "cp1252"


def _decode_log(content: bytes) -> str:
    if content.startswith(codecs.BOM_UTF16_LE) or content.startswith(codecs.BOM_UTF16_BE):
        return content.decode("utf-16", errors="replace")
    if content.startswith(codecs.BOM_UTF8):
        return content.decode("utf-8-sig", errors="replace")
    return content.decode("utf-8", errors="replace")


def patch_and_compile(
    source_path: str | Path,
    parameters: Mapping[str, object],
    *,
    metaeditor: str | Path,
    timeout_seconds: int = 120,
) -> CompileResult:
    """Compile a staged source copy first; replace the original only on a clean log."""
    source = Path(source_path).resolve(strict=True)
    compiler = Path(metaeditor).resolve(strict=True)
    if source.suffix.lower() != ".mq5":
        raise ValueError("source_path must identify an .mq5 file")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    original_bytes = source.read_bytes()
    original, encoding = _decode_source(original_bytes)
    patched = patch_source_text(original, parameters)
    staged = source.with_name(f"{source.stem}.{uuid.uuid4().hex}.staged{source.suffix}")
    log_path = source.with_name(f"{source.stem}.{uuid.uuid4().hex}.compile.log")
    staged.write_bytes(patched.encode(encoding))
    staged_binary = staged.with_suffix(".ex5")
    deployed_binary = source.with_suffix(".ex5")
    backup_binary = source.with_name(f"{source.stem}.{uuid.uuid4().hex}.backup.ex5")
    try:
        process = subprocess.run(
            [
                str(compiler),
                f"/compile:{staged}",
                f"/log:{log_path}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        log = _decode_log(log_path.read_bytes()) if log_path.exists() else ""
        log += process.stdout + "\n" + process.stderr
        errors, warnings, has_summary = _parse_compile_log(log)
        if process.returncode != 0 or not log_path.exists() or not has_summary:
            errors = max(errors, 1)
        result = CompileResult(source, log_path, errors, warnings)
        if not result.passed:
            raise RuntimeError(
                f"MT5 compile failed (return code {process.returncode}, "
                f"{errors} errors, {warnings} warnings); see {log_path}"
            )
        if not staged_binary.is_file():
            raise RuntimeError(f"MT5 compile did not produce a binary: {staged_binary}")
        if source.read_bytes() != original_bytes:
            raise RuntimeError(f"MQL5 source changed while compilation was running: {source}")
        had_deployed_binary = deployed_binary.exists()
        if had_deployed_binary:
            backup_binary.write_bytes(deployed_binary.read_bytes())
        try:
            os.replace(staged_binary, deployed_binary)
            os.replace(staged, source)
        except OSError as deployment_error:
            try:
                if had_deployed_binary and backup_binary.exists():
                    os.replace(backup_binary, deployed_binary)
                elif not had_deployed_binary:
                    deployed_binary.unlink(missing_ok=True)
            except OSError as rollback_error:
                raise RuntimeError(
                    f"MT5 deployment and rollback failed; preserve recovery binary at {backup_binary}"
                ) from rollback_error
            raise deployment_error
        backup_binary.unlink(missing_ok=True)
        return result
    finally:
        staged.unlink(missing_ok=True)
        staged_binary.unlink(missing_ok=True)
