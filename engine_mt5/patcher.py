"""Marker-restricted MQL5 edits and fail-closed local compilation."""

from __future__ import annotations

import codecs
import math
import os
import re
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SourcePatchError(ValueError):
    """A requested parameter cannot be safely and uniquely patched."""


class CompilationError(RuntimeError):
    """Compilation evidence is missing, dirty, or inconsistent."""


def _decode(content: bytes) -> tuple[str, str, bytes]:
    """Preserve the exact source BOM and endianness; never guess legacy encodings."""
    for bom, encoding in (
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
    ):
        if content.startswith(bom):
            return content[len(bom):].decode(encoding), encoding, bom
    return content.decode("utf-8"), "utf-8", b""


def _code_locations(source: str) -> tuple[str, set[int]]:
    """Mask comments/strings while retaining offsets and real line-comment starts."""
    pattern = re.compile(r'//[^\r\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
    masked = list(source)
    comments: set[int] = set()
    for match in pattern.finditer(source):
        if match.group().startswith("//"):
            comments.add(match.start())
        for index in range(match.start(), match.end()):
            if source[index] not in "\r\n":
                masked[index] = " "
    return "".join(masked), comments


def _literal(value: Any, mql_type: str, name: str) -> str:
    if mql_type == "bool":
        if type(value) is not bool:
            raise SourcePatchError(f"{name} requires a boolean")
        return "true" if value else "false"
    integer_ranges = {
        "char": (-2**7, 2**7 - 1), "uchar": (0, 2**8 - 1),
        "short": (-2**15, 2**15 - 1), "ushort": (0, 2**16 - 1),
        "int": (-2**31, 2**31 - 1), "uint": (0, 2**32 - 1),
        "long": (-2**63, 2**63 - 1), "ulong": (0, 2**64 - 1),
    }
    if mql_type in integer_ranges:
        low, high = integer_ranges[mql_type]
        if type(value) is not int or not low <= value <= high:
            raise SourcePatchError(f"{name} requires an in-range MQL5 {mql_type}")
        return str(value)
    if mql_type in ("double", "float"):
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid or (mql_type == "float" and abs(value) > 3.4028234663852886e38):
            raise SourcePatchError(f"{name} requires a finite in-range {mql_type}")
        return str(value)
    # Restrict enum values to identifiers; arbitrary expressions/source are forbidden.
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        if mql_type not in ("string", "datetime", "color"):
            return value
    raise SourcePatchError(f"unsupported or unsafe value/type for {name}: {mql_type}")


_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"


def _resolve_dimension(token: str, source: str, mask: str, name: str) -> int:
    """Resolve a literal or a uniquely ``#define``d integer array dimension."""
    if token.isdigit():
        size = int(token)
    else:
        pattern = re.compile(rf"^[ \t]*#define[ \t]+{re.escape(token)}[ \t]+(\d+)[ \t]*(?://[^\r\n]*)?\r?$",
                             re.MULTILINE)
        found = [m for m in pattern.finditer(source) if mask[m.start():m.end(1)] == source[m.start():m.end(1)]]
        if len(found) != 1:
            raise SourcePatchError(f"{name} dimension {token} must have exactly one integer #define")
        size = int(found[0].group(1))
    if size < 1:
        raise SourcePatchError(f"{name} dimension {token} must be positive")
    return size


def _matrix_literal(value: Any, rows: int, cols: int | None, mql_type: str, name: str,
                    indent: str, newline: str) -> str:
    """Render an exact rows x cols (or rows-vector) finite numeric initializer.

    Ragged, flattened/nested-mismatched, bool, and non-numeric input is refused.
    """
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise SourcePatchError(f"{name} requires exactly {rows} {'rows' if cols else 'values'}")
    if cols is None:
        if any(isinstance(item, (list, tuple)) for item in value):
            raise SourcePatchError(f"{name} is one-dimensional; nested values are refused")
        items = ", ".join(_literal(item, mql_type, f"{name}[{index}]") for index, item in enumerate(value))
        return f"{indent}{{{items}}}"
    rendered_rows = []
    for index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise SourcePatchError(f"{name} row {index} requires exactly {cols} values")
        rendered_rows.append(", ".join(_literal(item, mql_type, f"{name}[{index}]") for item in row))
    inner = indent + "   "
    body = ("," + newline).join(f"{inner}{{{row}}}" for row in rendered_rows)
    return f"{indent}{{{newline}{body}{newline}{indent}}}"


class MQL5Patcher:
    """Patch explicit tagged parameters and verify a fresh clean local build.

    Args:
        timeout_seconds: Strict positive finite MetaEditor process timeout.
            Defaults to 30 seconds.

    Notes:
        Source edits are atomic after every requested patch is validated, but
        patch_source does not compile or roll back a successful source edit.
        Work on a staged EA copy for combined patch/compile workflows.
        Successful compilation is not OOS validation or deployment permission.
    """

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)

    def patch_source(
        self, file_path: str | Path, active_config: Mapping[str, Any]
    ) -> Path:
        """Atomically replace defaults at exactly one real anchor per parameter.

        Args:
            file_path: Existing local .mq5 source, UTF-8 or BOM-marked UTF-16.
            active_config: Nonempty parameter mapping, or an active-config
                envelope containing a nonempty ``parameters`` mapping.
                Scalars patch ``input``/``sinput`` lines tagged
                ``// OPTIMIZED:<Name>``. A list-of-lists patches the brace
                initializer under ``const double|float <Name>[R][C] =
                // OPTIMIZED_MATRIX:<Name>``; a flat list patches a 1-D
                ``<Name>[N]`` declaration. Dimensions are integer literals or
                uniquely ``#define``d integers and the shape must match exactly.
                This method does not certify that governance approved it.

        Returns:
            Patched source path; encoding, BOM, and line endings are preserved.

        Raises:
            SourcePatchError: Source, types, anchors, I/O, or concurrent source
                modifications prevent a safe all-or-nothing edit.
        """
        temporary: str | None = None
        try:
            path = Path(file_path)
            if path.suffix.lower() != ".mq5" or not path.is_file():
                raise SourcePatchError("file_path must be an existing .mq5 file")
            if not isinstance(active_config, Mapping):
                raise SourcePatchError("active_config must be a parameter mapping")
            parameters = active_config.get("parameters", active_config)
            if not isinstance(parameters, Mapping) or not parameters:
                raise SourcePatchError("parameters must be a nonempty mapping")
            original = path.read_bytes()
            source, encoding, bom = _decode(original)
            mask, comments = _code_locations(source)
            edits: list[tuple[int, int, str]] = []
            for name, value in parameters.items():
                if not isinstance(name, str) or not re.fullmatch(_IDENTIFIER, name):
                    raise SourcePatchError(f"invalid parameter name: {name!r}")
                if isinstance(value, (list, tuple)):
                    edits.append(self._matrix_edit(source, mask, comments, name, value))
                    continue
                pattern = re.compile(
                    rf"^[ \t]*(?P<declaration>input|sinput)[ \t]+"
                    rf"(?P<type>{_IDENTIFIER})[ \t]+{re.escape(name)}"
                    rf"[ \t]*=[ \t]*(?P<value>[^;\r\n]+?)[ \t]*;"
                    rf"[ \t]*(?P<anchor>//)[ \t]*OPTIMIZED:{re.escape(name)}"
                    rf"(?![A-Za-z0-9_])[^\r\n]*\r?$", re.MULTILINE,
                )
                matches = [
                    match for match in pattern.finditer(source)
                    if mask[match.start("declaration"):match.end("declaration")] == match["declaration"]
                    and match.start("anchor") in comments
                ]
                if len(matches) != 1:
                    raise SourcePatchError(f"expected one OPTIMIZED:{name} declaration, found {len(matches)}")
                match = matches[0]
                edits.append((match.start("value"), match.end("value"), _literal(value, match["type"], name)))
            patched = source
            for start, end, value in sorted(edits, key=lambda edit: edit[0], reverse=True):
                patched = patched[:start] + value + patched[end:]
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(bom + patched.encode(encoding))
                stream.flush()
                os.fsync(stream.fileno())
            if path.read_bytes() != original:
                raise SourcePatchError("source changed during patching; edit refused")
            os.replace(temporary, path)
            temporary = None
            return path
        except (OSError, UnicodeError) as exc:
            raise SourcePatchError(f"cannot patch source: {exc}") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _matrix_edit(
        source: str, mask: str, comments: set[int], name: str, value: Any
    ) -> tuple[int, int, str]:
        """Locate exactly one anchored const matrix and render its replacement."""
        pattern = re.compile(
            rf"^[ \t]*(?P<declaration>(?:static[ \t]+)?const)[ \t]+(?P<type>double|float)[ \t]+"
            rf"{re.escape(name)}[ \t]*\[[ \t]*(?P<rows>{_IDENTIFIER}|\d+)[ \t]*\]"
            rf"(?:[ \t]*\[[ \t]*(?P<cols>{_IDENTIFIER}|\d+)[ \t]*\])?[ \t]*=[ \t]*"
            rf"(?P<anchor>//)[ \t]*OPTIMIZED_MATRIX:{re.escape(name)}(?![A-Za-z0-9_])[^\r\n]*"
            rf"(?P<newline>\r?\n)(?P<value>(?P<indent>[ \t]*)\{{[^;]*?\}})[ \t]*;", re.MULTILINE,
        )
        matches = [
            match for match in pattern.finditer(source)
            if mask[match.start("declaration"):match.end("declaration")] == match["declaration"]
            and match.start("anchor") in comments
        ]
        if len(matches) != 1:
            raise SourcePatchError(f"expected one OPTIMIZED_MATRIX:{name} declaration, found {len(matches)}")
        match = matches[0]
        body = source[match.start("value"):match.end("value")]
        # The replaced initializer may contain only numbers/braces/commas: no comments or code.
        if mask[match.start("value"):match.end("value")] != body or not re.fullmatch(r"[\s{},0-9eE+\-.]+", body):
            raise SourcePatchError(f"OPTIMIZED_MATRIX:{name} initializer must be a plain numeric literal")
        rows = _resolve_dimension(match["rows"], source, mask, name)
        cols = None if match["cols"] is None else _resolve_dimension(match["cols"], source, mask, name)
        rendered = _matrix_literal(value, rows, cols, match["type"], name, match["indent"], match["newline"])
        return match.start("value"), match.end("value"), rendered

    def compile_ea(self, metaeditor_path: str | Path, source_path: str | Path) -> Path:
        """Compile with /compile and /log, requiring fresh binary and clean log.

        Args:
            metaeditor_path: Existing local metaeditor64.exe executable.
            source_path: Existing local .mq5 file; its .ex5 is removed before
                compiling so an old binary cannot masquerade as a new build.

        Returns:
            Fresh unique compile-log path, retained for audit.

        Raises:
            CompilationError: Timeout, process failure, dirty/missing log,
                missing/empty binary, source change, or cleanup failure.
        """
        source = Path(source_path).absolute()
        compiler = Path(metaeditor_path).absolute()
        if source.suffix.lower() != ".mq5" or not source.is_file():
            raise CompilationError("source_path must be an existing .mq5 file")
        binary = source.with_suffix(".ex5")
        log_path = source.with_name(f"{source.stem}.{uuid.uuid4().hex}.compile.log")
        try:
            if compiler.name.lower() != "metaeditor64.exe" or not compiler.is_file():
                raise CompilationError("metaeditor_path must identify an existing metaeditor64.exe")
            original = source.read_bytes()
            binary.unlink(missing_ok=True)
            process = subprocess.run(
                [str(compiler), f"/compile:{source}", f"/log:{log_path}"],
                capture_output=True, timeout=self.timeout_seconds, check=False,
            )
            if process.returncode != 0:
                raise CompilationError(f"MetaEditor exited with code {process.returncode}; see {log_path}")
            self.verify_compile_log(log_path, source_path=source)
            if not binary.is_file() or binary.stat().st_size == 0:
                raise CompilationError("MetaEditor produced no nonempty .ex5 binary")
            if source.read_bytes() != original:
                raise CompilationError("source changed during compilation")
            return log_path
        except (OSError, subprocess.SubprocessError, CompilationError) as exc:
            self._remove_binary(binary)
            raise CompilationError(f"compilation refused: {exc}") from exc

    def verify_compile_log(
        self, log_path: str | Path, *, source_path: str | Path | None = None
    ) -> None:
        """Require a final zero-error/zero-warning summary and no diagnostics.

        Args:
            log_path: UTF-8 or BOM UTF-16 MetaEditor log.
            source_path: Source whose binary must be removed on failure.
                If omitted, standard EA.log -> EA.ex5 naming is assumed.
                compile_ea supplies this explicitly for its unique log names.

        Raises:
            CompilationError: Missing, unreadable, malformed, ambiguous, or
                dirty log. The associated .ex5 is deleted on failure.
        """
        log = Path(log_path)
        if source_path is not None and Path(source_path).suffix.lower() != ".mq5":
            raise CompilationError("source_path for log verification must end in .mq5")
        if source_path is None and log.suffix.lower() != ".log":
            raise CompilationError("log_path must end in .log when source_path is omitted")
        binary = Path(source_path).with_suffix(".ex5") if source_path is not None else log.with_suffix(".ex5")
        try:
            text, _, _ = _decode(log.read_bytes())
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            summary = re.compile(
                r"(?:Result:\s*)?(\d+)\s+error(?:\(s\)|s)?\s*,\s*"
                r"(\d+)\s+warning(?:\(s\)|s)?"
                r"(?:\s*,\s*\d+(?:\.\d+)?\s+msec elapsed(?:\s*,\s*cpu='[^']*')?)?",
                re.IGNORECASE,
            )
            summaries = [summary.fullmatch(line) for line in lines]
            final = summaries[-1] if summaries else None
            if (
                final is None
                or tuple(map(int, final.groups())) != (0, 0)
                or sum(match is not None for match in summaries) != 1
            ):
                raise CompilationError(f"log must conclude with one clean 0 error(s), 0 warning(s) summary: {log}")
            if any(re.search(r"\b(?:error|warning)(?:\s+\d+)?\s*:", line, re.IGNORECASE)
                   for line in lines[:-1]):
                raise CompilationError(f"log contains error/warning diagnostics despite clean summary: {log}")
        except (OSError, UnicodeError, CompilationError) as exc:
            self._remove_binary(binary)
            raise CompilationError(f"compile-log verification failed: {exc}") from exc

    @staticmethod
    def _remove_binary(binary: Path) -> None:
        try:
            binary.unlink(missing_ok=True)
        except OSError as exc:
            raise CompilationError(f"UNSAFE BINARY COULD NOT BE REMOVED: {binary}: {exc}") from exc
