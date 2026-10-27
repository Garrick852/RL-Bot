"""Safe source edits and simulated local MetaEditor evidence."""

import codecs
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engine_mt5.patcher import CompilationError, MQL5Patcher, SourcePatchError


class MQL5PatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "EA.mq5"
        self.binary = self.source.with_suffix(".ex5")
        self.compiler = self.root / "metaeditor64.exe"
        self.compiler.write_bytes(b"mock compiler only")
        self.source.write_bytes(
            b"input int InpLeftBars = 5; // OPTIMIZED:InpLeftBars\r\n"
            b"input double Risk = 0.1; // OPTIMIZED:Risk\r\n"
            b"input bool Enable = true; // OPTIMIZED:Enable\r\n"
        )
        self.patcher = MQL5Patcher()

    def test_deterministic_typed_patch_and_encoding_preservation(self):
        original = self.source.read_bytes().decode("utf-8")
        for bom, encoding in ((b"", "utf-8"), (codecs.BOM_UTF8, "utf-8"),
                              (codecs.BOM_UTF16_LE, "utf-16-le"),
                              (codecs.BOM_UTF16_BE, "utf-16-be")):
            with self.subTest(encoding=encoding, bom=bom):
                self.source.write_bytes(bom + original.encode(encoding))
                result = self.patcher.patch_source(self.source, {
                    "parameters": {"InpLeftBars": 12, "Risk": .25, "Enable": False},
                })
                self.assertEqual(result, self.source)
                data = self.source.read_bytes()
                self.assertEqual(data, bom + original.replace("= 5;", "= 12;")
                                 .replace("= 0.1;", "= 0.25;")
                                 .replace("= true;", "= false;").encode(encoding))
                self.patcher.patch_source(self.source, {"InpLeftBars": 12})
                self.assertEqual(data, self.source.read_bytes())

    def test_patch_failure_is_atomic_and_missing_duplicate_commented_anchors_rejected(self):
        original = self.source.read_bytes()
        for parameters in ({"InpLeftBars": 9, "Missing": 1}, {"Risk": float("nan")},
                           {"InpLeftBars": True}, {"InpLeftBars": 2**40},
                           {"Risk": "1; dangerous()"}, {}, {"Enable": 1}):
            with self.subTest(parameters=parameters), self.assertRaises(SourcePatchError):
                self.patcher.patch_source(self.source, parameters)
            self.assertEqual(self.source.read_bytes(), original)
        for source in (
            "// input int A = 1; // OPTIMIZED:A\n",
            "/*\ninput int A = 1; // OPTIMIZED:A\n*/",
            'string text = "input int A = 1; // OPTIMIZED:A";',
            "input int A = 1; // OPTIMIZED:AB",
            "input int A = 1; // OPTIMIZED:A\ninput int A = 2; // OPTIMIZED:A",
        ):
            self.source.write_text(source)
            with self.subTest(source=source), self.assertRaises(SourcePatchError):
                self.patcher.patch_source(self.source, {"A": 3})
            self.assertEqual(self.source.read_text(), source)

    def test_log_variants_and_strict_final_summary(self):
        log = self.root / "EA.log"
        for summary in (
            "0 error(s), 0 warning(s)",
            "Result: 0 errors, 0 warnings, 123 msec elapsed, cpu='X64 Regular'",
            "0 errors, 0 warnings",
        ):
            self.binary.write_bytes(b"compiled")
            log.write_bytes(codecs.BOM_UTF16_LE + (summary + "\r\n").encode("utf-16-le"))
            self.patcher.verify_compile_log(log)
            self.assertTrue(self.binary.exists())
        for content in (
            "", "0 errors, 1 warnings", "1 errors, 0 warnings",
            "0 errors, 0 warnings\nunfinished", "no summary",
            "warning 42: bad\n0 errors, 0 warnings",
            "0 errors, 0 warnings\n0 errors, 0 warnings",
        ):
            self.binary.write_bytes(b"unsafe")
            log.write_text(content)
            with self.subTest(content=content), self.assertRaises(CompilationError):
                self.patcher.verify_compile_log(log)
            self.assertFalse(self.binary.exists())
        log.unlink()
        self.binary.write_bytes(b"unsafe")
        with self.assertRaises(CompilationError):
            self.patcher.verify_compile_log(log)
        self.assertFalse(self.binary.exists())

    def compiler_mock(self, *, log_text="0 error(s), 0 warning(s)", binary=True, code=0):
        def run(arguments, **kwargs):
            self.assertEqual(kwargs["timeout"], 30.)
            self.assertFalse(kwargs["check"])
            self.assertFalse(self.binary.exists(), "stale binary must be removed first")
            self.assertIn(f"/compile:{self.source}", arguments)
            log = Path(next(arg[len("/log:"):] for arg in arguments if arg.startswith("/log:")))
            if log_text is not None:
                log.write_text(log_text)
            if binary:
                self.binary.write_bytes(b"new binary")
            return SimpleNamespace(returncode=code)
        return run

    def test_compile_requires_fresh_clean_binary(self):
        self.binary.write_bytes(b"stale")
        with patch("engine_mt5.patcher.subprocess.run", side_effect=self.compiler_mock()):
            log = self.patcher.compile_ea(self.compiler, self.source)
        self.assertTrue(log.is_file())
        self.assertEqual(self.binary.read_bytes(), b"new binary")
        self.assertIn(".compile.log", log.name)

    def test_compile_failures_and_timeouts_remove_binary(self):
        for options in (
            {"log_text": "0 errors, 1 warnings"}, {"log_text": None},
            {"binary": False}, {"code": 1},
        ):
            self.binary.write_bytes(b"stale")
            with self.subTest(options=options), patch(
                "engine_mt5.patcher.subprocess.run", side_effect=self.compiler_mock(**options)
            ), self.assertRaises(CompilationError):
                self.patcher.compile_ea(self.compiler, self.source)
            self.assertFalse(self.binary.exists())
        with patch("engine_mt5.patcher.subprocess.run", side_effect=subprocess.TimeoutExpired("mock", 30)):
            with self.assertRaisesRegex(CompilationError, "timed out"):
                self.patcher.compile_ea(self.compiler, self.source)
        self.assertFalse(self.binary.exists())

    def test_source_change_during_compilation_refuses_binary(self):
        compile_mock = self.compiler_mock()
        def run(*args, **kwargs):
            result = compile_mock(*args, **kwargs)
            self.source.write_bytes(b"changed")
            return result
        with patch("engine_mt5.patcher.subprocess.run", side_effect=run):
            with self.assertRaisesRegex(CompilationError, "source changed"):
                self.patcher.compile_ea(self.compiler, self.source)
        self.assertFalse(self.binary.exists())

    def test_invalid_timeout_and_cleanup_failure_are_explicit(self):
        for timeout in (True, 0, -1, float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                MQL5Patcher(timeout_seconds=timeout)
        log = self.root / "EA.log"
        log.write_text("1 errors, 0 warnings")
        with patch.object(Path, "unlink", side_effect=PermissionError("locked binary")):
            with self.assertRaisesRegex(CompilationError, "COULD NOT BE REMOVED"):
                self.patcher.verify_compile_log(log)


if __name__ == "__main__":
    unittest.main()
