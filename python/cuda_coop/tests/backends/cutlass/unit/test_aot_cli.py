# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda.coop.cutlass import _aot_cli, aot


def _script(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "workload.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_script_workload_preserves_python_argv_and_import_path(tmp_path):
    result_path = tmp_path / "result.json"
    script = _script(
        tmp_path,
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(result_path)!r}).write_text(\n"
        "    json.dumps({'argv': sys.argv, 'path0': sys.path[0]}),\n"
        "    encoding='utf-8',\n"
        ")\n",
    )
    original_argv = sys.argv
    original_path = sys.path
    original_path_contents = list(sys.path)

    code = _aot_cli._run_python_workload(
        [sys.executable, str(script), "first", "second"]
    )

    assert code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "argv": [str(script.resolve()), "first", "second"],
        "path0": str(tmp_path),
    }
    assert sys.argv is original_argv
    assert sys.path is original_path
    assert sys.path == original_path_contents


def test_module_workload_uses_current_directory_and_arguments(
    tmp_path,
    monkeypatch,
):
    result_path = tmp_path / "module-result.json"
    module = tmp_path / "captured_module.py"
    module.write_text(
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(result_path)!r}).write_text(\n"
        "    json.dumps({'argv': sys.argv, 'path0': sys.path[0]}),\n"
        "    encoding='utf-8',\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    code = _aot_cli._run_python_workload(
        [sys.executable, "-m", "captured_module", "argument"]
    )

    assert code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["argv"][1:] == ["argument"]
    assert Path(result["argv"][0]) == module
    assert result["path0"] == str(tmp_path)


def test_capture_rolls_back_on_nonzero_system_exit(
    tmp_path,
    monkeypatch,
):
    script = _script(tmp_path, "raise SystemExit(7)\n")
    observed = {}

    class FakeCapture:
        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            observed["exception_type"] = exception_type
            return False

    monkeypatch.setattr(
        _aot_cli.aot, "capture", lambda *_args, **_kwargs: FakeCapture()
    )
    arguments = argparse.Namespace(
        output=tmp_path / "unused",
        name=None,
        command=[sys.executable, str(script)],
    )

    assert _aot_cli._run_capture(arguments) == 7
    assert observed["exception_type"] is _aot_cli._WorkloadExit


def test_capture_publishes_on_zero_system_exit(
    tmp_path,
    monkeypatch,
    capsys,
):
    script = _script(tmp_path, "raise SystemExit(0)\n")
    output = tmp_path / "captured.coop-aot"
    observed = {}

    class FakeCapture:
        result = SimpleNamespace(entries=(object(),), path=output)

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            observed["exception_type"] = exception_type
            return False

    monkeypatch.setattr(
        _aot_cli.aot, "capture", lambda *_args, **_kwargs: FakeCapture()
    )
    arguments = argparse.Namespace(
        output=output,
        name="workload",
        command=[sys.executable, str(script)],
    )

    assert _aot_cli._run_capture(arguments) == 0
    assert observed["exception_type"] is None
    assert "Captured 1 provider bundle(s)" in capsys.readouterr().err


def test_run_selects_pack_and_preserves_workload_exit_code(
    tmp_path,
    monkeypatch,
):
    script = _script(tmp_path, "raise SystemExit(5)\n")
    observed = {}

    @contextmanager
    def fake_use(pack, *, mode):
        observed["selection"] = (pack, mode)
        yield

    monkeypatch.setattr(_aot_cli.aot, "use", fake_use)
    pack = tmp_path / "selected.coop-aot"
    arguments = argparse.Namespace(
        pack=pack,
        mode="required",
        command=[sys.executable, str(script)],
    )

    assert _aot_cli._run_with_pack(arguments) == 5
    assert observed["selection"] == (pack, "required")


def test_inspect_emits_structured_json(tmp_path, monkeypatch, capsys):
    pack = tmp_path / "inspected.coop-aot"
    info = aot.PackInfo(
        path=pack,
        name="inspected",
        schema_version=1,
        provider_abi_version=1,
        writer_version="1.2.3",
        entries=(),
    )
    monkeypatch.setattr(_aot_cli.aot, "inspect", lambda _path: info)

    assert _aot_cli._run_inspect(argparse.Namespace(pack=pack)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "artifact_bytes": 0,
        "entries": [],
        "name": "inspected",
        "path": str(pack),
        "provider_abi_version": 1,
        "schema_version": 1,
        "writer_version": "1.2.3",
    }


def test_workload_rejects_a_different_python_interpreter(monkeypatch):
    monkeypatch.setattr(_aot_cli, "_same_interpreter", lambda _command: False)

    with pytest.raises(
        _aot_cli._CommandError,
        match="same Python interpreter",
    ):
        _aot_cli._python_workload(["python", "workload.py"])
