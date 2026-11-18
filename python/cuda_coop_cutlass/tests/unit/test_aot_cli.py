# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise script and module execution through the public CLI."""

import os
import sys

import pytest

from cuda.coop.cutlass import _aot_pack, aot
from cuda.coop.cutlass._aot_cli import main

pytestmark = [pytest.mark.unit, pytest.mark.backend_cutlass]


def test_run_module_receives_argv_and_can_import_sibling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if getattr(sys.flags, "safe_path", sys.flags.isolated):
        monkeypatch.syspath_prepend(str(tmp_path))
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "aot_dependency.py").write_text("value = 'dependency'\n")
    monkeypatch.syspath_prepend(str(extra))
    (tmp_path / "aot_sibling.py").write_text("value = 'sibling'\n")
    (tmp_path / "aot_workload.py").write_text(
        "from pathlib import Path\n"
        "import sys, aot_sibling, aot_dependency\n"
        "value = aot_sibling.value + aot_dependency.value\n"
        "Path(sys.argv[1]).write_text(value)\n"
        "raise SystemExit(0)\n"
    )
    result = tmp_path / "result"
    args = ["--", sys.executable, "-m", "aot_workload", str(result)]
    assert main(["run", "--mode", "off", "--pack", "unused", *args]) == 0
    assert result.read_text() == "siblingdependency"


def test_failed_script_does_not_publish_pack(tmp_path):
    script = tmp_path / "workload.py"
    script.write_text("raise SystemExit(5)\n")
    pack = tmp_path / "pack"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "capture",
                "--output",
                str(pack),
                "--",
                sys.executable,
                str(script),
            ]
        )
    assert error.value.code == 5
    assert not pack.exists()


def test_required_pack_reaches_child_and_restores_environment(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    with aot.capture(pack):
        _aot_pack.record("0" * 64, "compute_90", b"test-ir", {}, (13, 3))

    child = tmp_path / "child.py"
    child.write_text(
        "import os, sys\n"
        "from cuda.coop.cutlass import aot\n"
        "from cuda.coop.cutlass._compiler import _bundle\n"
        "assert os.environ['CUDA_COOP_CUTLASS_AOT_PACK_PATH'] == sys.argv[1]\n"
        "try:\n"
        "    _bundle.compile_bundle_source(\n"
        "        'missing', arch='compute_90', required_headers=())\n"
        "except aot.PackMissError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('child did not enforce required mode')\n"
    )
    script = tmp_path / "workload.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, *sys.argv[1:]], check=True)\n"
        "raise SystemExit(9)\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, "previous")
    monkeypatch.delenv(_aot_pack.PACK_MODE_ENV, raising=False)
    args = ["--", sys.executable, str(script), str(child), str(pack)]
    with pytest.raises(SystemExit) as error:
        main(["run", "--pack", "pack", "--mode", "required", *args])
    assert error.value.code == 9
    assert os.environ[_aot_pack.PACK_PATH_ENV] == "previous"
    assert _aot_pack.PACK_MODE_ENV not in os.environ


@pytest.mark.parametrize("isolated", [True, False])
def test_launcher_import_path(tmp_path, isolated):
    import subprocess

    script = tmp_path / "workload.py"
    script.write_text("import sys\nassert sys.path == sys.expected_path\n")
    expected_path = sys.path.copy()
    if not isolated:
        expected_path[:1] = [str(tmp_path)]
    setup = (
        f"import sys; sys.path[:] = {sys.path!r}\n"
        "from cuda.coop.cutlass._aot_cli import main\n"
        f"sys.expected_path = {expected_path!r}\n"
        "sys.exit(main())"
    )
    result = subprocess.run(
        [
            sys.executable,
            *(["-I"] if isolated else []),
            "-c",
            setup,
            "run",
            "--pack",
            "unused",
            "--mode",
            "off",
            "--",
            sys.executable,
            str(script),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
