import os
import subprocess
import sys

import pytest
from numba import cuda


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_mamba_nvrtc_compile_count_drop():
    script = r"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests", "coop"))

from cuda.coop import _nvrtc
from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

_nvrtc.reset_compile_counter()
_nvrtc._set_compile_counter_enabled(True)

# Run the actual kernel test
_test = test_mamba_selective_scan_fwd_simple
_test()

print("__NVRTC_COUNT__=" + str(_nvrtc.get_compile_counter()))
"""

    def run(bundle, dump_dir=None):
        env = os.environ.copy()
        env["NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT"] = "1"
        env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = bundle
        if dump_dir is not None:
            env["NUMBA_CCCL_COOP_NVRTC_DUMP_DIR"] = dump_dir
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Subprocess failed (bundle={bundle}):\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        out = result.stdout + result.stderr
        count = None
        for line in out.splitlines():
            if line.startswith("__NVRTC_COUNT__="):
                count = int(line.split("=", 1)[1])
        if count is None:
            raise AssertionError(f"Missing NVRTC count in output: {out}")
        return count

    count_off = run("0")
    count_on = run("1")

    assert count_on < count_off


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_mamba_nvrtc_dump_bundle_only(tmp_path):
    script = r"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests", "coop"))

from cuda.coop import _nvrtc
from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

_nvrtc.reset_compile_counter()
_nvrtc._set_compile_counter_enabled(True)

# Run the actual kernel test
_test = test_mamba_selective_scan_fwd_simple
_test()

print("__NVRTC_COUNT__=" + str(_nvrtc.get_compile_counter()))
"""

    def run(bundle, dump_dir):
        env = os.environ.copy()
        env["NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT"] = "1"
        env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = bundle
        env["NUMBA_CCCL_COOP_NVRTC_DUMP_DIR"] = dump_dir
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Subprocess failed (bundle={bundle}):\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    dump_dir = tmp_path / "nvrtc_dump"
    dump_dir.mkdir()
    run("1", str(dump_dir))

    lto_files = [p for p in dump_dir.iterdir() if p.name.endswith("_lto.cu")]
    # gpu_dataclass bundle + kernel bundle
    assert len(lto_files) == 2
