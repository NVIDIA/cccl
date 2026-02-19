import os
import re
import subprocess
import sys

import pytest
from numba import cuda


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_coalesce_identical_one_shot_shims(tmp_path):
    script = r"""
import numpy as np
from numba import cuda
import cuda.coop as coop

@cuda.jit
def kernel(d_in, out1, out2):
    val1 = coop.block.sum(d_in[cuda.threadIdx.x], items_per_thread=1)
    val2 = coop.block.sum(d_in[cuda.threadIdx.x], items_per_thread=1)
    if cuda.threadIdx.x == 0:
        out1[0] = val1
        out2[0] = val2

threads = 128
h_input = np.ones(threads, dtype=np.int32)
d_input = cuda.to_device(h_input)
d_out1 = cuda.device_array(1, dtype=np.int32)
d_out2 = cuda.device_array(1, dtype=np.int32)
kernel[1, threads](d_input, d_out1, d_out2)
cuda.synchronize()
"""

    dump_dir = tmp_path / "nvrtc_dump"
    dump_dir.mkdir()
    env = os.environ.copy()
    env["NUMBA_CCCL_COOP_NVRTC_DUMP_DIR"] = str(dump_dir)
    env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = "1"
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
            f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    sources = []
    for path in dump_dir.iterdir():
        if path.name.endswith("_lto.cu"):
            sources.append(path.read_text(encoding="utf-8"))

    if not sources:
        raise AssertionError("No NVRTC LTO dump files found")

    merged = "\n".join(sources)
    matches = re.findall(r'extern "C" __device__ void (block_reduce\w*)\s*\(', merged)
    if not matches:
        raise AssertionError("Expected block_reduce shim definition in NVRTC dump")

    assert len(matches) == 1
