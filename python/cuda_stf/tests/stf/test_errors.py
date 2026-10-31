# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Failures inside the C++ library reach Python as typed exceptions.

The C library catches every C++ exception at its boundary and records the
category and message; the bindings turn that record into an ``STFError``
subclass carrying the original text, instead of a generic ``RuntimeError`` or
a crashed interpreter.
"""

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


def _device_count():
    from cuda.bindings import runtime as cudart

    err, count = cudart.cudaGetDeviceCount()
    assert err == cudart.cudaError_t.cudaSuccess
    return count


def test_exception_hierarchy():
    assert issubclass(stf.STFError, RuntimeError)
    assert issubclass(stf.STFInvalidArgument, stf.STFError)
    assert issubclass(stf.STFInvalidArgument, ValueError)
    assert issubclass(stf.STFCudaError, stf.STFError)
    assert issubclass(stf.STFMemoryError, stf.STFError)
    assert issubclass(stf.STFMemoryError, MemoryError)


def test_invalid_device_ordinal_is_invalid_argument():
    bad = _device_count()
    with pytest.raises(stf.STFInvalidArgument, match="invalid device id") as info:
        stf.exec_place.device(bad)
    assert info.value.code is not None

    with pytest.raises(ValueError):
        stf.data_place.device(-1)


def test_message_carries_the_cpp_text():
    # Replicated data places are read-only: STF raises std::invalid_argument on
    # a write access. The C++ message must survive the trip through C.
    ctx = stf.context()
    ld = ctx.logical_data(np.zeros(8, dtype=np.float32))
    replicated = stf.data_place.replicated()
    with pytest.raises(stf.STFInvalidArgument, match="replicated"):
        with ctx.task(ld.write(replicated)):
            pass
    ctx.finalize()


def test_cuda_failure_is_cuda_error():
    dp = stf.data_place.device(0)
    with pytest.raises(stf.STFCudaError, match="CUDA"):
        dp.allocate(1 << 60)
    # Not sticky: the place still works afterwards.
    ptr = dp.allocate(1024)
    assert ptr != 0
    dp.deallocate(ptr, 1024)


def test_repeat_count_zero_is_invalid_argument():
    ctx = stf.stackable_context()
    with pytest.raises(stf.STFInvalidArgument, match="repeat count"):
        with ctx.repeat(0):
            pass
    ctx.finalize()


def test_python_side_checks_do_not_report_a_stale_record():
    # Leave a failure in the thread-local record, then trigger a check that the
    # bindings perform themselves. It must not pick up the earlier message.
    with pytest.raises(stf.STFInvalidArgument, match="invalid device id"):
        stf.exec_place.device(_device_count())

    ctx = stf.context()
    ctx.finalize()
    with pytest.raises(stf.STFError) as info:
        ctx.fence()
    assert type(info.value) is stf.STFError
    assert info.value.code is None
    assert "invalid device id" not in str(info.value)
