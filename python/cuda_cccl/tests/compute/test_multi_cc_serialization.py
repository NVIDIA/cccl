# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for multi-compute-capability / no-GPU ahead-of-time compilation.

``make_<algo>(..., compute_capability=...)`` compiles one build result per target
arch without touching the CUDA driver (so it works on a build machine with no
GPU, using ``ProxyArray`` / ``ProxyValue`` placeholders). The resulting object
holds a ``{cc_key: build_result}`` map; the matching build result is loaded lazily on
the first call on a real device, and the whole map round-trips through
serialize/deserialize.

The executable tests use ``unary_transform`` as a representative algorithm and
discover which target arches this toolchain can actually compile for, so they
adapt to the installed CTK rather than hard-coding architectures.
"""

import numpy as np
import pytest

from cuda.compute import (
    ProxyArray,
    deserialize,
    make_unary_transform,
    serialize,
)
from cuda.compute._cccl_interop import (
    cc_to_key,
    current_device_cc_key,
    key_to_cc,
    normalize_compute_capabilities,
)

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    USING_V2, reason="serialization not supported on v2 (HostJIT) backend"
)


def _add_one(a):
    return a + 1


# ----------------------------------------------------------------------------
# Pure-logic tests (no GPU, no build result needed)
# ----------------------------------------------------------------------------


def test_normalize_compute_capabilities():
    assert normalize_compute_capabilities(None) is None
    assert normalize_compute_capabilities(90) == [(9, 0)]
    assert normalize_compute_capabilities((8, 0)) == [(8, 0)]
    assert normalize_compute_capabilities("75") == [(7, 5)]
    # list of packed ints, deduplicated + sorted
    assert normalize_compute_capabilities([90, 80, 90]) == [(8, 0), (9, 0)]
    # list of pairs
    assert normalize_compute_capabilities([(9, 0), (8, 0)]) == [(8, 0), (9, 0)]


def test_cc_key_roundtrip():
    for cc in (70, 75, 80, 86, 90, 120):
        assert cc_to_key(key_to_cc(cc)) == cc


def test_proxy_array_data_pointer_raises():
    p = ProxyArray(np.float32)
    with pytest.raises(RuntimeError, match="build-time placeholder"):
        _ = p.__cuda_array_interface__["data"]


# ----------------------------------------------------------------------------
# Helpers to build with proxies (no GPU data) and discover compilable arches
# ----------------------------------------------------------------------------


def _make_transform(cc):
    """Compile a unary transform for compute capability/ies ``cc`` using proxies."""
    return make_unary_transform(
        d_in=ProxyArray(np.int32),
        d_out=ProxyArray(np.int32),
        op=_add_one,
        compute_capability=cc,
    )


def _compilable_ccs(want=2):
    """Return up to ``want`` cc keys this toolchain can compile for (current first)."""
    current = current_device_cc_key()
    candidates = [current, 90, 100, 89, 120, 80, 86, 75]
    found = []
    for cc in dict.fromkeys(candidates):  # de-dupe, preserve order
        try:
            _make_transform(cc)
        except Exception:
            continue
        found.append(cc)
        if len(found) == want:
            break
    return found


# ----------------------------------------------------------------------------
# No-GPU compile via proxies (needs the built extension, but no device data)
# ----------------------------------------------------------------------------


def test_proxy_single_cc_compile_is_not_loaded():
    cc = current_device_cc_key()
    builder = _make_transform(cc)
    assert set(builder.build_results.keys()) == {cc}
    # compile() must not touch the driver — no kernels loaded yet.
    assert not builder.build_results[cc]._loaded


def test_proxy_multi_cc_compile_produces_one_build_result_per_cc():
    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")
    builder = _make_transform(ccs)
    assert set(builder.build_results.keys()) == set(ccs)
    assert all(not br._loaded for br in builder.build_results.values())


def test_multi_cc_serialize_deserialize_roundtrip_defers_load():
    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")
    builder = _make_transform(ccs)
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)
    assert set(loaded.build_results.keys()) == set(ccs)
    # Deserialized build results are not loaded (load is deferred to first call) — so
    # a multi-arch artifact stays portable and needs no live device to load.
    assert all(not br._loaded for br in loaded.build_results.values())


# ----------------------------------------------------------------------------
# GPU execution: lazily-loaded build result for the current device runs correctly
# ----------------------------------------------------------------------------


def test_compile_only_then_lazy_load_and_execute():
    cp = pytest.importorskip("cupy")
    cc = current_device_cc_key()

    # Single-cc compile-only build (no fused load) for the current device.
    reducer = _make_transform(cc)
    assert not reducer.build_results[cc]._loaded

    h = np.arange(1024, dtype=np.int32)
    d_in = cp.asarray(h)
    d_out = cp.empty_like(d_in)
    reducer(d_in=d_in, d_out=d_out, op=_add_one, num_items=d_in.size)
    cp.cuda.runtime.deviceSynchronize()

    # First call lazily loaded the current-device build result and ran correctly.
    assert reducer.build_results[cc]._loaded
    np.testing.assert_array_equal(d_out.get(), h + 1)


def test_multi_cc_loads_only_the_current_device_build_result():
    cp = pytest.importorskip("cupy")
    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")
    cc = current_device_cc_key()
    assert cc in ccs
    other = next(c for c in ccs if c != cc)

    builder = _make_transform(ccs)
    blob = serialize(builder)
    loaded = deserialize(blob)

    h = np.arange(256, dtype=np.int32)
    d_in = cp.asarray(h)
    d_out = cp.empty_like(d_in)
    loaded(d_in=d_in, d_out=d_out, op=_add_one, num_items=d_in.size)
    cp.cuda.runtime.deviceSynchronize()

    np.testing.assert_array_equal(d_out.get(), h + 1)
    # Only the current device's build result is loaded; the other stays lazy.
    assert loaded.build_results[cc]._loaded
    assert not loaded.build_results[other]._loaded


def test_fused_fast_path_unchanged_when_no_cc_given():
    cp = pytest.importorskip("cupy")
    h = np.arange(512, dtype=np.int32)
    d_in = cp.asarray(h)
    d_out = cp.empty_like(d_in)

    # compute_capability omitted -> single, already-loaded build result (fused build).
    builder = make_unary_transform(d_in=d_in, d_out=d_out, op=_add_one)
    (only,) = builder.build_results.values()
    assert only._loaded  # fused build already loaded the kernels

    builder(d_in=d_in, d_out=d_out, op=_add_one, num_items=d_in.size)
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(d_out.get(), h + 1)
