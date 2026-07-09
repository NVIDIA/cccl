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

import concurrent.futures
import threading

import numpy as np
import pytest

import cuda.compute._cccl_interop as cccl_interop
from cuda.compute import (
    ProxyArray,
    deserialize,
    make_select,
    make_unary_transform,
    serialize,
)
from cuda.compute._caching import (
    _BuildResultCollection,
    cache_build_results,
    cache_with_registered_key_functions,
)
from cuda.compute._cccl_interop import (
    cc_to_key,
    current_device_cc_key,
    key_to_cc,
    normalize_compute_capabilities,
)
from cuda.compute._target_cc import target_cc
from cuda.compute.iterators import CountingIterator, TransformIterator
from cuda.compute.types import from_numpy_dtype

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


class _FakeBuildResult:
    def __init__(self, payload: bytes, load_gate=None, load_failures=0):
        self.payload = payload
        self.load_gate = load_gate
        self.load_failures = load_failures
        self.load_count = 0
        self.serialize_count = 0
        self._loaded = False

    def load(self):
        self.load_count += 1
        if self.load_gate is not None:
            started, proceed = self.load_gate
            started.set()
            assert proceed.wait(timeout=5)
        if self.load_count <= self.load_failures:
            raise RuntimeError("synthetic build-result load failure")
        self._loaded = True

    def serialize(self):
        self.serialize_count += 1
        return self.payload

    @classmethod
    def deserialize(cls, blob, load=True, check_cc=True):
        del check_cc
        result = cls(blob)
        if load:
            result.load()
        return result


class _CoordinatedGetCache(dict):
    """Make concurrent callers all observe the initial cache miss."""

    def __init__(self, participants):
        super().__init__()
        self._get_barrier = threading.Barrier(participants)
        self._setdefault_barrier = threading.Barrier(participants)

    def get(self, key, default=None):
        value = super().get(key, default)
        self._get_barrier.wait(timeout=5)
        return value

    def setdefault(self, key, default=None):
        value = super().setdefault(key, default)
        self._setdefault_barrier.wait(timeout=5)
        return value


class _TrackingLock:
    """Signal when a second caller tries to acquire an already-held lock."""

    def __init__(self):
        self.second_attempt = threading.Event()
        self._attempt_count = 0
        self._attempt_count_lock = threading.Lock()
        self._lock = threading.Lock()

    def __enter__(self):
        with self._attempt_count_lock:
            self._attempt_count += 1
            if self._attempt_count == 2:
                self.second_attempt.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def test_aot_build_results_load_once_per_device_without_recompiling():
    source = _FakeBuildResult(b"sm80-cubin")
    build_results = _BuildResultCollection({80: source})

    device_0_result = build_results.resolve(80, 0)
    device_1_result = build_results.resolve(80, 1)

    assert device_0_result is source
    assert device_1_result is not source
    assert build_results.resolve(80, 0) is device_0_result
    assert build_results.resolve(80, 1) is device_1_result
    assert source.load_count == 1
    assert source.serialize_count == 1
    assert device_1_result.load_count == 1


def test_aot_build_result_concurrent_load_is_coalesced_per_device():
    load_started = threading.Event()
    allow_load = threading.Event()
    source = _FakeBuildResult(b"sm80-cubin", (load_started, allow_load))
    build_results = _BuildResultCollection({80: source})
    barrier = threading.Barrier(4)

    def resolve():
        barrier.wait()
        return build_results.resolve(80, 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(resolve) for _ in range(4)]
        assert load_started.wait(timeout=5)
        allow_load.set()
        results = [future.result() for future in futures]

    assert all(result is source for result in results)
    assert source.load_count == 1


def test_aot_build_result_load_failure_is_shared_and_retryable():
    thread_count = 4
    source = _FakeBuildResult(b"sm80-cubin", load_failures=1)
    build_results = _BuildResultCollection({80: source})
    build_results._loaded_results = _CoordinatedGetCache(thread_count)
    barrier = threading.Barrier(thread_count)

    def resolve():
        barrier.wait()
        try:
            build_results.resolve(80, 0)
        except RuntimeError as error:
            return error
        raise AssertionError("the first load must fail")

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        errors = [
            future.result()
            for future in [executor.submit(resolve) for _ in range(thread_count)]
        ]

    assert len({id(error) for error in errors}) == 1
    assert str(errors[0]) == "synthetic build-result load failure"
    assert source.load_count == 1

    # A failed in-flight entry is removed. A later caller retries the load and
    # can cache the successful result.
    build_results._loaded_results = {}
    assert build_results.resolve(80, 0) is source
    assert source.load_count == 2
    assert source._loaded


def test_aot_serialization_waits_for_canonical_first_load():
    load_started = threading.Event()
    allow_load = threading.Event()
    source = _FakeBuildResult(b"sm80-cubin", (load_started, allow_load))
    build_results = _BuildResultCollection({80: source})
    source_lock = _TrackingLock()
    build_results._source_locks[80] = source_lock

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        load_future = executor.submit(build_results.resolve, 80, 0)
        assert load_started.wait(timeout=5)
        serialize_future = executor.submit(build_results.serialize_build_result, 80)
        try:
            assert source_lock.second_attempt.wait(timeout=5)
            assert source.serialize_count == 0
        finally:
            allow_load.set()

        assert load_future.result() is source
        assert serialize_future.result() == b"sm80-cubin"

    assert source.serialize_count == 1


def test_current_device_build_result_preserves_device_query_free_fast_path(
    monkeypatch,
):
    source = _FakeBuildResult(b"sm80-cubin")
    source._loaded = True
    build_results = _BuildResultCollection({80: source}, loaded_device_id=0)

    def fail_device_query():
        raise AssertionError("current-device builds must not query on invocation")

    monkeypatch.setattr(cccl_interop, "current_device_info", fail_device_query)

    assert cccl_interop.resolve_build_result(build_results) is source
    assert source.load_count == 0


def test_explicit_aot_compilation_cache_normalizes_compute_capability():
    from cuda.compute import clear_all_caches

    clear_all_caches()
    build_count = 0

    def build():
        nonlocal build_count
        build_count += 1
        return _BuildResultCollection({80: _FakeBuildResult(b"sm80-cubin")})

    try:
        first = cache_build_results(
            _FakeBuildResult,
            "specialization",
            compute_capability=80,
            builder=build,
        )
        second = cache_build_results(
            _FakeBuildResult,
            "specialization",
            compute_capability=(8, 0),
            builder=build,
        )
    finally:
        clear_all_caches()

    assert first is second
    assert build_count == 1


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
    """Return up to ``want`` cc keys this toolchain can compile for.

    Device-free: compilation does not require a GPU, so this discovers arches
    without querying the current device (used by the no-GPU proxy tests).
    """
    candidates = [90, 100, 89, 120, 80, 86, 75]
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
    ccs = _compilable_ccs(1)
    if not ccs:
        pytest.skip("toolchain compiles for no target arch")
    cc = ccs[0]
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
# Iterator device code must be recompiled per target arch, even when the same
# iterator instance is reused across builds (no GPU needed)
# ----------------------------------------------------------------------------


def _counting_case():
    """Leaf iterator: the arch-specific bytes are its own advance/deref ops,
    memoized by IteratorBase's per-op caches."""

    def make():
        return CountingIterator(np.int32(0))

    def arch_bytes(it):
        # CountingIterator op symbol names derive only from (type, value_type),
        # so these LTO-IR blobs are byte-stable across instances at a given arch.
        return (
            bytes(it.get_advance_op().code.op_bytes),
            bytes(it.get_input_deref_op().code.op_bytes),
        )

    return make, arch_bytes


def _transform_case():
    """Compound iterator: the arch-specific leak surface is the compiled transform
    op, memoized by TransformIterator._compiled_op and embedded as an extra_ltoir.

    (The top-level wrapper LTO-IR is deliberately excluded: its symbol name is
    instance-dependent, so it is not byte-stable across instances. It is always
    recompiled per arch anyway; the compiled op below is the byte we must guard.)
    """

    def make():
        return TransformIterator(CountingIterator(np.int32(0)), _add_one)

    def arch_bytes(it):
        it.get_input_deref_op()  # drive the real path that compiles the op
        # wrapped_<fn> symbol names are content-based, so byte-stable per arch.
        return (bytes(it._get_compiled_op().code.op_bytes),)

    return make, arch_bytes


@pytest.mark.parametrize(
    "case", [_counting_case(), _transform_case()], ids=["counting", "transform"]
)
def test_iterator_op_ltoir_tracks_target_cc_across_instance_reuse(case):
    """Reusing one iterator instance across builds targeting different arches
    must recompile its device code for each arch.

    Iterators memoize their compiled ops on the instance (``IteratorBase``'s
    per-op caches and ``TransformIterator._compiled_op``). If that memo is not
    keyed on the target compute capability, passing the same iterator object to
    two ``make_<algo>`` calls whose targets *decrease* (e.g. 90 then 80) leaves
    the first build's newer-arch LTO-IR cached on the instance; it then gets
    linked into the older-arch build result, which nvJitLink rejects (final SM
    must be >= every linked input's arch). Guard against that by requiring the
    reused instance to yield, for each target, the same LTO-IR a fresh instance
    produces.
    """
    make, arch_bytes = case
    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")
    hi, lo = sorted(ccs, reverse=True)  # build the higher arch first

    # Ground truth: fresh instances compiled directly for each target arch.
    ref = {}
    for cc in (hi, lo):
        with target_cc(cc):
            ref[cc] = arch_bytes(make())
    # The two arches must produce different device code, else the test below
    # could pass vacuously.
    assert ref[hi] != ref[lo]

    # Reuse a single instance across both targets, higher arch first.
    it = make()
    with target_cc(hi):
        assert arch_bytes(it) == ref[hi]
    with target_cc(lo):
        # Regression: the memoized hi-arch LTO-IR must not leak into the lo build.
        assert arch_bytes(it) == ref[lo]


def test_default_build_keys_device_code_on_resolved_device_cc(monkeypatch):
    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")
    lo_cc, hi_cc = key_to_cc(sorted(ccs)[0]), key_to_cc(sorted(ccs)[-1])

    import cuda.compute._caching as _caching_mod

    class _FakeDevice:
        cc = hi_cc  # flipped between builds to emulate two different devices
        device_id = 0

        def __init__(self, *a, **k):
            pass

        @property
        def compute_capability(self):
            return _FakeDevice.cc

    monkeypatch.setattr(_caching_mod, "Device", _FakeDevice)

    # A cc-cached build that exercises the iterator's deref op under the default
    # (compute_capability=None) path, returning the build's target cc and LTO-IR.
    @cache_with_registered_key_functions
    def _build(*, it, compute_capability=None):
        from cuda.compute._target_cc import get_target_cc

        return get_target_cc(), bytes(it.get_input_deref_op().code.op_bytes)

    it = CountingIterator(np.int32(0))
    _FakeDevice.cc = hi_cc
    seen_hi, bytes_hi = _build(it=it)
    _FakeDevice.cc = lo_cc
    seen_lo, bytes_lo = _build(it=it)

    # The build's target cc is the resolved device cc, never None.
    assert seen_hi == hi_cc and seen_lo == lo_cc
    # Regression: the second device must not reuse the first device's arch.
    assert bytes_hi != bytes_lo
    assert set(it._input_deref_op.keys()) == {hi_cc, lo_cc}


def test_select_always_false_op_recompiles_per_target_cc():
    """``_always_false_op`` is a module-global cache whose LTO-IR is linked into
    every ``make_select`` build (as the three-way-partition's second predicate).

    It must be keyed on the target compute capability. A cc-unkeyed cache would
    compile once for whatever arch built first and hand that same op to every
    later build, linking newer-arch code into an older-arch result — which
    nvJitLink rejects. Unlike the iterator memos this leaks process-globally,
    across unrelated select builds.
    """
    from cuda.compute.algorithms._select import _get_always_false_op

    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")
    hi, lo = sorted(ccs, reverse=True)  # populate the cache for the higher arch first

    def code(cc):
        with target_cc(cc):
            return bytes(_get_always_false_op()._ltoir.op_bytes)

    b_hi = code(hi)
    b_lo = code(lo)
    # Regression: lo must not return the cached hi-arch LTO-IR.
    assert b_hi != b_lo


# ----------------------------------------------------------------------------
# GPU-free construction / deserialization (no device query, no recompile)
# ----------------------------------------------------------------------------


def test_unannotated_transform_iterator_construction_needs_no_gpu(monkeypatch):
    """Constructing an unannotated TransformIterator infers the lambda's return
    type; that inference is architecture-independent and must not query the
    current device.

    Otherwise a GPU-free build machine cannot even construct the iterator to hand
    to ``make_<algo>(compute_capability=...)`` — the device query happens at
    construction, before the explicit cc can take effect.
    """
    import cuda.compute._caching as _caching_mod

    class _NoDevice:
        def __init__(self, *a, **k):
            raise AssertionError("iterator construction must not query a CUDA device")

    # Any current-device query (e.g. the build cache's cc salt) now fails loudly.
    monkeypatch.setattr(_caching_mod, "Device", _NoDevice)

    it = TransformIterator(CountingIterator(np.int32(0)), lambda x: x + 1)
    # int32 + Python int promotes to int64 in Numba; the point is that the type
    # resolved with no device query.
    assert it.value_type == from_numpy_dtype(np.dtype(np.int64))


def test_select_deserialize_needs_no_gpu_and_no_recompile(monkeypatch):
    """``deserialize()`` must neither recompile device code nor require a GPU.

    ``_Select``'s always-false predicate is not serialized: its LTO-IR is already
    baked into the serialized three-way-partition build result, and ``__call__``
    needs only its (empty) runtime state. Regression: ``_after_deserialize`` must
    not call ``_always_false_op()``, which on a cold cache compiles via NVRTC and
    falls back to ``Device().compute_capability``.
    """
    # A multi-arch blob defers the single-target cc check at deserialize (see
    # _check_loadable), so deserialization is device-independent — the real
    # GPU-free AoT scenario (ship one blob, load on the target).
    ccs = _compilable_ccs(2)
    if len(ccs) < 2:
        pytest.skip("toolchain compiles for <2 target arches")

    def _cond(x):
        return np.uint8(x < np.int32(5))

    # Device-free AoT build + serialize (this may legitimately compile).
    sel = make_select(
        d_in=ProxyArray(np.int32),
        d_out=ProxyArray(np.int32),
        d_num_selected_out=ProxyArray(np.int32),
        cond=_cond,
        compute_capability=ccs,
    )
    blob = serialize(sel)

    # Now simulate a GPU-free machine on which any recompile or device query is
    # fatal, and require deserialize to succeed regardless.
    import cuda.compute._caching as _caching_mod
    import cuda.compute._cpp_compile as _cpp_mod
    import cuda.compute.algorithms._select as _select_mod

    def _boom(*a, **k):
        raise AssertionError("deserialize must not recompile the always-false op")

    class _NoDevice:
        def __init__(self, *a, **k):
            raise AssertionError("deserialize must not query a CUDA device")

    monkeypatch.setattr(_select_mod, "_always_false_op", _boom)
    monkeypatch.setattr(_select_mod, "_get_always_false_op", _boom)
    monkeypatch.setattr(_cpp_mod, "Device", _NoDevice)
    monkeypatch.setattr(_caching_mod, "Device", _NoDevice)

    loaded = deserialize(blob)
    # The always-false predicate is stateless, so its reconstructed runtime state
    # is empty — exactly what __call__ reads.
    assert loaded.always_false_op.get_state() == b""
    assert set(loaded.partitioner.build_results.keys()) == set(ccs)


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
    # This test executes kernels, so it legitimately needs a GPU; ensure the
    # current device's cc is one of the built arches.
    cc = current_device_cc_key()
    other = next((c for c in _compilable_ccs(3) if c != cc), None)
    if other is None:
        pytest.skip("toolchain compiles for <2 target arches")
    ccs = [cc, other]

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
