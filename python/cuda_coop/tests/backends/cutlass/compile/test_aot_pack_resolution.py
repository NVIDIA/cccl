# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

# isort: off
from cuda.coop.cutlass._compiler import (
    _bundle as provider_bundle,
    _bundle_contract as provider_contract,
    _cache as provider_cache,
    _nvrtc as provider_nvrtc,
)

# isort: on


class _NvrtcResult:
    NVRTC_SUCCESS = 0


class _FakeNvrtc:
    nvrtcResult = _NvrtcResult

    def __init__(self, version=(13, 0)):
        self.calls = []
        self.blob = b"jit-ltoir"
        self.version = version

    def nvrtcVersion(self):
        return 0, *self.version

    def nvrtcCreateProgram(self, source, name, num_headers, headers, include_names):
        self.calls.append(("create", source))
        return 0, object()

    def nvrtcCompileProgram(self, program, num_options, options):
        self.calls.append(("compile", tuple(options)))
        return (0,)

    def nvrtcGetLTOIRSize(self, program):
        self.calls.append(("get_ltoir_size",))
        return 0, len(self.blob)

    def nvrtcGetLTOIR(self, program, blob):
        self.calls.append(("get_ltoir",))
        blob[:] = self.blob
        return (0,)

    def nvrtcDestroyProgram(self, program):
        self.calls.append(("destroy",))
        return (0,)


def _compile_kwargs():
    return {
        "scope": "cuda.coop.cutlass",
        "provider_dir": provider_bundle.__file__,
        "registered_headers": dict,
        "select_bundle_format": lambda: "ltoir",
        "resolve_nvrtc_sm_arch": lambda: "sm_80",
        "resolve_nvrtc_arch": lambda: "compute_80",
        "symbols": ("provider_a", "provider_z"),
    }


def _precompiled_resolver(artifact: Path, *, producer_version: str | None = "13.0"):
    def resolver(request):
        return provider_contract.BundleResolution(
            request=request,
            path=str(artifact),
            layouts_by_expression={},
            route=provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
            producer_compiler="nvrtc",
            producer_compiler_version=producer_version,
            producer_toolkit_version="13.0",
            phase_timings_ns={},
        )

    return resolver


def _capture_pack(tmp_path: Path) -> Path:
    from cuda.coop.cutlass import aot

    output = tmp_path / "provider.coop-aot"
    with aot.capture(output):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )
    return output


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_COOP_CUTLASS_AOT_PACK_PATH", raising=False)
    monkeypatch.delenv("CUDA_COOP_CUTLASS_AOT_MODE", raising=False)
    monkeypatch.setenv(
        provider_cache.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", _FakeNvrtc())
    provider_bundle.reset_compile_state()
    yield
    provider_bundle.reset_compile_state()


def test_default_jit_path_does_not_import_or_time_aot_dispatch(monkeypatch):
    module_name = "cuda.coop.cutlass._aot_pack"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import_module = importlib.import_module

    def guarded_import(name, package=None):
        if name == module_name:
            raise AssertionError("ordinary JIT must not import the AOT pack module")
        return real_import_module(name, package)

    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(importlib, "import_module", guarded_import)
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []

    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void ordinary_jit() {}\n',
            **_compile_kwargs(),
        )

    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_NVRTC
    assert "pack_lookup" not in observed[0].phase_timings_ns
    assert module_name not in sys.modules


def test_import_and_inspect_do_not_activate_aot_resolution(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    assert aot.inspect(pack).entries
    provider_bundle.reset_compile_state()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("importing or inspecting a pack must not activate it")

    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(_aot_pack, "resolve_precompiled_bundle", forbidden)
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []
    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void inspect_is_inert() {}\n',
            **_compile_kwargs(),
        )

    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_NVRTC
    assert "pack_lookup" not in observed[0].phase_timings_ns


def test_required_hit_skips_nvrtc_and_mutable_jit_io(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact AOT hits must bypass mutable JIT I/O")

    observed = []
    with monkeypatch.context() as hit_patch:
        hit_patch.setattr(provider_bundle, "cccl_include_dirs", forbidden)
        hit_patch.setattr(provider_bundle, "include_dirs_identity", forbidden)
        hit_patch.setattr(provider_bundle, "maybe_dump_source", forbidden)
        hit_patch.setattr(provider_cache, "ensure_cache_dir", forbidden)
        hit_patch.setattr(provider_nvrtc, "cuda_nvrtc", forbidden)
        with (
            aot.use(pack, mode="required"),
            provider_bundle.activate_bundle_resolution_observer(observed.append),
        ):
            path = provider_bundle.compile_bundle_source(
                'extern "C" __device__ void provider_a() {}\n',
                **_compile_kwargs(),
            )

    assert Path(path).read_bytes() == b"jit-ltoir"
    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_AOT_PACK
    assert "pack_lookup" in observed[0].phase_timings_ns
    assert "precompile_resolvers" not in observed[0].phase_timings_ns
    assert "header_resolution" not in observed[0].phase_timings_ns


def test_auto_miss_falls_back_and_required_miss_precedes_nvrtc(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)

    with aot.use(pack, mode="auto"):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void pack_miss() {}\n',
            **_compile_kwargs(),
        )
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1

    provider_bundle.reset_compile_state()
    fake_nvrtc.calls.clear()
    mismatched_symbols = _compile_kwargs()
    mismatched_symbols["symbols"] = ("provider_other",)
    with (
        aot.use(pack, mode="required"),
        pytest.raises(aot.PackMissError, match="entry is absent"),
    ):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **mismatched_symbols,
        )
    assert fake_nvrtc.calls == []


def test_selected_artifact_is_stable_after_pack_mutation(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    payload = _aot_pack._parse_manifest((pack / "manifest.json").read_bytes())
    entry = payload.entries[0]
    pack_artifact = pack / "artifacts" / f"{entry.artifact_sha256}.ltoir"
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )

    with aot.use(pack, mode="required"):
        pack_artifact.write_bytes(b"mutated-after-validation")
        path = provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    assert path != str(pack_artifact)
    assert path.startswith("/proc/self/fd/")
    assert Path(path).read_bytes() == b"jit-ltoir"
    with pytest.raises(PermissionError):
        Path(path).write_bytes(b"attempted-mutation")


def test_auto_version_incompatibility_falls_back_to_nvrtc(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    provider_bundle.reset_compile_state()
    monkeypatch.setenv(
        provider_cache.CACHE_DIR_ENV,
        str(tmp_path / "fallback-provider-cache"),
    )
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=12, minor=9),
    )
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []

    with (
        aot.use(pack, mode="auto"),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_NVRTC
    assert "pack_lookup" in observed[0].phase_timings_ns


def test_environment_selection_and_explicit_off_override(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )
    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, str(pack))
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "required")

    observed = []
    with provider_bundle.activate_bundle_resolution_observer(observed.append):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )
    assert observed[-1].route == provider_contract.RESOLUTION_ROUTE_AOT_PACK

    fallback = tmp_path / "fallback.ltoir"
    fallback.write_bytes(b"fallback")
    with (
        aot.use(tmp_path / "ignored", mode="off"),
        provider_bundle.activate_bundle_precompile_resolver(
            _precompiled_resolver(fallback)
        ),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )
    assert observed[-1].route == provider_contract.RESOLUTION_ROUTE_PRECOMPILED


def test_aot_and_generic_resolvers_follow_context_nesting_and_restore(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    fallback = tmp_path / "fallback.ltoir"
    fallback.write_bytes(b"fallback")
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )
    observed = []

    with (
        provider_bundle.activate_bundle_precompile_resolver(
            _precompiled_resolver(fallback)
        ),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        with aot.use(pack, mode="required"):
            provider_bundle.compile_bundle_source(
                'extern "C" __device__ void provider_a() {}\n',
                **_compile_kwargs(),
            )
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    with (
        aot.use(pack, mode="required"),
        provider_bundle.activate_bundle_precompile_resolver(
            _precompiled_resolver(fallback)
        ),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    assert [resolution.route for resolution in observed] == [
        provider_contract.RESOLUTION_ROUTE_AOT_PACK,
        provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
        provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
    ]
    assert "pack_lookup" in observed[0].phase_timings_ns
    assert "pack_lookup" not in observed[1].phase_timings_ns
    assert "pack_lookup" not in observed[2].phase_timings_ns


def test_copy_context_propagates_required_selection_to_worker_thread(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    observed = []

    with (
        aot.use(pack, mode="required"),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                context.run,
                provider_bundle.compile_bundle_source,
                'extern "C" __device__ void provider_a() {}\n',
                **_compile_kwargs(),
            ).result()

    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_AOT_PACK
    assert fake_nvrtc.calls == []


def test_copy_context_propagates_capture_to_worker_thread(tmp_path):
    from cuda.coop.cutlass import aot

    output = tmp_path / "threaded.coop-aot"
    with aot.capture(output):
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                context.run,
                provider_bundle.compile_bundle_source,
                'extern "C" __device__ void threaded_capture() {}\n',
                **_compile_kwargs(),
            ).result()

    assert len(aot.inspect(output).entries) == 1


def test_environment_resolver_is_outermost_to_context_resolvers(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack

    pack = _capture_pack(tmp_path)
    fallback = tmp_path / "fallback.ltoir"
    fallback.write_bytes(b"fallback")
    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, str(pack))
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "required")
    observed = []

    with (
        provider_bundle.activate_bundle_precompile_resolver(
            _precompiled_resolver(fallback)
        ),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        path = provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    assert path == str(fallback)
    assert observed[0].route == provider_contract.RESOLUTION_ROUTE_PRECOMPILED
    assert "pack_lookup" not in observed[0].phase_timings_ns


def test_typed_resolver_rejects_a_resolution_from_another_route(tmp_path):
    artifact = tmp_path / "wrong-route.ltoir"
    artifact.write_bytes(b"wrong-route")
    resolver = provider_bundle._bundle_precompile_resolver(
        _precompiled_resolver(artifact),
        route=provider_contract.RESOLUTION_ROUTE_AOT_PACK,
        phase="pack_lookup",
    )

    with (
        provider_bundle.activate_bundle_precompile_resolver(resolver),
        pytest.raises(ValueError, match="invalid route"),
    ):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void wrong_route() {}\n',
            **_compile_kwargs(),
        )


def test_environment_pack_path_must_be_absolute(monkeypatch):
    from cuda.coop.cutlass import _aot_pack, aot

    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, "relative/pack")
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "auto")

    with pytest.raises(aot.PackError, match="must be an absolute path"):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )


def test_auto_mode_does_not_fall_back_from_pack_integrity_errors(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    (pack / "manifest.json").write_text("not-json", encoding="utf-8")
    provider_bundle.reset_compile_state()
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, str(pack))
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "auto")

    with pytest.raises(aot.PackIntegrityError, match="valid canonical JSON"):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )
    assert fake_nvrtc.calls == []


@pytest.mark.parametrize(
    ("consumer", "mode", "route_or_error"),
    [
        ((13, 0), "required", provider_contract.RESOLUTION_ROUTE_AOT_PACK),
        ((13, 1), "required", provider_contract.RESOLUTION_ROUTE_AOT_PACK),
        ((12, 9), "required", "different major versions"),
    ],
)
def test_nvjitlink_compatibility_matrix(
    monkeypatch,
    tmp_path,
    consumer,
    mode,
    route_or_error,
):
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(*consumer),
    )
    observed = []
    context = aot.use(pack, mode=mode)
    if route_or_error == provider_contract.RESOLUTION_ROUTE_AOT_PACK:
        with (
            context,
            provider_bundle.activate_bundle_resolution_observer(observed.append),
        ):
            provider_bundle.compile_bundle_source(
                'extern "C" __device__ void provider_a() {}\n',
                **_compile_kwargs(),
            )
        assert observed[0].route == route_or_error
    else:
        with context, pytest.raises(aot.PackMissError, match=route_or_error):
            provider_bundle.compile_bundle_source(
                'extern "C" __device__ void provider_a() {}\n',
                **_compile_kwargs(),
            )


def test_consumer_nvjitlink_version_is_cached(monkeypatch):
    import cuda.bindings.nvjitlink as cuda_nvjitlink
    from cuda.coop.cutlass import _aot_pack

    calls = []

    def version():
        calls.append(None)
        return 13, 1

    monkeypatch.setattr(cuda_nvjitlink, "version", version)
    _aot_pack._consumer_nvjitlink_version.cache_clear()
    try:
        assert _aot_pack._consumer_nvjitlink_version() == _aot_pack._CudaVersion(
            13,
            1,
        )
        assert _aot_pack._consumer_nvjitlink_version() == _aot_pack._CudaVersion(
            13,
            1,
        )
    finally:
        _aot_pack._consumer_nvjitlink_version.cache_clear()
    assert len(calls) == 1


@pytest.mark.parametrize("failure_kind", ["runtime", "binding"])
@pytest.mark.parametrize("mode", ["auto", "required"])
def test_nvjitlink_version_failure_obeys_pack_mode(
    monkeypatch,
    tmp_path,
    failure_kind,
    mode,
):
    import cuda.bindings.nvjitlink as cuda_nvjitlink
    from cuda.coop.cutlass import _aot_pack, aot

    pack = _capture_pack(tmp_path)
    monkeypatch.setenv(
        provider_cache.CACHE_DIR_ENV,
        str(tmp_path / "fallback-provider-cache"),
    )
    provider_bundle.reset_compile_state()

    def unavailable_version():
        if failure_kind == "runtime":
            raise RuntimeError("nvJitLink version symbol is unavailable")
        raise cuda_nvjitlink.nvJitLinkError(cuda_nvjitlink.Result.ERROR_INTERNAL.value)

    monkeypatch.setattr(cuda_nvjitlink, "version", unavailable_version)
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", fake_nvrtc)
    _aot_pack._consumer_nvjitlink_version.cache_clear()
    try:
        if mode == "auto":
            observed = []
            with (
                aot.use(pack, mode=mode),
                provider_bundle.activate_bundle_resolution_observer(observed.append),
            ):
                provider_bundle.compile_bundle_source(
                    'extern "C" __device__ void provider_a() {}\n',
                    **_compile_kwargs(),
                )
            assert observed[0].route == provider_contract.RESOLUTION_ROUTE_NVRTC
            assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1
        else:
            with (
                aot.use(pack, mode=mode),
                pytest.raises(
                    aot.PackMissError,
                    match="Unable to determine the consumer nvJitLink version",
                ),
            ):
                provider_bundle.compile_bundle_source(
                    'extern "C" __device__ void provider_a() {}\n',
                    **_compile_kwargs(),
                )
            assert fake_nvrtc.calls == []
    finally:
        _aot_pack._consumer_nvjitlink_version.cache_clear()


def test_required_rejects_older_consumer_minor(monkeypatch, tmp_path):
    from cuda.coop.cutlass import _aot_pack, aot

    monkeypatch.setattr(provider_nvrtc, "cuda_nvrtc", _FakeNvrtc((13, 1)))
    pack = tmp_path / "provider.coop-aot"
    with aot.capture(pack):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=0),
    )

    with (
        aot.use(pack, mode="required"),
        pytest.raises(aot.PackMissError, match="older than producer NVRTC"),
    ):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )


def test_capture_reexports_aot_hit_and_preserves_producer_version(
    monkeypatch,
    tmp_path,
):
    from cuda.coop.cutlass import _aot_pack, aot

    first_pack = _capture_pack(tmp_path)
    second_pack = tmp_path / "reexported.coop-aot"
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=1),
    )

    with aot.capture(second_pack), aot.use(first_pack, mode="required"):
        provider_bundle.compile_bundle_source(
            'extern "C" __device__ void provider_a() {}\n',
            **_compile_kwargs(),
        )

    first_entry = aot.inspect(first_pack).entries[0]
    second_entry = aot.inspect(second_pack).entries[0]
    assert second_entry == first_entry
    assert second_entry.producer_version == (13, 0)


def test_capture_rejects_missing_legacy_producer_version(tmp_path):
    from cuda.coop.cutlass import aot

    output = tmp_path / "legacy.coop-aot"
    source = 'extern "C" __device__ void provider_a() {}\n'
    provider_bundle.compile_bundle_source(
        source,
        **_compile_kwargs(),
    )
    metadata_path = next((tmp_path / "provider-cache").glob("*.layouts.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["producer"]["compiler_version"] = None
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    provider_bundle.reset_compile_state()

    with (
        pytest.raises(aot.CaptureError, match="legacy provider disk cache"),
        aot.capture(output),
    ):
        provider_bundle.compile_bundle_source(
            source,
            **_compile_kwargs(),
        )
    assert not output.exists()
