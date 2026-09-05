# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import threading
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cuda.coop.cutlass import _aot_pack, aot
from cuda.coop.cutlass._compiler import _bundle_contract as provider_contract
from cuda.coop.cutlass._compiler import _nvrtc as provider_nvrtc


class _NvrtcResult:
    NVRTC_SUCCESS = 0


class _FakeNvrtc:
    nvrtcResult = _NvrtcResult

    def __init__(self):
        self.blob = b"portable-ltoir"

    def nvrtcVersion(self):
        return 0, 13, 0

    def nvrtcCreateProgram(self, source, name, num_headers, headers, include_names):
        return 0, object()

    def nvrtcCompileProgram(self, program, num_options, options):
        return (0,)

    def nvrtcGetLTOIRSize(self, program):
        return 0, len(self.blob)

    def nvrtcGetLTOIR(self, program, blob):
        blob[:] = self.blob
        return (0,)

    def nvrtcDestroyProgram(self, program):
        return (0,)


@pytest.fixture
def provider_bundle(monkeypatch, tmp_path):
    pytest.importorskip("cutlass")
    from cuda.coop.cutlass._compiler import _bundle as _provider_bundle
    from cuda.coop.cutlass._compiler import _cache as _provider_cache
    from cuda.coop.cutlass._compiler import _nvrtc as _provider_nvrtc

    monkeypatch.setenv(
        _provider_cache.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setattr(_provider_nvrtc, "cuda_nvrtc", _FakeNvrtc())
    _provider_bundle.reset_compile_state()
    try:
        yield _provider_bundle
    finally:
        _provider_bundle.reset_compile_state()


def _compile_kwargs(provider_bundle, arch: str):
    suffix = arch.removeprefix("sm_")
    return {
        "scope": "cuda.coop.cutlass",
        "provider_dir": provider_bundle.__file__,
        "registered_headers": lambda: {},
        "select_bundle_format": lambda: "ltoir",
        "resolve_nvrtc_sm_arch": lambda: arch,
        "resolve_nvrtc_arch": lambda: f"compute_{suffix}",
    }


def _precompiled_resolver(provider_bundle, artifact: Path):
    def resolver(request):
        return provider_contract.BundleResolution(
            request=request,
            path=str(artifact),
            layouts_by_expression={
                expression: provider_contract.StorageLayout(16, 8)
                for expression in request.identity.layout_expressions
            },
            route=provider_contract.RESOLUTION_ROUTE_PRECOMPILED,
            producer_compiler="nvrtc",
            producer_compiler_version="13.0",
            producer_toolkit_version="13.0",
            phase_timings_ns={},
        )

    return resolver


def _compile_source(
    provider_bundle,
    source: str,
    arch: str = "sm_80",
    *,
    symbols: tuple[str, ...] = (
        "provider_z",
        "provider_a",
        "provider_z",
    ),
) -> str:
    return provider_bundle.compile_bundle_source(
        source,
        symbols=symbols,
        **_compile_kwargs(provider_bundle, arch),
    )


def _capture_pack(
    tmp_path: Path,
    provider_bundle,
    *,
    output_name: str = "captured.coop-aot",
    sources: tuple[tuple[str, str], ...] = (
        ('extern "C" __device__ int provider_a() { return 1; }\n', "sm_80"),
    ),
) -> tuple[Path, aot.CaptureResult]:
    output = tmp_path / output_name
    with aot.capture(output, name="test-pack") as captured:
        for source, arch in sources:
            _compile_source(provider_bundle, source, arch)
    return output, captured.result


def _manifest_payload(pack: Path) -> dict:
    return json.loads((pack / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(pack: Path, payload: dict) -> None:
    (pack / "manifest.json").write_bytes(_aot_pack._canonical_json_bytes(payload))


def test_non_linux_pack_operations_fail_before_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_aot_pack.sys, "platform", "win32")
    pack = tmp_path / "missing.coop-aot"
    output = tmp_path / "captured.coop-aot"

    with pytest.raises(aot.PackError, match="currently require Linux"):
        aot.inspect(pack)
    with pytest.raises(aot.PackError, match="currently require Linux"):
        with aot.use(pack, mode="required"):
            pass
    with pytest.raises(aot.PackError, match="currently require Linux"):
        with aot.capture(output):
            pass

    with aot.use(pack, mode="off") as selected:
        assert selected is None
    assert not output.exists()


def test_capture_is_canonical_relocatable_and_deduplicates_exact_bundles(
    tmp_path,
    provider_bundle,
):
    source_a = 'extern "C" __device__ int provider_a() { return 1; }\n'
    source_b = 'extern "C" __device__ long provider_b() { return 2; }\n'
    output, result = _capture_pack(
        tmp_path,
        provider_bundle,
        sources=(
            (source_b, "sm_90a"),
            (source_a, "sm_80"),
            (source_b, "sm_90a"),
        ),
    )

    assert result.path == output
    assert result.name == "test-pack"
    assert result.observations == 3
    assert len(result.entries) == 2
    assert [entry.entry_id for entry in result.entries] == sorted(
        entry.entry_id for entry in result.entries
    )
    assert {(entry.compute_arch, entry.sm_arch) for entry in result.entries} == {
        ("compute_80", "sm_80"),
        ("compute_90a", "sm_90a"),
    }
    assert all(
        entry.symbols == ("provider_a", "provider_z") for entry in result.entries
    )
    manifest_bytes = (output / "manifest.json").read_bytes()
    assert manifest_bytes == _aot_pack._canonical_json_bytes(json.loads(manifest_bytes))

    relocated = tmp_path / "relocated.coop-aot"
    output.rename(relocated)
    info = aot.inspect(relocated)

    assert info.path == relocated
    assert info.name == "test-pack"
    assert info.writer_version
    assert info.entries == result.entries
    assert info.artifact_bytes == result.artifact_bytes
    assert all("/" not in entry.entry_id for entry in info.entries)


def test_capture_result_and_pack_info_are_frozen(tmp_path, provider_bundle):
    output, result = _capture_pack(tmp_path, provider_bundle)
    info = aot.inspect(output)

    with pytest.raises(FrozenInstanceError):
        result.name = "changed"
    with pytest.raises(FrozenInstanceError):
        info.name = "changed"
    with pytest.raises(FrozenInstanceError):
        info.entries[0].sm_arch = "sm_90"


def _forked_aot_lock_worker(lock_name, pack, connection):
    if lock_name == "_PACK_CACHE_LOCK":
        _aot_pack._load_pack_cached(pack)
    else:
        artifact = b"forked-materialized-artifact"
        _aot_pack._materialize_artifact(hashlib.sha256(artifact).hexdigest(), artifact)
    connection.send("acquired")
    connection.close()


def _forked_capture_worker(inherited_capture, child_output, connection):
    from cuda.coop.cutlass._compiler import _bundle as _provider_bundle

    result = {
        "inherited_capture_cleared": _aot_pack._ACTIVE_CAPTURE.get() is None,
    }
    try:
        with aot.capture(child_output) as child_capture:
            _compile_source(
                _provider_bundle,
                'extern "C" __device__ void child_capture() {}\n',
            )
    except BaseException as exc:
        result["fresh_capture_error"] = (type(exc).__name__, str(exc))
    else:
        result["fresh_capture_error"] = None
        result["fresh_capture_observations"] = child_capture.result.observations

    try:
        inherited_capture.__exit__(None, None, None)
    except BaseException as exc:
        result["inherited_exit_error"] = (type(exc).__name__, str(exc))
    else:
        result["inherited_exit_error"] = None

    connection.send(result)
    connection.close()


@pytest.mark.parametrize(
    "lock_name",
    ["_PACK_CACHE_LOCK", "_MATERIALIZED_ARTIFACTS_LOCK"],
)
@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="test requires POSIX fork state reset",
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_process_locks_reset_after_fork(
    tmp_path,
    provider_bundle,
    lock_name,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    lock = getattr(_aot_pack, lock_name)
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_lock():
        with lock:
            lock_held.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    process = None
    parent_connection = None
    child_connection = None
    try:
        assert lock_held.wait(timeout=5)
        context = multiprocessing.get_context("fork")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_forked_aot_lock_worker,
            args=(lock_name, output, child_connection),
        )
        process.start()
        assert parent_connection.poll(5)
        assert parent_connection.recv() == "acquired"
        process.join(timeout=5)
    finally:
        release_lock.set()
        holder.join(timeout=5)
        if process is not None and process.is_alive():
            process.terminate()
        if process is not None:
            process.join(timeout=5)
        if parent_connection is not None:
            parent_connection.close()
        if child_connection is not None:
            child_connection.close()

    assert process is not None
    assert not holder.is_alive()
    assert not process.is_alive()
    assert process.exitcode == 0


@pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="test requires POSIX fork behavior",
)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_active_capture_detaches_before_child_capture(
    tmp_path,
    provider_bundle,
):
    parent_output = tmp_path / "parent.coop-aot"
    child_output = tmp_path / "child.coop-aot"
    process = None
    parent_connection = None
    child_connection = None
    with aot.capture(parent_output) as capture:
        try:
            context = multiprocessing.get_context("fork")
            parent_connection, child_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_forked_capture_worker,
                args=(capture, child_output, child_connection),
            )
            process.start()
            assert parent_connection.poll(5)
            result = parent_connection.recv()
            process.join(timeout=5)
        finally:
            if process is not None and process.is_alive():
                process.terminate()
            if process is not None:
                process.join(timeout=5)
            if parent_connection is not None:
                parent_connection.close()
            if child_connection is not None:
                child_connection.close()

        assert process is not None
        assert process.exitcode == 0
        expected = (
            "CaptureError",
            _aot_pack._CAPTURE_PROCESS_BOUNDARY_ERROR,
        )
        assert result["inherited_capture_cleared"]
        assert result["fresh_capture_error"] is None
        assert result["fresh_capture_observations"] == 1
        assert result["inherited_exit_error"] == expected
        _compile_source(
            provider_bundle,
            'extern "C" __device__ void parent_capture() {}\n',
        )

    assert capture.result.observations == 1
    assert len(aot.inspect(parent_output).entries) == 1
    assert len(aot.inspect(child_output).entries) == 1
    assert len(tuple((parent_output / "sources").iterdir())) == 1
    assert len(tuple((child_output / "sources").iterdir())) == 1


def test_symbol_sets_are_bound_into_distinct_coexisting_entry_ids(
    tmp_path,
    provider_bundle,
    monkeypatch,
):
    source = 'extern "C" __device__ void shared_source() {}\n'
    output = tmp_path / "symbols.coop-aot"

    with aot.capture(output) as captured:
        _compile_source(
            provider_bundle,
            source,
            symbols=("provider_a",),
        )
        _compile_source(
            provider_bundle,
            source,
            symbols=("provider_b",),
        )

    entries = captured.result.entries
    assert len(entries) == 2
    assert len({entry.entry_id for entry in entries}) == 2
    assert {entry.symbols for entry in entries} == {
        ("provider_a",),
        ("provider_b",),
    }
    assert len({entry.source_sha256 for entry in entries}) == 1
    assert len({entry.artifact_sha256 for entry in entries}) == 1
    assert len(tuple((output / "sources").iterdir())) == 1
    assert len(tuple((output / "artifacts").iterdir())) == 1

    observed = []
    provider_bundle.reset_compile_state()
    monkeypatch.setattr(
        _aot_pack,
        "_consumer_nvjitlink_version",
        lambda: _aot_pack._CudaVersion(major=13, minor=0),
    )
    with (
        aot.use(output, mode="required"),
        provider_bundle.activate_bundle_resolution_observer(observed.append),
    ):
        _compile_source(
            provider_bundle,
            source,
            symbols=("provider_a",),
        )
        _compile_source(
            provider_bundle,
            source,
            symbols=("provider_b",),
        )
    assert [resolution.route for resolution in observed] == [
        provider_contract.RESOLUTION_ROUTE_AOT_PACK,
        provider_contract.RESOLUTION_ROUTE_AOT_PACK,
    ]


def test_shared_artifact_rejects_conflicting_declared_sizes(
    tmp_path,
    provider_bundle,
):
    source = 'extern "C" __device__ void shared_source() {}\n'
    output = tmp_path / "shared-artifact-size.coop-aot"

    with aot.capture(output):
        _compile_source(provider_bundle, source, symbols=("provider_a",))
        _compile_source(provider_bundle, source, symbols=("provider_b",))

    manifest = _manifest_payload(output)
    assert len(manifest["entries"]) == 2
    assert (
        manifest["entries"][0]["artifact_sha256"]
        == manifest["entries"][1]["artifact_sha256"]
    )
    manifest["entries"][1]["artifact_size"] += 1
    _write_manifest(output, manifest)

    with pytest.raises(
        aot.PackIntegrityError,
        match="conflicting declared sizes",
    ):
        aot.inspect(output)


def test_same_identity_and_symbols_reject_conflicting_artifacts(
    tmp_path,
    provider_bundle,
):
    source = 'extern "C" __device__ void provider_a() {}\n'
    output = tmp_path / "conflict.coop-aot"
    fake_nvrtc = provider_nvrtc.cuda_nvrtc

    with pytest.raises(aot.CaptureError, match="Conflicting provider artifacts"):
        with aot.capture(output):
            first_path = _compile_source(
                provider_bundle,
                source,
                symbols=("provider_a",),
            )
            Path(first_path).unlink()
            provider_bundle.reset_compile_state()
            fake_nvrtc.blob = b"different-portable-ltoir"
            _compile_source(
                provider_bundle,
                source,
                symbols=("provider_a",),
            )

    assert not output.exists()


def test_capture_rejects_external_precompile_resolution_origin(
    tmp_path,
    provider_bundle,
):
    artifact = tmp_path / "external.ltoir"
    artifact.write_bytes(b"external-ltoir")
    output = tmp_path / "external.coop-aot"

    with pytest.raises(aot.CaptureError, match="trusted native-code inputs"):
        with aot.capture(output):
            with provider_bundle.activate_bundle_precompile_resolver(
                _precompiled_resolver(provider_bundle, artifact)
            ):
                _compile_source(provider_bundle, 'extern "C" {}')

    assert not output.exists()
    assert "trusted native-code" in aot.capture.__doc__
    assert "trusted native-code" in aot.inspect.__doc__


def test_nested_capture_is_rejected_without_disrupting_outer_capture(
    tmp_path,
    provider_bundle,
):
    outer_path = tmp_path / "outer.coop-aot"

    with aot.capture(outer_path) as outer:
        with pytest.raises(aot.CaptureError, match="Nested AOT capture"):
            with aot.capture(tmp_path / "inner.coop-aot"):
                pass
        _compile_source(provider_bundle, 'extern "C" {}')

    assert outer.result.path == outer_path
    assert len(outer.result.entries) == 1


def test_empty_and_exceptional_capture_publish_nothing(tmp_path):
    empty = tmp_path / "empty.coop-aot"
    with pytest.raises(aot.CaptureError, match="observed no provider bundles"):
        with aot.capture(empty):
            pass
    assert not empty.exists()

    failed = tmp_path / "failed.coop-aot"
    with pytest.raises(RuntimeError, match="user failure"):
        with aot.capture(failed):
            raise RuntimeError("user failure")
    assert not failed.exists()
    assert not tuple(tmp_path.glob(".*.staging-*"))


def test_capture_is_create_only(tmp_path, provider_bundle):
    output = tmp_path / "existing.coop-aot"
    output.mkdir()

    with pytest.raises(aot.CaptureError, match="already exists"):
        with aot.capture(output):
            pass

    assert list(output.iterdir()) == []


def test_concurrent_capture_publication_has_exactly_one_winner(
    tmp_path,
    provider_bundle,
):
    output = tmp_path / "raced.coop-aot"
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def worker() -> None:
        try:
            with aot.capture(output) as captured:
                _compile_source(provider_bundle, 'extern "C" {}')
                barrier.wait()
            results.append(captured.result)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], aot.CaptureError)
    assert "already exists" in str(errors[0])
    assert aot.inspect(output).entries == results[0].entries
    assert not tuple(tmp_path.glob(".*.staging-*"))


def test_parent_fsync_failure_does_not_report_a_published_pack_as_failed(
    tmp_path,
    provider_bundle,
    monkeypatch,
):
    original_fsync_directory = _aot_pack._fsync_directory

    def fail_parent_fsync(path):
        if path == tmp_path:
            raise OSError("injected parent fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(_aot_pack, "_fsync_directory", fail_parent_fsync)
    with pytest.warns(RuntimeWarning, match="was published atomically"):
        output, result = _capture_pack(
            tmp_path,
            provider_bundle,
            output_name="parent-fsync.coop-aot",
        )

    assert result.path == output
    assert aot.inspect(output).entries == result.entries
    assert not tuple(tmp_path.glob(".*.staging-*"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        strict_output, strict_result = _capture_pack(
            tmp_path,
            provider_bundle,
            output_name="strict-warning-filter.coop-aot",
        )
    assert strict_result.path == strict_output
    assert aot.inspect(strict_output).entries == strict_result.entries


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 2, "unsupported schema version"),
        ("provider_abi_version", 2, "unsupported provider ABI version"),
        ("format", "other", "unsupported format"),
    ],
)
def test_manifest_rejects_unsupported_schema_and_abi(
    tmp_path,
    provider_bundle,
    field,
    value,
    match,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    payload[field] = value
    _write_manifest(output, payload)

    with pytest.raises(aot.PackIntegrityError, match=match):
        aot.inspect(output)


def test_manifest_rejects_duplicate_entries_and_noncanonical_encoding(
    tmp_path,
    provider_bundle,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    payload["entries"].append(payload["entries"][0])
    _write_manifest(output, payload)

    with pytest.raises(aot.PackIntegrityError, match="unique sorted IDs"):
        aot.inspect(output)

    output, _ = _capture_pack(
        tmp_path,
        provider_bundle,
        output_name="noncanonical.coop-aot",
    )
    payload = _manifest_payload(output)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(aot.PackIntegrityError, match="not canonically encoded"):
        aot.inspect(output)


def test_manifest_rejects_duplicate_json_keys(tmp_path, provider_bundle):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = manifest.replace(
        '"format":"cuda.coop.cutlass.aot-pack"',
        '"format":"cuda.coop.cutlass.aot-pack","format":"duplicate"',
        1,
    )
    (output / "manifest.json").write_text(manifest, encoding="utf-8")

    with pytest.raises(aot.PackIntegrityError, match="valid canonical JSON"):
        aot.inspect(output)


@pytest.mark.parametrize(
    "kind",
    [
        "artifact-digest",
        "artifact-size",
        "artifact-oversize",
        "source",
        "source-oversize",
    ],
)
def test_pack_rejects_corrupt_content(tmp_path, provider_bundle, kind, monkeypatch):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    entry = payload["entries"][0]
    if kind == "artifact-digest":
        artifact = output / "artifacts" / f"{entry['artifact_sha256']}.ltoir"
        artifact.write_bytes(b"corrupt-ltoir")
    elif kind == "artifact-size":
        entry["artifact_size"] += 1
        _write_manifest(output, payload)
    elif kind == "artifact-oversize":
        entry["artifact_size"] -= 1
        _write_manifest(output, payload)
    elif kind == "source":
        source = output / "sources" / f"{entry['identity']['source_sha256']}.cu"
        source.write_text("corrupt source", encoding="utf-8")
    else:
        source = output / "sources" / f"{entry['identity']['source_sha256']}.cu"
        monkeypatch.setattr(_aot_pack, "MAX_SOURCE_BYTES", source.stat().st_size - 1)

    with pytest.raises(
        aot.PackIntegrityError,
        match="invalid (digest|size)|unexpectedly large",
    ):
        aot.inspect(output)


def test_capture_rejects_oversized_source(tmp_path, provider_bundle, monkeypatch):
    monkeypatch.setattr(_aot_pack, "MAX_SOURCE_BYTES", 1)

    with pytest.raises(aot.CaptureError, match="unexpectedly large"):
        _capture_pack(tmp_path, provider_bundle)

    assert not (tmp_path / "captured.coop-aot").exists()
    assert not tuple(tmp_path.glob(".*.staging-*"))


def test_pack_rejects_huge_declared_artifact_size_as_integrity_error(
    tmp_path,
    provider_bundle,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    payload["entries"][0]["artifact_size"] = sys.maxsize
    _write_manifest(output, payload)

    with pytest.raises(aot.PackIntegrityError, match="invalid size"):
        aot.inspect(output)


@pytest.mark.parametrize("kind", ["root", "artifact", "source"])
def test_pack_rejects_symlinks(tmp_path, provider_bundle, kind):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    entry = payload["entries"][0]
    inspected_path = output
    if kind == "root":
        inspected_path = tmp_path / "pack-link"
        inspected_path.symlink_to(output, target_is_directory=True)
    else:
        if kind == "artifact":
            path = output / "artifacts" / f"{entry['artifact_sha256']}.ltoir"
        else:
            path = output / "sources" / f"{entry['identity']['source_sha256']}.cu"
        replacement = tmp_path / f"{kind}.replacement"
        path.rename(replacement)
        path.symlink_to(replacement)

    with pytest.raises(aot.PackIntegrityError, match="(real directory|regular file)"):
        aot.inspect(inspected_path)


def test_pack_rejects_bad_types_and_unexpected_inventory(tmp_path, provider_bundle):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    artifact = (
        output / "artifacts" / f"{payload['entries'][0]['artifact_sha256']}.ltoir"
    )
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(aot.PackIntegrityError, match="regular file"):
        aot.inspect(output)

    output, _ = _capture_pack(
        tmp_path,
        provider_bundle,
        output_name="inventory.coop-aot",
    )
    (output / "unexpected").write_text("unexpected", encoding="utf-8")
    with pytest.raises(aot.PackIntegrityError, match="unexpected or missing"):
        aot.inspect(output)


def test_pack_rejects_conflicting_identity_and_layout_metadata(
    tmp_path,
    provider_bundle,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    entry = payload["entries"][0]
    entry["identity"]["layout_expressions"] = ["sizeof(Other)"]
    _write_manifest(output, payload)

    with pytest.raises(
        aot.PackIntegrityError,
        match="ID does not match its exact bundle identity",
    ):
        aot.inspect(output)


def test_pack_rejects_symbol_changes_not_bound_by_entry_id(
    tmp_path,
    provider_bundle,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    payload["entries"][0]["symbols"] = ["provider_other"]
    _write_manifest(output, payload)

    with pytest.raises(
        aot.PackIntegrityError,
        match="identity and symbol set",
    ):
        aot.inspect(output)


@pytest.mark.parametrize(
    ("compute_arch", "sm_arch"),
    [
        ("compute_80", "sm_90"),
        ("compute80", "sm_80"),
        ("compute_80", "sm80"),
        ("compute_80x", "sm_80x"),
    ],
)
def test_pack_rejects_invalid_or_mismatched_architecture_pairs(
    tmp_path,
    provider_bundle,
    compute_arch,
    sm_arch,
):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    identity = payload["entries"][0]["identity"]
    identity["compute_arch"] = compute_arch
    identity["sm_arch"] = sm_arch
    _write_manifest(output, payload)

    with pytest.raises(aot.PackIntegrityError, match="exact matching"):
        aot.inspect(output)


def test_pack_rejects_non_utf8_audit_source(tmp_path, provider_bundle):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    payload = _manifest_payload(output)
    entry = payload["entries"][0]
    invalid_source = b"\xff"
    digest = hashlib.sha256(invalid_source).hexdigest()
    old_source = output / "sources" / f"{entry['identity']['source_sha256']}.cu"
    old_source.unlink()
    (output / "sources" / f"{digest}.cu").write_bytes(invalid_source)
    entry["identity"]["source_sha256"] = digest
    entry["entry_id"] = _aot_pack._entry_id(
        _aot_pack._parse_identity(entry["identity"], entry_index=0),
        tuple(entry["symbols"]),
    )
    _write_manifest(output, payload)

    with pytest.raises(aot.PackIntegrityError, match="not valid UTF-8"):
        aot.inspect(output)


def test_relocated_copy_contains_no_caller_paths(tmp_path, provider_bundle):
    output, _ = _capture_pack(tmp_path, provider_bundle)
    manifest = (output / "manifest.json").read_text(encoding="utf-8")

    assert str(tmp_path) not in manifest
    assert "argv" not in manifest
    assert "environment" not in manifest
    assert "cubin" not in manifest
    assert "caller" not in manifest

    copied = tmp_path / "copied.coop-aot"
    shutil.copytree(output, copied)
    assert aot.inspect(copied).entries == aot.inspect(output).entries
