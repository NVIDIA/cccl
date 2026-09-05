# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

from cutlass.base_dsl.common import DSLRuntimeError

from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle


class _NvrtcResult:
    NVRTC_SUCCESS = 0


class _FakeNvrtc:
    nvrtcResult = _NvrtcResult

    def __init__(self):
        self.calls = []
        self.blob = b"fake-ltoir"

    def nvrtcCreateProgram(self, source, name, num_headers, headers, include_names):
        self.calls.append(("create", source))
        return 0, object()

    def nvrtcAddNameExpression(self, program, expression):
        self.calls.append(("add", expression))
        return (0,)

    def nvrtcCompileProgram(self, program, num_options, options):
        self.calls.append(("compile", tuple(options)))
        return (0,)

    def nvrtcGetLoweredName(self, program, expression):
        self.calls.append(("get_lowered", expression))
        decoded = expression.decode("utf-8")
        symbol_match = re.match(r"&([^<]+)<", decoded)
        assert symbol_match is not None
        symbol = symbol_match.group(1)
        if "LargeStorage" in decoded:
            size_in_bytes, alignment = 40, 8
        else:
            size_in_bytes, alignment = 20, 4
        return (
            0,
            f"_Z{len(symbol)}{symbol}ILy{size_in_bytes}ELy{alignment}EE".encode(),
        )

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
        "registered_headers": lambda: {},
        "select_bundle_format": lambda: "ltoir",
        "resolve_nvrtc_sm_arch": lambda: "sm_80",
        "resolve_nvrtc_arch": lambda: "compute_80",
    }


@pytest.fixture(autouse=True)
def _isolated_bundle_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    provider_bundle.reset_compile_state()
    yield
    provider_bundle.reset_compile_state()


def _layout_probes():
    return (
        provider_bundle.LayoutProbe(
            key="small-a",
            size_expression="sizeof(SmallStorage)",
            alignment_expression="alignof(SmallStorage)",
        ),
        provider_bundle.LayoutProbe(
            key="small-b",
            size_expression="sizeof(SmallStorage)",
            alignment_expression="alignof(SmallStorage)",
        ),
        provider_bundle.LayoutProbe(
            key="large",
            size_expression="sizeof(LargeStorage)",
            alignment_expression="alignof(LargeStorage)",
        ),
    )


@pytest.mark.parametrize("configured_arch", ["sm_103a", "sm_103f"])
def test_sm103_provider_target_resolvers_preserve_arch_suffix(configured_arch):
    def configured_gpu_arch():
        return configured_arch

    assert (
        provider_bundle.resolve_nvrtc_sm_arch(
            "cuda.coop.cutlass",
            configured_gpu_arch,
        )
        == configured_arch
    )
    assert provider_bundle.resolve_nvrtc_arch(
        "cuda.coop.cutlass",
        configured_gpu_arch,
    ) == configured_arch.replace("sm_", "compute_", 1)


def test_layout_probes_share_one_nvrtc_program(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)

    compilation = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )

    assert compilation.layouts == {
        "small-a": provider_bundle.StorageLayout(20, 4),
        "small-b": provider_bundle.StorageLayout(20, 4),
        "large": provider_bundle.StorageLayout(40, 8),
    }
    assert compilation.path.endswith(".ltoir")
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    call_names = [call[0] for call in fake_nvrtc.calls]
    assert call_names.count("add") == 2
    assert call_names.count("compile") == 1
    assert call_names.count("get_lowered") == 2
    assert max(
        i for i, name in enumerate(call_names) if name == "add"
    ) < call_names.index("compile")
    assert call_names.index("compile") < min(
        i for i, name in enumerate(call_names) if name == "get_lowered"
    )
    compiled_source = fake_nvrtc.calls[0][1].decode("utf-8")
    assert "template <unsigned long long Size" in compiled_source
    assert "__device__ unsigned char cuda_coop_layout_probe_" in compiled_source


def test_layout_metadata_survives_an_in_memory_cache_reset(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    first = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )
    metadata_path = provider_bundle._layout_metadata_path(first.path)
    with open(metadata_path, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    assert (
        metadata["version"]
        == provider_bundle.LAYOUT_METADATA_VERSION
        == provider_bundle.BUNDLE_METADATA_VERSION
        == 3
    )
    contract_digest = Path(first.path).stem.removeprefix("bundle_v3_")
    assert re.fullmatch(r"[0-9a-f]{64}", contract_digest)
    assert metadata["contract_digest"] == contract_digest
    assert metadata["cache_key"] == f"v3:{contract_digest}"
    assert len(metadata["layouts"]) == 2

    provider_bundle.reset_compile_state()
    second = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )

    assert second == first
    assert provider_bundle.get_nvrtc_compile_program_counter() == 0
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 1


def test_contract_digest_mismatch_is_recompiled(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    first = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )
    metadata_path = Path(provider_bundle._layout_metadata_path(first.path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["contract_digest"] = hashlib.sha256(b"different-contract").hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    provider_bundle.reset_compile_state()
    second = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )

    assert second == first
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2


def test_corrupt_layout_sidecar_is_recompiled(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    first = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )
    with open(
        provider_bundle._layout_metadata_path(first.path),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        metadata_file.write("not-json")

    provider_bundle.reset_compile_state()
    second = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )

    assert second == first
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2


def test_artifact_hash_mismatch_is_recompiled(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    first = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )
    Path(first.path).write_bytes(fake_nvrtc.blob[::-1])

    provider_bundle.reset_compile_state()
    second = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )

    assert second == first
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2


@pytest.mark.parametrize(
    "mutation",
    ["corrupt-in-place", "replace", "symlink", "unlink"],
)
def test_mutated_artifact_invalidates_in_memory_cache(monkeypatch, mutation):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    first = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )
    artifact = Path(first.path)
    if mutation == "corrupt-in-place":
        artifact.write_bytes(fake_nvrtc.blob[::-1])
    elif mutation == "unlink":
        artifact.unlink()
    else:
        replacement = artifact.with_name(f"{mutation}.ltoir")
        replacement.write_bytes(fake_nvrtc.blob[::-1])
        if mutation == "replace":
            replacement.replace(artifact)
        else:
            artifact.unlink()
            artifact.symlink_to(replacement)

    second = provider_bundle.compile_bundle_source_with_layouts(
        'extern "C" {}',
        layout_probes=_layout_probes(),
        **_compile_kwargs(),
    )

    assert second == first
    assert provider_bundle.get_nvrtc_compile_program_counter() == 2
    assert [call[0] for call in fake_nvrtc.calls].count("compile") == 2
    assert not artifact.is_symlink()
    assert artifact.read_bytes() == fake_nvrtc.blob


def test_existing_bundle_api_still_returns_a_path(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)

    path = provider_bundle.compile_bundle_source(
        'extern "C" {}',
        **_compile_kwargs(),
    )

    assert isinstance(path, str)
    assert path.endswith(".ltoir")
    assert not any(call[0] == "add" for call in fake_nvrtc.calls)


def test_layout_metadata_rejects_bitcode_without_compiling(monkeypatch):
    fake_nvrtc = _FakeNvrtc()
    monkeypatch.setattr(provider_bundle, "cuda_nvrtc", fake_nvrtc)
    kwargs = _compile_kwargs()
    kwargs["select_bundle_format"] = lambda: "bc"

    with pytest.raises(DSLRuntimeError, match="requires an NVRTC LTO-IR bundle"):
        provider_bundle.compile_bundle_source_with_layouts(
            'extern "C" {}',
            layout_probes=_layout_probes(),
            **kwargs,
        )

    assert fake_nvrtc.calls == []


def test_layout_probe_decoder_is_strict():
    symbol = "cuda_coop_layout_probe_test"
    expression = f"&{symbol}<(sizeof(Storage)), (alignof(Storage))>"

    assert provider_bundle._decode_layout_probe_name(
        f"_Z{len(symbol)}{symbol}ILy40ELy8EE",
        symbol=symbol,
        expression=expression,
    ) == provider_bundle.StorageLayout(40, 8)
    with pytest.raises(ValueError, match="unexpected lowered layout-probe name"):
        provider_bundle._decode_layout_probe_name(
            f"_Z{len(symbol)}{symbol}ILy40ELy8ELy2EE",
            symbol=symbol,
            expression=expression,
        )
    with pytest.raises(ValueError, match="Invalid storage layout"):
        provider_bundle._decode_layout_probe_name(
            f"_Z{len(symbol)}{symbol}ILy40ELy3EE",
            symbol=symbol,
            expression=expression,
        )


def test_real_nvrtc_layout_probe_and_disk_cache():
    source = """
struct alignas(8) Storage { unsigned char data[40]; };
extern "C" __device__ void provider_entry() {}
"""
    probe = provider_bundle.LayoutProbe(
        key="storage",
        size_expression="sizeof(Storage)",
        alignment_expression="alignof(Storage)",
    )

    first = provider_bundle.compile_bundle_source_with_layouts(
        source,
        layout_probes=(probe,),
        **_compile_kwargs(),
    )

    assert first.layouts == {"storage": provider_bundle.StorageLayout(40, 8)}
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1

    provider_bundle.reset_compile_state()
    second = provider_bundle.compile_bundle_source_with_layouts(
        source,
        layout_probes=(probe,),
        **_compile_kwargs(),
    )

    assert second == first
    assert provider_bundle.get_nvrtc_compile_program_counter() == 0
