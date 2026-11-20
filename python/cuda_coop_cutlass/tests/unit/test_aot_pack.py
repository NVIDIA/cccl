# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise pack publication, integrity checks, and exact lookup."""

import errno
import json
import shutil
from types import SimpleNamespace

import cutlass
import pytest

from cuda.bindings import nvjitlink
from cuda.coop.cutlass import _aot_pack, aot
from cuda.coop.cutlass._compiler._types import ScratchLayout

pytestmark = [pytest.mark.unit, pytest.mark.backend_cutlass]

_KEY = "a" * 64
_ARCH = "compute_90"
_BLOB = b"provider bytes"
_LAYOUTS = {"size and alignment probe": ScratchLayout(64, 16)}
_VERSION = (13, 2)


def record(blob=_BLOB):
    _aot_pack.record(_KEY, _ARCH, blob, _LAYOUTS, _VERSION)


@pytest.fixture
def pack(tmp_path, monkeypatch):
    monkeypatch.setattr(nvjitlink, "version", lambda: (13, 1))
    monkeypatch.setattr(
        cutlass, "CUDA_VERSION", SimpleNamespace(major=13, minor=3)
    )
    path = tmp_path / "pack"
    with aot.capture(path, name="reductions"):
        record()
        record()
    return path


def test_relocated_pack_snapshots_bytes_and_layouts(pack, tmp_path):
    relocated = tmp_path / "other" / "renamed"
    shutil.move(pack, relocated)
    info = aot.inspect(relocated)
    assert info.name == "reductions"
    assert info.artifact_bytes == len(_BLOB)
    assert [entry.key for entry in info.entries] == [_KEY]
    with aot.use(relocated, mode="required") as selected:
        assert selected == info
        shutil.rmtree(relocated)
        blob, layouts, version = _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS))
        assert (blob, layouts, version) == (_BLOB, _LAYOUTS, _VERSION)
        layouts.clear()
        with aot.use(relocated, mode="off"):
            assert not _aot_pack.active()
        assert _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS))[1] == _LAYOUTS
        with pytest.raises(aot.PackMissError, match="absent"):
            _aot_pack.lookup("b" * 64, _ARCH, tuple(_LAYOUTS))
    assert not _aot_pack.active()


@pytest.mark.parametrize("major, minor", [(13, 1), (14, 0)])
def test_incompatible_cute_toolchain_is_a_miss(monkeypatch, pack, major, minor):
    monkeypatch.setattr(nvjitlink, "version", lambda: _VERSION)
    monkeypatch.setattr(
        cutlass, "CUDA_VERSION", SimpleNamespace(major=major, minor=minor)
    )
    with aot.use(pack, mode="auto"):
        assert _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS)) is None
    with (
        aot.use(pack, mode="required"),
        pytest.raises(aot.PackMissError, match="incompatible"),
    ):
        _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS))


def test_failed_conflicting_and_empty_captures_leave_no_pack(tmp_path):
    path = tmp_path / "pack"
    with (
        pytest.raises(RuntimeError, match="workload failed"),
        aot.capture(path),
    ):
        record()
        raise RuntimeError("workload failed")
    with (
        pytest.raises(aot.CaptureError, match="Conflicting"),
        aot.capture(path),
    ):
        record()
        record(b"other bytes")
    with (
        pytest.raises(aot.CaptureError, match="no provider"),
        aot.capture(path),
    ):
        pass
    assert not path.exists()
    assert not _aot_pack.active()


def test_capture_preserves_directory_created_at_publication(tmp_path):
    destination = tmp_path / "new-pack"
    with (
        pytest.raises(aot.CaptureError, match="already exists"),
        aot.capture(destination),
    ):
        destination.mkdir()
        record()
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("rename", ["unsupported", "overwrites"])
def test_capture_checks_destination_filesystem_before_workload(
    tmp_path, monkeypatch, rename
):
    def unsupported(source, target):
        raise OSError(errno.EINVAL, "unsupported rename flag")

    def overwrites(source, target):
        if target.exists():
            target.rmdir()
        source.rename(target)

    monkeypatch.setattr(
        _aot_pack,
        "_publish",
        unsupported if rename == "unsupported" else overwrites,
    )
    destination = tmp_path / "new-parent" / "pack"
    with (
        pytest.raises(aot.CaptureError, match="atomic no-replace"),
        aot.capture(destination),
    ):
        pytest.fail("capture ran the workload before checking publication")
    assert list(destination.parent.iterdir()) == []


def test_capture_use_and_inspect_accept_home_relative_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    with aot.capture("~/pack"):
        record()
    with aot.use("~/pack") as selected:
        assert (
            selected == aot.inspect(tmp_path / "pack") == aot.inspect("~/pack")
        )


@pytest.mark.parametrize("corruption", ["artifact", "layout", "key", "digest"])
def test_pack_corruption_fails_closed_in_auto_mode(pack, corruption):
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if corruption == "artifact":
        (pack / f"{_KEY}.ltoir").write_bytes(b"!" * len(_BLOB))
    elif corruption == "layout":
        manifest["entries"][0]["layouts"][0][1] *= 2
    elif corruption == "key":
        manifest["entries"][0]["key"] = "../escape"
        del manifest["sha256"]
        manifest["sha256"] = _aot_pack._checksum(manifest)
    else:
        manifest["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(aot.PackIntegrityError), aot.use(pack, mode="auto"):
        pytest.fail("a corrupt pack was selected")


def test_environment_selection_and_explicit_override(monkeypatch, pack):
    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, str(pack))
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "  ")
    assert _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS))[0] == _BLOB
    assert _aot_pack.lookup("b" * 64, _ARCH, ()) is None
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "required")
    assert _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS))[0] == _BLOB
    with aot.use(pack, mode="off"):
        assert not _aot_pack.active()
    with pytest.raises(aot.PackMissError, match="absent"):
        _aot_pack.lookup("b" * 64, _ARCH, ())
    monkeypatch.delenv(_aot_pack.PACK_PATH_ENV)
    with pytest.raises(aot.PackMissError, match=_aot_pack.PACK_PATH_ENV):
        _aot_pack.active()
    with aot.use(pack):
        assert _aot_pack.lookup("b" * 64, _ARCH, ()) is None
    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, "relative")
    with pytest.raises(aot.PackError, match="absolute"):
        _aot_pack.active()


def test_missing_cute_toolchain_version_is_a_pack_miss(monkeypatch, pack):
    monkeypatch.delattr(cutlass, "CUDA_VERSION")
    with aot.use(pack):
        assert _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS)) is None
    with (
        aot.use(pack, mode="required"),
        pytest.raises(aot.PackMissError, match="cannot determine"),
    ):
        _aot_pack.lookup(_KEY, _ARCH, tuple(_LAYOUTS))
