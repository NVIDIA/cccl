# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate standalone wheel and source archives without importing them."""

from __future__ import annotations

import argparse
import email.parser
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_PACKAGE = "cuda/coop/cutlass/"
_HEADERS = _PACKAGE + "_headers/"
_LICENSES = {"LICENSE", "cub/LICENSE.TXT", "libcudacxx/LICENSE.TXT", "thrust/LICENSE"}
_REQUIRED = (
    {
        _PACKAGE + name
        for name in (
            "__init__.py",
            "__init__.pyi",
            "py.typed",
            "_thread_data.py",
            "_thread_data.pyi",
            "_temp_storage.py",
            "_temp_storage.pyi",
            "_compiler/_runtime.py",
            "_compiler/_storage.py",
            "_headers/__init__.py",
        )
    }
    | {
        _HEADERS + "include/" + name
        for name in (
            "cub/block/block_load.cuh",
            "cub/block/block_store.cuh",
            "cub/block/block_reduce.cuh",
            "cub/block/block_scan.cuh",
            "cub/warp/warp_load.cuh",
            "thrust/detail/raw_pointer_cast.h",
            "cuda/std/cstdint",
            "nv/target",
        )
    }
    | {_HEADERS + "cccl-bundle-provenance.json"}
)


def _one(names: set[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {suffix}: {matches}")
    return matches[0]


def _metadata(content: bytes, version: str | None) -> None:
    metadata = email.parser.BytesParser().parsebytes(content)
    if metadata.get("Name") != "cuda-coop-cutlass":
        raise SystemExit("Unexpected standalone distribution name")
    if version is not None and metadata.get("Version") != version:
        raise SystemExit(
            f"Expected distribution version {version}, got {metadata.get('Version')}"
        )
    if metadata.get("Requires-Python") != ">=3.10":
        raise SystemExit("The standalone package requires Python >=3.10")
    if set(metadata.get_all("Provides-Extra", [])) != {"cu13", "test"}:
        raise SystemExit("Unexpected standalone package extras")
    requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist", [])]
    names = {canonicalize_name(r.name) for r in requirements}
    if any(name == "cuda-coop" or "numba" in name for name in names):
        raise SystemExit("The standalone wheel includes a neutral/Numba dependency")
    base = {
        canonicalize_name(r.name): r
        for r in requirements
        if r.marker is None or r.marker.evaluate({"extra": ""})
    }
    required = {
        "nvidia-cutlass-dsl",
        "cuda-bindings",
        "cuda-core",
        "cuda-pathfinder",
        "numpy",
        "packaging",
        "typing-extensions",
    }
    if not required <= base.keys():
        raise SystemExit(f"Missing standalone dependencies: {required - base.keys()}")
    compiler = base["nvidia-cutlass-dsl"].specifier
    if (
        "cu13" not in base["nvidia-cutlass-dsl"].extras
        or "4.9.0.dev1" not in compiler
        or "4.9" not in compiler
        or "4.9.0.dev0" in compiler
        or "4.8.1" in compiler
        or "5.0" in compiler
    ):
        raise SystemExit("CUTLASS dependency must select the CUDA 13 4.9 integration")


def _provenance(content: bytes, source_commit: str | None) -> None:
    provenance = json.loads(content)
    if (
        set(provenance) != {"cccl_source_commit"}
        or not isinstance(provenance["cccl_source_commit"], str)
        or re.fullmatch(r"[0-9A-Za-z._+-]+", provenance["cccl_source_commit"]) is None
    ):
        raise SystemExit("Invalid bundled-header source revision")
    if source_commit is not None and provenance["cccl_source_commit"] != source_commit:
        raise SystemExit(
            f"Expected bundled headers from {source_commit}, got {provenance}"
        )


def validate_wheel(path: Path, version: str | None, source_commit: str | None) -> None:
    if not path.name.endswith("-py3-none-any.whl"):
        raise SystemExit("cuda-coop-cutlass must produce a py3-none-any wheel")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        _metadata(archive.read(_one(names, ".dist-info/METADATA")), version)
        metadata = email.parser.BytesParser().parsebytes(
            archive.read(_one(names, ".dist-info/WHEEL"))
        )
        if metadata.get("Root-Is-Purelib", "").lower() != "true" or metadata.get_all(
            "Tag", []
        ) != ["py3-none-any"]:
            raise SystemExit("Unexpected standalone wheel platform metadata")
        if missing := _REQUIRED - names:
            raise SystemExit(f"Missing wheel payload: {sorted(missing)}")
        for name in names:
            if name == "cuda/__init__.py" or (
                name.startswith("cuda/coop/") and not name.startswith(_PACKAGE)
            ):
                raise SystemExit(f"Unexpected namespace-parent payload: {name}")
            if any(
                s.lower() in {".so", ".dll", ".a", ".pyd", ".dylib", ".exe"}
                for s in PurePosixPath(name).suffixes
            ):
                raise SystemExit(f"Native binary in universal wheel: {name}")
        licenses = {
            n.split(".dist-info/licenses/", 1)[1]
            for n in names
            if ".dist-info/licenses/" in n
        }
        if missing := _LICENSES - licenses:
            raise SystemExit(f"Missing wheel licenses: {sorted(missing)}")
        _provenance(
            archive.read(_HEADERS + "cccl-bundle-provenance.json"), source_commit
        )


def validate_sdist(path: Path, version: str | None, source_commit: str | None) -> None:
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(m.name).parts[0] for m in members}
        if len(roots) != 1:
            raise SystemExit("The sdist must have exactly one root directory")
        root = roots.pop() + "/"
        names = {m.name[len(root) :] for m in members if m.isfile()}
        required = {
            "pyproject.toml",
            "CMakeLists.txt",
            "README.md",
            "PKG-INFO",
            "cccl-bundle-provenance.json",
            _PACKAGE + "__init__.py",
            "_cccl/cmake/CCCLInstallRules.cmake",
            "_cccl/cmake/install/cub.cmake",
            "_cccl/cub/cub/block/block_load.cuh",
            "_cccl/thrust/thrust/detail/raw_pointer_cast.h",
            "_cccl/libcudacxx/include/cuda/std/cstdint",
        } | {"_cccl/" + license for license in _LICENSES}
        if missing := required - names:
            raise SystemExit(f"Missing self-contained sdist inputs: {sorted(missing)}")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if ".." in parts or member.name.startswith("/"):
                raise SystemExit(f"Unsafe sdist path: {member.name}")
            if any(part in {".venv", "__pycache__", ".git"} for part in parts):
                raise SystemExit(f"Generated state in sdist: {member.name}")
        metadata = archive.extractfile(root + "PKG-INFO")
        provenance = archive.extractfile(root + "cccl-bundle-provenance.json")
        assert metadata is not None and provenance is not None
        _metadata(metadata.read(), version)
        _provenance(provenance.read(), source_commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-tag", help="Require the wheel and sdist for this tag"
    )
    parser.add_argument(
        "--source-commit", help="Require this CCCL commit in both header bundles"
    )
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()
    version = None
    if args.release_tag is not None:
        match = re.fullmatch(
            r"cuda-coop-cutlass-v([0-9]+\.[0-9]+\.[0-9]+(?:rc[0-9]+)?)",
            args.release_tag,
        )
        if not match:
            parser.error(
                "expected tag cuda-coop-cutlass-vX.Y.Z or cuda-coop-cutlass-vX.Y.ZrcN"
            )
        version = match.group(1)
        prefix = f"cuda_coop_cutlass-{version}"
        expected = sorted([f"{prefix}-py3-none-any.whl", f"{prefix}.tar.gz"])
        if sorted(path.name for path in args.artifacts) != expected:
            parser.error(f"expected release artifacts {expected}")
    if args.source_commit is not None and not re.fullmatch(
        "[0-9a-f]{40}", args.source_commit
    ):
        parser.error("expected a full Git commit SHA")
    for path in args.artifacts:
        if path.suffix == ".whl":
            validate_wheel(path, version, args.source_commit)
        elif path.name.endswith(".tar.gz"):
            validate_sdist(path, version, args.source_commit)
        else:
            raise SystemExit(f"Unsupported artifact: {path}")


if __name__ == "__main__":
    main()
