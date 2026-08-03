#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wheel(
    path: Path,
    entries: dict[str, str],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, contents in entries.items():
            archive.writestr(name, contents)


def _write_installable_wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
    payload: dict[str, str],
    requirements: tuple[str, ...] = (),
) -> None:
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {distribution}",
        f"Version: {version}",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        "",
    ]
    entries = {
        **payload,
        f"{dist_info}/METADATA": "\n".join(metadata),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: cccl-wheel-script-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = "".join(f"{name},,\n" for name in entries)
    entries[f"{dist_info}/RECORD"] = record + f"{dist_info}/RECORD,,\n"
    _write_wheel(path, entries)


def _write_coordinated_wheel_set(
    wheelhouse: Path, version: str, compute_wheel_names: tuple[str, ...]
) -> None:
    _write_wheel(
        wheelhouse / f"cccl_headers-{version}-py3-none-any.whl",
        {
            f"cccl_headers-{version}.dist-info/METADATA": (
                f"Name: cccl-headers\nVersion: {version}\n"
            ),
            "cuda/cccl/__init__.py": "",
            "cuda/cccl/headers/include/cub/version.cuh": "",
            "cuda/cccl/headers/include/thrust/version.h": "",
            "cuda/cccl/headers/include/cuda/version": "",
            "cuda/cccl/headers/include/cuda/experimental/coop.cuh": "",
        },
    )
    for wheel_name in compute_wheel_names:
        _write_wheel(
            wheelhouse / wheel_name,
            {
                f"cuda_compute-{version}.dist-info/METADATA": (
                    f"Name: cuda-compute\nVersion: {version}\n"
                    f"Requires-Dist: cccl-headers=={version}\n"
                ),
                "cuda/compute/__init__.py": "",
            },
        )
    _write_wheel(
        wheelhouse / f"cuda_cccl-{version}-py3-none-any.whl",
        {
            f"cuda_cccl-{version}.dist-info/METADATA": (
                f"Name: cuda-cccl\nVersion: {version}\n"
                f"Requires-Dist: cuda-compute=={version}\n"
            )
        },
    )


def _write_workflow(path: Path, producers: tuple[tuple[str, str, str], ...]) -> None:
    producer_jobs = []
    for operating_system, architecture, python_version in producers:
        compiler_family = "MSVC" if operating_system == "windows" else "GCC"
        producer_jobs.append(
            {
                "name": f"Build {operating_system} {architecture} py{python_version}",
                "runner": f"{operating_system}-{architecture}-cpu16",
                "origin": {
                    "workflow_name": "python-wheels",
                    "matrix_job": {
                        "project": "python",
                        "jobs": ["build_py_wheel"],
                        "cpu": architecture,
                        "py_version": python_version,
                        "cxx_family": compiler_family,
                    },
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "Python wheels": {
                    "two_stage": [
                        {
                            "producers": producer_jobs,
                            "consumers": [],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def _write_release_artifact(
    root: Path,
    operating_system: str,
    architecture: str,
    python_version: str,
    version: str = "1.2.3",
) -> None:
    artifact = root / f"wheel-cccl-{operating_system}-{architecture}-py{python_version}"
    python_tag = f"cp{python_version.replace('.', '')}"
    platform_tag = {
        ("linux", "amd64"): "manylinux_2_17_x86_64.manylinux2014_x86_64",
        ("linux", "arm64"): "manylinux_2_28_aarch64",
        ("windows", "amd64"): "win_amd64",
    }[(operating_system, architecture)]
    _write_wheel(
        artifact / f"cccl_headers-{version}-py3-none-any.whl",
        {f"cccl_headers-{version}.dist-info/METADATA": f"Version: {version}\n"},
    )
    _write_wheel(
        artifact
        / f"cuda_compute-{version}-{python_tag}-{python_tag}-{platform_tag}.whl",
        {f"cuda_compute-{version}.dist-info/METADATA": f"Version: {version}\n"},
    )
    _write_wheel(
        artifact / f"cuda_cccl-{version}-py3-none-any.whl",
        {f"cuda_cccl-{version}.dist-info/METADATA": f"Version: {version}\n"},
    )


class WheelScriptTests(unittest.TestCase):
    collector = None
    validator = None

    def test_collects_logically_identical_universal_wheels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel_name = "cccl_headers-1.2.3-py3-none-any.whl"
            canonical = root / "wheel-cccl-linux-amd64-py3.10" / wheel_name
            duplicate = root / "wheel-cccl-windows-amd64-py3.11" / wheel_name
            entries = {
                "cccl_headers-1.2.3.dist-info/METADATA": "Version: 1.2.3\n",
                "cuda/cccl/__init__.py": "",
            }
            _write_wheel(canonical, entries)
            _write_wheel(
                duplicate,
                dict(reversed(entries.items())),
                compression=zipfile.ZIP_DEFLATED,
            )

            self.assertNotEqual(canonical.read_bytes(), duplicate.read_bytes())

            destination = root / "dist"
            self.collector.collect_wheels(
                root, destination, "wheel-cccl-linux-amd64-py3.10"
            )

            self.assertEqual(
                (destination / wheel_name).read_bytes(), canonical.read_bytes()
            )

    def test_rejects_conflicting_duplicate_wheels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel_name = "cccl_headers-1.2.3-py3-none-any.whl"
            # Keep the canonical artifact lexically after the duplicate to
            # exercise comparison independent of candidate ordering.
            canonical_artifact = "wheel-cccl-windows-amd64-py3.10"
            _write_wheel(
                root / canonical_artifact / wheel_name,
                {"cccl_headers-1.2.3.dist-info/METADATA": "Version: 1.2.3\n"},
            )
            _write_wheel(
                root / "wheel-cccl-linux-amd64-py3.10" / wheel_name,
                {"cccl_headers-1.2.3.dist-info/METADATA": "Version: 1.2.4\n"},
            )

            with self.assertRaisesRegex(RuntimeError, "Conflicting wheels"):
                self.collector.collect_wheels(root, root / "dist", canonical_artifact)

    def test_rejects_universal_wheel_without_canonical_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel_name = "cuda_cccl-1.2.3-py3-none-any.whl"
            _write_wheel(
                root / "wheel-cccl-windows-amd64-py3.10" / wheel_name,
                {"cuda_cccl-1.2.3.dist-info/METADATA": "Version: 1.2.3\n"},
            )

            with self.assertRaisesRegex(RuntimeError, "Canonical artifact"):
                self.collector.collect_wheels(
                    root, root / "dist", "wheel-cccl-linux-amd64-py3.10"
                )

    def test_excludes_nonrelease_wheel_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            release_wheel = "cuda_compute-1.2.3-cp312-cp312-linux_x86_64.whl"
            _write_wheel(
                root / "wheel-cccl-linux-amd64-py3.12" / release_wheel,
                {"release": "wheel"},
            )
            for artifact in (
                "wheel-cccl-v2-linux-amd64-py3.12",
                "wheel-cccl-tsan-linux-amd64-py3.12",
                "wheel-other-linux-amd64-py3.12",
            ):
                _write_wheel(root / artifact / f"{artifact}.whl", {"excluded": "wheel"})

            destination = root / "dist"
            self.collector.collect_wheels(
                root, destination, "wheel-cccl-linux-amd64-py3.10"
            )

            self.assertEqual(
                [path.name for path in destination.glob("*.whl")], [release_wheel]
            )

    def test_collects_complete_generated_workflow_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            producers = (
                ("linux", "amd64", "3.10"),
                ("linux", "arm64", "3.11"),
                ("windows", "amd64", "3.12"),
            )
            workflow = root / "workflow.json"
            _write_workflow(workflow, producers)
            for producer in producers:
                _write_release_artifact(root, *producer)

            destination = root / "dist"
            self.collector.collect_wheels(
                root,
                destination,
                "wheel-cccl-linux-amd64-py3.10",
                workflow,
            )

            self.assertEqual(
                len(list(destination.glob("cuda_compute-*.whl"))), len(producers)
            )
            self.assertEqual(len(list(destination.glob("cccl_headers-*.whl"))), 1)
            self.assertEqual(len(list(destination.glob("cuda_cccl-*.whl"))), 1)

    def test_rejects_missing_generated_workflow_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflow = root / "workflow.json"
            _write_workflow(
                workflow,
                (
                    ("linux", "amd64", "3.10"),
                    ("windows", "amd64", "3.10"),
                ),
            )
            _write_release_artifact(root, "linux", "amd64", "3.10")

            with self.assertRaisesRegex(RuntimeError, "missing=.*windows"):
                self.collector.collect_wheels(
                    root,
                    root / "dist",
                    "wheel-cccl-linux-amd64-py3.10",
                    workflow,
                )

    def test_validates_coordinated_wheel_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse = Path(temp)
            version = "1.2.3"
            _write_coordinated_wheel_set(
                wheelhouse,
                version,
                (
                    f"cuda_compute-{version}-cp312-cp312-linux_x86_64.whl",
                    f"cuda_compute-{version}-cp312-cp312-win_amd64.whl",
                ),
            )

            workflow = wheelhouse / "workflow.json"
            _write_workflow(
                workflow,
                (
                    ("linux", "amd64", "3.12"),
                    ("windows", "amd64", "3.12"),
                ),
            )

            self.validator.validate(wheelhouse, workflow)

    def test_rejects_missing_compute_wheel_from_generated_workflow(self):
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse = Path(temp)
            version = "1.2.3"
            _write_coordinated_wheel_set(
                wheelhouse,
                version,
                (f"cuda_compute-{version}-cp312-cp312-linux_x86_64.whl",),
            )
            workflow = wheelhouse / "workflow.json"
            _write_workflow(
                workflow,
                (
                    ("linux", "amd64", "3.12"),
                    ("linux", "arm64", "3.12"),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "Missing cuda-compute wheels"):
                self.validator.validate(wheelhouse, workflow)

    def test_rejects_duplicate_compute_compatibility_tags(self):
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse = Path(temp)
            version = "1.2.3"
            _write_coordinated_wheel_set(
                wheelhouse,
                version,
                (
                    f"cuda_compute-{version}-1-cp312-cp312-linux_x86_64.whl",
                    f"cuda_compute-{version}-2-cp312-cp312-linux_x86_64.whl",
                ),
            )

            with self.assertRaisesRegex(
                RuntimeError, "Duplicate cuda-compute compatibility tags"
            ):
                self.validator.validate(wheelhouse)

    def test_rejects_unexpected_release_wheel(self):
        with tempfile.TemporaryDirectory() as temp:
            wheelhouse = Path(temp)
            version = "1.2.3"
            _write_coordinated_wheel_set(
                wheelhouse,
                version,
                (f"cuda_compute-{version}-cp312-cp312-linux_x86_64.whl",),
            )
            _write_wheel(
                wheelhouse / f"cuda_coop-{version}-py3-none-any.whl",
                {f"cuda_coop-{version}.dist-info/METADATA": "Name: cuda-coop\n"},
            )

            with self.assertRaisesRegex(RuntimeError, "Unexpected wheels"):
                self.validator.validate(wheelhouse)

    def test_monolithic_to_split_migration_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheelhouse = root / "wheelhouse"
            old_version = "1.0.0"
            new_version = "2.0.0"

            old_meta = wheelhouse / f"cuda_cccl-{old_version}-py3-none-any.whl"
            _write_installable_wheel(
                old_meta,
                distribution="cuda-cccl",
                version=old_version,
                payload={
                    "cuda/compute/__init__.py": "OLD_COMPUTE = True\n",
                    "cuda/compute/legacy_only.py": "LEGACY_ONLY = True\n",
                    "cuda/cccl/__init__.py": "OLD_HEADERS = True\n",
                    "cuda/cccl/legacy_only.py": "LEGACY_ONLY = True\n",
                },
            )
            _write_installable_wheel(
                wheelhouse / f"cccl_headers-{new_version}-py3-none-any.whl",
                distribution="cccl-headers",
                version=new_version,
                payload={"cuda/cccl/__init__.py": "NEW_HEADERS = True\n"},
            )
            _write_installable_wheel(
                wheelhouse / f"cuda_compute-{new_version}-py3-none-any.whl",
                distribution="cuda-compute",
                version=new_version,
                payload={"cuda/compute/__init__.py": "NEW_COMPUTE = True\n"},
                requirements=(f"cccl-headers=={new_version}",),
            )
            new_meta = wheelhouse / f"cuda_cccl-{new_version}-py3-none-any.whl"
            _write_installable_wheel(
                new_meta,
                distribution="cuda-cccl",
                version=new_version,
                payload={},
                requirements=(f"cuda-compute=={new_version}",),
            )

            environment = root / "environment"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / (
                "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
            )

            subprocess.run(
                [python, "-m", "pip", "install", "--no-index", old_meta],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [python, "-m", "pip", "uninstall", "-y", "cuda-cccl"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    python,
                    "-c",
                    (
                        "import importlib.util; "
                        "assert importlib.util.find_spec('cuda.compute') is None; "
                        "assert importlib.util.find_spec('cuda.cccl') is None"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    wheelhouse,
                    new_meta,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    python,
                    "-c",
                    (
                        "import cuda.cccl, cuda.compute; "
                        "assert cuda.cccl.NEW_HEADERS; "
                        "assert cuda.compute.NEW_COMPUTE; "
                        "assert not hasattr(cuda.cccl, 'OLD_HEADERS'); "
                        "assert not hasattr(cuda.compute, 'OLD_COMPUTE'); "
                        "import importlib.util; "
                        "assert importlib.util.find_spec("
                        "'cuda.cccl.legacy_only') is None; "
                        "assert importlib.util.find_spec("
                        "'cuda.compute.legacy_only') is None"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [python, "-m", "pip", "check"],
                check=True,
                capture_output=True,
                text=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collector", required=True, type=Path)
    parser.add_argument("--validator", required=True, type=Path)
    args, unittest_args = parser.parse_known_args()
    sys.path.insert(0, str(args.collector.parent))
    WheelScriptTests.collector = _load_module("wheel_collector", args.collector)
    WheelScriptTests.validator = _load_module("wheel_validator", args.validator)
    unittest.main(argv=[__file__, *unittest_args])


if __name__ == "__main__":
    main()
