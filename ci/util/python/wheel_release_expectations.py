# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Derive release wheel expectations from a generated CCCL workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class WheelExpectation:
    artifact_name: str
    operating_system: str
    architecture: str
    python_version: str

    @property
    def python_tag(self) -> str:
        match = re.fullmatch(r"(\d+)\.(\d+)", self.python_version)
        if match is None:
            raise RuntimeError(
                f"Unsupported release Python version: {self.python_version}"
            )
        return f"cp{match.group(1)}{match.group(2)}"

    def matches_compute_tags(self, tags: tuple[str, str, str]) -> bool:
        python_tag, abi_tag, platform_tag = tags
        if python_tag != self.python_tag or abi_tag != self.python_tag:
            return False

        if self.operating_system == "windows":
            expected_platform = {
                "amd64": "win_amd64",
                "arm64": "win_arm64",
            }.get(self.architecture)
            if expected_platform is None:
                raise RuntimeError(
                    f"Unsupported Windows release architecture: {self.architecture}"
                )
            return platform_tag == expected_platform

        if self.operating_system != "linux":
            raise RuntimeError(
                f"Unsupported release operating system: {self.operating_system}"
            )

        platform_architecture = {
            "amd64": "x86_64",
            "arm64": "aarch64",
        }.get(self.architecture)
        if platform_architecture is None:
            raise RuntimeError(
                f"Unsupported Linux release architecture: {self.architecture}"
            )

        # auditwheel may emit more than one equivalent manylinux platform tag.
        manylinux_tag = re.compile(
            rf"manylinux(?:1|2010|2014|_[0-9]+_[0-9]+)_"
            rf"{re.escape(platform_architecture)}"
        )
        return all(
            manylinux_tag.fullmatch(tag) is not None for tag in platform_tag.split(".")
        )


def wheel_compatibility_tags(wheel: Path) -> tuple[str, str, str]:
    components = wheel.name.removesuffix(".whl").rsplit("-", 3)
    if len(components) != 4 or not all(components[-3:]):
        raise RuntimeError(f"Unable to parse wheel compatibility tags: {wheel.name}")
    return components[-3], components[-2], components[-1]


def _walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def load_release_expectations(workflow_file: Path) -> tuple[WheelExpectation, ...]:
    with workflow_file.open(encoding="utf-8") as stream:
        workflow = json.load(stream)

    expectations: dict[str, WheelExpectation] = {}
    for job in _walk_json(workflow):
        origin = job.get("origin")
        if (
            not isinstance(origin, dict)
            or origin.get("workflow_name") != "python-wheels"
        ):
            continue
        matrix_job = origin.get("matrix_job")
        if not isinstance(matrix_job, dict):
            continue
        jobs = matrix_job.get("jobs")
        if matrix_job.get("project") != "python" or not isinstance(jobs, list):
            continue
        if "build_py_wheel" not in jobs:
            continue

        architecture = matrix_job.get("cpu")
        python_version = matrix_job.get("py_version")
        compiler_family = matrix_job.get("cxx_family")
        runner = job.get("runner")
        if not all(
            isinstance(field, str)
            for field in (architecture, python_version, compiler_family, runner)
        ):
            raise RuntimeError(f"Incomplete Python wheel producer in {workflow_file}")

        if compiler_family == "MSVC":
            operating_system = "windows"
        elif runner.startswith("linux-"):
            operating_system = "linux"
        else:
            raise RuntimeError(
                f"Unable to determine the release OS for producer {job.get('name')!r}"
            )

        artifact_name = (
            f"wheel-cccl-{operating_system}-{architecture}-py{python_version}"
        )
        expectation = WheelExpectation(
            artifact_name=artifact_name,
            operating_system=operating_system,
            architecture=architecture,
            python_version=python_version,
        )
        if artifact_name in expectations:
            raise RuntimeError(
                f"Duplicate release producer for artifact {artifact_name}"
            )
        expectations[artifact_name] = expectation

    if not expectations:
        raise RuntimeError(
            f"No standard Python wheel producers found in {workflow_file}"
        )
    return tuple(sorted(expectations.values()))
