# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("resolve-matrices.py")
SHA = "0123456789abcdef0123456789abcdef01234567"


def resolve(profile, config, *, stdin=False):
    command = [
        sys.executable,
        str(SCRIPT),
        f"--profile={profile}",
        f"--default-ref={SHA}",
        "--default-path=ci/matrix.yaml",
    ]
    if not stdin:
        command.append("--ci-matrices=" + json.dumps(config))
    return subprocess.run(
        command,
        input=json.dumps(config) if stdin else None,
        capture_output=True,
        text=True,
        check=False,
    )


def test_selected_profile():
    result = resolve(
        "nightly",
        {
            "nightly": {
                "standard": "ci/matrix.yaml",
                "extended": "ci/extended.yaml@ci-configs",
            },
            "weekly": {"other": "ci/weekly.yaml"},
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"standard {SHA} ci/matrix.yaml",
        "extended ci-configs ci/extended.yaml",
    ]


def test_different_refs():
    result = resolve(
        "nightly",
        {
            "nightly": {
                "standard": "ci/matrix.yaml",
                "extended": "ci/matrix.yaml@ci-configs",
            },
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"standard {SHA} ci/matrix.yaml",
        "extended ci-configs ci/matrix.yaml",
    ]


def test_definition_defaults():
    result = resolve(
        "nightly",
        {
            "nightly": {
                "full": "ci/full.yaml@other",
                "path": "ci/path.yaml",
                "ref": "@other",
                "empty": "",
            }
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "full other ci/full.yaml",
        f"path {SHA} ci/path.yaml",
        "ref other ci/matrix.yaml",
        f"empty {SHA} ci/matrix.yaml",
    ]


@pytest.mark.parametrize("profile", ["missing", "empty"])
def test_missing_profile(profile):
    result = resolve(profile, {"empty": {}, "other": {"ignored": ""}})
    assert result.returncode == 0 and result.stdout == ""


def test_stdin_and_leading_dash():
    result = resolve("-nightly", {"-nightly": {"standard": ""}}, stdin=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"standard {SHA} ci/matrix.yaml\n"


@pytest.mark.parametrize(
    "config", [None, [], {"nightly": []}, {"nightly": {"standard": 42}}]
)
def test_invalid_config(config):
    result = resolve("nightly", config)
    assert result.returncode != 0
    assert "CI_MATRICES must be a JSON object" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("profile", ["../nightly", "nightly weekly"])
def test_invalid_profile(profile):
    result = resolve(profile, {})
    assert result.returncode != 0 and "Invalid profile name" in result.stderr
    assert result.stdout == ""


def test_invalid_name():
    result = resolve(
        "nightly",
        {"nightly": {"standard": "", "bad.name": "ci/extended.yaml"}},
    )
    assert result.returncode != 0 and "Invalid name" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("weekly", [{"shared": "ci/b.yaml"}, {"other": "ci/a.yaml"}])
def test_unselected_collision(weekly):
    result = resolve(
        "nightly",
        {"nightly": {"shared": "ci/a.yaml"}, "weekly": weekly},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"shared {SHA} ci/a.yaml\n"


@pytest.mark.parametrize(
    "definitions",
    [
        {"standard": "ci/matrix.yaml", "mirror": "ci/matrix.yaml"},
        {"standard": "", "mirror": f"@{SHA}"},
        {"standard": "ci/matrix.yaml", "mirror": f" ci/matrix.yaml @ {SHA} "},
    ],
)
def test_file_conflicts(definitions):
    result = resolve("nightly", {"nightly": definitions})
    assert result.returncode != 0
    assert "Conflicting matrix names" in result.stderr
    assert result.stdout == ""
