# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ACTION_DIR = Path(__file__).resolve().parent


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repository.git"
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.name", "CI test")
    git(repo, "config", "user.email", "ci@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "config", "core.hooksPath", "/dev/null")
    git(repo, "config", "uploadpack.allowFilter", "true")
    matrix = repo / "ci/matrix.yaml"
    matrix.parent.mkdir()
    matrix.write_text("standard matrix\n")
    configs = repo / ".devcontainer"
    (configs / "cuda13.4-gcc16").mkdir(parents=True)
    (configs / "cuda13.4-gcc16/devcontainer.json").write_text('{"image":"standard"}')
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "Standard configuration")
    sha = git(repo, "rev-parse", "HEAD")

    git(repo, "checkout", "--quiet", "-b", "ci-configs")
    (repo / "ci/extra matrix.yaml").write_text("extended matrix\n")
    (configs / "cuda13.4-gcc16/devcontainer.json").write_text('{"image":"extended"}')
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "Extended configuration")
    git(repo, "tag", "--no-sign", "matrix-configs")

    git(repo, "checkout", "--quiet", "-b", "without-configs")
    shutil.rmtree(configs)
    git(repo, "add", "-u")
    git(repo, "commit", "--quiet", "-m", "Matrix only")
    return repo, sha


def download(tmp_path, repository, **inputs):
    repo, sha = repository
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    # Replace GitHub URL resolution only; retain the real Git clone and checkout.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "assert sys.argv[1:3] == ['repo', 'clone']\n"
        "assert sys.argv[5:7] == ['--no-upstream', '--']\n"
        "os.execvp('git', ['git', 'clone', *sys.argv[7:], sys.argv[3] + '.git', sys.argv[4]])\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "outputs"
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-euo",
            "pipefail",
            "-c",
            action["runs"]["steps"][0]["run"],
        ],
        cwd=workspace,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GITHUB_ACTION_PATH": str(ACTION_DIR),
            "GITHUB_SERVER_URL": repo.parent.as_uri(),
            "GITHUB_REPOSITORY": repo.name.removesuffix(".git"),
            "GITHUB_SHA": sha,
            "GITHUB_OUTPUT": str(output),
            "RUNNER_TEMP": str(runner_temp),
            "INPUT_MATRIX_REF": "",
            "INPUT_MATRIX_PATH": "",
            "INPUT_CI_PROFILE": "",
            "CI_MATRICES": "",
        }
        | inputs,
        capture_output=True,
        text=True,
        check=False,
    )
    assert not list(runner_temp.iterdir()), result.stderr
    return result, workspace, output


def test_default_bundle(tmp_path, repository):
    result, workspace, output = download(tmp_path, repository)
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == "matrix_files=workflow/default/matrix.yaml"
    bundle = workspace / "workflow/default"
    assert (bundle / "matrix.yaml").read_text() == "standard matrix\n"
    assert json.loads(
        (bundle / ".devcontainer/cuda13.4-gcc16/devcontainer.json").read_text()
    ) == {"image": "standard"}


def test_profile_bundles_configs_from_each_ref(tmp_path, repository):
    _, sha = repository
    result, workspace, output = download(
        tmp_path,
        repository,
        INPUT_CI_PROFILE="nightly",
        CI_MATRICES=json.dumps(
            {
                "nightly": {
                    "standard": f"ci/matrix.yaml@{sha}",
                    "extended": "ci/extra matrix.yaml@ci-configs",
                }
            }
        ),
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == (
        "matrix_files=workflow/standard/matrix.yaml workflow/extended/matrix.yaml"
    )
    for name in ["standard", "extended"]:
        bundle = workspace / "workflow" / name
        assert (bundle / "matrix.yaml").read_text() == f"{name} matrix\n"
        assert json.loads(
            (bundle / ".devcontainer/cuda13.4-gcc16/devcontainer.json").read_text()
        ) == {"image": name}


@pytest.mark.parametrize("ref", ["matrix-configs", "without-configs"])
def test_explicit_matrix_and_optional_configs(tmp_path, repository, ref):
    result, workspace, output = download(
        tmp_path,
        repository,
        INPUT_MATRIX_REF=ref,
        INPUT_MATRIX_PATH="ci/extra matrix.yaml",
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == "matrix_files=workflow/manual/matrix.yaml"
    bundle = workspace / "workflow/manual"
    assert (bundle / "matrix.yaml").read_text() == "extended matrix\n"
    assert (bundle / ".devcontainer").exists() == (ref == "matrix-configs")


@pytest.mark.parametrize(
    "inputs",
    [
        {"INPUT_MATRIX_PATH": "missing.yaml"},
        {"INPUT_MATRIX_REF": "missing-ref"},
    ],
)
def test_failed_download_does_not_publish_outputs(tmp_path, repository, inputs):
    result, _, output = download(tmp_path, repository, **inputs)
    assert result.returncode != 0
    assert not output.exists()
