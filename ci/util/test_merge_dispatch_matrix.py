# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("merge_dispatch_matrix.py")
merge_matrix_files = runpy.run_path(str(SCRIPT))["merge_matrix_files"]


def write_matrix(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"include": entries}))
    return path


def test_single_matrix_preserves_ids_and_comment_headers(tmp_path):
    row = {"id": "headers", "comment_header": "compile-time-bench-headers"}
    path = write_matrix(tmp_path / "public" / "compile-time.json", [row])
    assert merge_matrix_files([f"public:{path}"]) == {"include": [row]}


def test_multiple_matrices_keep_distinct_rows_with_conflicting_ids(tmp_path):
    public = write_matrix(
        tmp_path / "public" / "compile-time.json",
        [
            {
                "id": "headers",
                "comment_header": "compile-time-bench-headers",
                "preset": "all-dev",
            }
        ],
    )
    auxiliary = write_matrix(
        tmp_path / "auxiliary" / "compile-time.json",
        [
            {
                "id": "headers",
                "comment_header": "compile-time-bench-headers",
                "preset": "cub-cpp20",
            },
            {"id": "headers-auxiliary", "preset": "thrust-cpp20"},
        ],
    )
    rows = merge_matrix_files([f"public:{public}", f"auxiliary:{auxiliary}"])["include"]
    assert [row["id"] for row in rows] == [
        "public.headers",
        "auxiliary.headers",
        "auxiliary.headers-auxiliary",
    ]
    assert [row.get("comment_header") for row in rows] == [
        "public.compile-time-bench-headers",
        "auxiliary.compile-time-bench-headers",
        None,
    ]
    assert [row["preset"] for row in rows] == ["all-dev", "cub-cpp20", "thrust-cpp20"]


def test_duplicate_rows(tmp_path):
    row = {"id": "sass", "project": "cub"}
    matrix_files = [
        f"{label}:{write_matrix(tmp_path / label / 'sass-diff.json', entries)}"
        for label, entries in [("public", [row]), ("empty", []), ("auxiliary", [row])]
    ]
    assert [entry["id"] for entry in merge_matrix_files(matrix_files)["include"]] == [
        "public.sass",
    ]


def test_ids_do_not_affect_deduplication(tmp_path):
    public_row = {
        "id": "headers",
        "comment_header": "public-comment",
        "preset": "cub-cpp20",
        "gpu": "t4",
    }
    auxiliary_row = {
        "gpu": "t4",
        "preset": "cub-cpp20",
        "comment_header": "auxiliary-comment",
        "id": "other-headers",
    }
    public = write_matrix(tmp_path / "public.json", [public_row])
    auxiliary = write_matrix(tmp_path / "auxiliary.json", [auxiliary_row])
    assert merge_matrix_files([f"public:{public}", f"auxiliary:{auxiliary}"]) == {
        "include": [
            {
                **public_row,
                "id": "public.headers",
                "comment_header": "public.public-comment",
            }
        ]
    }


def test_hash_collision(tmp_path):
    # Integer hashes repeat modulo this value. Distinct job contents must still
    # survive when their keys happen to have the same hash.
    retention_days = [1, 1 + sys.hash_info.modulus]
    matrix_files = [
        f"{label}:{write_matrix(tmp_path / f'{label}.json', [{'id': 'headers', 'artifact_retention_days': days}])}"
        for label, days in zip(["public", "auxiliary"], retention_days)
    ]
    rows = merge_matrix_files(matrix_files)["include"]
    assert [row["artifact_retention_days"] for row in rows] == retention_days


def test_hyphenated_matrix_names_cannot_collide_with_config_ids(tmp_path):
    paths = [
        write_matrix(
            tmp_path / "public" / "compile-time.json",
            [{"id": "auxiliary-headers", "preset": "cub-cpp20"}],
        ),
        write_matrix(
            tmp_path / "public-auxiliary" / "compile-time.json",
            [{"id": "headers", "preset": "thrust-cpp20"}],
        ),
    ]
    matrix_files = [
        f"{label}:{path}" for label, path in zip(["public", "public-auxiliary"], paths)
    ]
    assert {entry["id"] for entry in merge_matrix_files(matrix_files)["include"]} == {
        "public.auxiliary-headers",
        "public-auxiliary.headers",
    }


def test_explicit_labels_are_independent_of_file_and_directory_names(tmp_path):
    public = write_matrix(
        tmp_path / "first.json", [{"id": "headers", "preset": "cub-cpp20"}]
    )
    auxiliary = write_matrix(
        tmp_path / "second.json", [{"id": "headers", "preset": "thrust-cpp20"}]
    )
    rows = merge_matrix_files([f"public:{public}", f"auxiliary:{auxiliary}"])["include"]
    assert [row["id"] for row in rows] == ["public.headers", "auxiliary.headers"]


def test_unlabeled_files_use_their_input_index(tmp_path):
    row = {"id": "headers", "comment_header": "compile-time-bench-headers"}
    paths = [
        write_matrix(tmp_path / name, [{**row, "preset": preset}])
        for name, preset in [
            ("first.json", "cub-cpp20"),
            ("second.json", "thrust-cpp20"),
        ]
    ]
    rows = merge_matrix_files([str(path) for path in paths])["include"]
    assert [entry["id"] for entry in rows] == ["0.headers", "1.headers"]
    assert [entry["comment_header"] for entry in rows] == [
        "0.compile-time-bench-headers",
        "1.compile-time-bench-headers",
    ]
    mixed = merge_matrix_files([f"public:{paths[0]}", str(paths[1])])["include"]
    assert [entry["id"] for entry in mixed] == ["public.headers", "1.headers"]


def test_cli_defaults_to_stdout(tmp_path):
    public = write_matrix(
        tmp_path / "first.json", [{"id": "headers", "preset": "cub-cpp20"}]
    )
    auxiliary = write_matrix(
        tmp_path / "second.json", [{"id": "headers", "preset": "thrust-cpp20"}]
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), f"public:{public}", str(auxiliary)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "include": [
            {"id": "public.headers", "preset": "cub-cpp20"},
            {"id": "1.headers", "preset": "thrust-cpp20"},
        ]
    }


def test_absent_include_is_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    assert merge_matrix_files([str(path)]) == {"include": []}


def test_cli_writes_output_after_all_inputs_are_read(tmp_path):
    valid = write_matrix(tmp_path / "public" / "sass-diff.json", [{"id": "sass"}])
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    output = tmp_path / "combined" / "matrix.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(valid),
            str(invalid),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "JSONDecodeError" in result.stderr
    assert not output.exists()
    output.parent.mkdir()
    subprocess.run(
        [sys.executable, str(SCRIPT), str(valid), "--output", str(output)],
        check=True,
    )
    assert json.loads(output.read_text()) == {"include": [{"id": "sass"}]}
