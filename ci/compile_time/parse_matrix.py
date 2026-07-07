#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None
    YAML_ERROR_TYPES: tuple[type[BaseException], ...] = ()
else:
    YAML_ERROR_TYPES = (yaml.YAMLError,)

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
TIMINGS = {"inclusive", "exclusive"}
SORTS = {"total", "avg", "avg-root-tu", "max"}


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        die(f"{where} must be a mapping")
    return value


def require_field(mapping: dict[str, Any], field: str, where: str) -> Any:
    if field not in mapping:
        die(f"{where} is missing required field '{field}'")
    return mapping[field]


def require_string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        die(f"{where} must be a string")
    if nonempty and not value:
        die(f"{where} must be non-empty")
    return value


def require_id(value: Any, where: str) -> str:
    text = require_string(value, where)
    if not ID_RE.fullmatch(text):
        die(f"{where} must match {ID_RE.pattern}")
    return text


def require_string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        die(f"{where} must be a non-empty list")
    strings: list[str] = []
    for index, item in enumerate(value):
        strings.append(require_string(item, f"{where}[{index}]"))
    return strings


def require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        die(f"{where} must be a boolean")
    return value


def require_positive_int(value: Any, where: str) -> int:
    if not isinstance(value, int) or value <= 0:
        die(f"{where} must be a positive integer")
    return value


def validate_slice(
    slice_data: Any,
    *,
    where: str,
    seen_ids: set[str],
) -> dict[str, Any]:
    data = require_mapping(slice_data, where)
    slice_id = require_id(require_field(data, "id", where), f"{where}.id")
    if slice_id in seen_ids:
        die(f"duplicate slice id '{slice_id}' in {where}")
    seen_ids.add(slice_id)

    title = require_string(require_field(data, "title", where), f"{where}.title")
    filter_name = require_string(
        require_field(data, "filter", where), f"{where}.filter"
    )
    timing = require_string(require_field(data, "timing", where), f"{where}.timing")
    if timing not in TIMINGS:
        die(f"{where}.timing must be one of {sorted(TIMINGS)}")
    sort = require_string(require_field(data, "sort", where), f"{where}.sort")
    if sort not in SORTS:
        die(f"{where}.sort must be one of {sorted(SORTS)}")

    top = require_field(data, "top", where)
    if not isinstance(top, int) or top <= 0:
        die(f"{where}.top must be a positive integer")
    threshold = require_field(data, "threshold", where)
    if not isinstance(threshold, (int, float)) or threshold < 0:
        die(f"{where}.threshold must be a non-negative number")

    result: dict[str, Any] = {
        "id": slice_id,
        "title": title,
        "filter": filter_name,
        "timing": timing,
        "sort": sort,
        "top": top,
        "threshold": threshold,
    }
    for optional in ("scope_filter", "exclusive_scope"):
        if optional in data:
            result[optional] = require_string(
                data[optional], f"{where}.{optional}", nonempty=False
            )

    children = data.get("children", [])
    if not isinstance(children, list):
        die(f"{where}.children must be a list")
    if children:
        result["children"] = [
            validate_slice(
                child,
                where=f"{where}.children[{index}]",
                seen_ids=seen_ids,
            )
            for index, child in enumerate(children)
        ]
    return result


def validate_config(
    config_data: Any, *, where: str, seen_ids: set[str]
) -> dict[str, Any]:
    data = require_mapping(config_data, where)
    config_id = require_id(require_field(data, "id", where), f"{where}.id")
    if config_id in seen_ids:
        die(f"duplicate compile_time config id '{config_id}'")
    seen_ids.add(config_id)

    targets = require_string_list(
        require_field(data, "targets", where), f"{where}.targets"
    )
    slices = require_field(data, "slices", where)
    if not isinstance(slices, list) or not slices:
        die(f"{where}.slices must be a non-empty list")

    slice_ids: set[str] = set()
    normalized_slices = [
        validate_slice(
            slice_data,
            where=f"{where}.slices[{index}]",
            seen_ids=slice_ids,
        )
        for index, slice_data in enumerate(slices)
    ]

    return {
        "id": config_id,
        "name": require_string(require_field(data, "name", where), f"{where}.name"),
        "gpu": require_string(require_field(data, "gpu", where), f"{where}.gpu"),
        "launch_args": require_string(
            require_field(data, "launch_args", where), f"{where}.launch_args"
        ),
        "baseline_ref": require_string(
            require_field(data, "baseline_ref", where), f"{where}.baseline_ref"
        ),
        "preset": require_string(
            require_field(data, "preset", where), f"{where}.preset"
        ),
        "targets": targets,
        "args": require_string(data.get("args", ""), f"{where}.args", nonempty=False),
        "comment": require_bool(data.get("comment", True), f"{where}.comment"),
        "artifact_retention_days": require_positive_int(
            data.get("artifact_retention_days", 14),
            f"{where}.artifact_retention_days",
        ),
        "slices": normalized_slices,
    }


def matrix_entry(config: dict[str, Any]) -> dict[str, Any]:
    config_id = config["id"]
    return {
        "id": config_id,
        "name": config["name"],
        "gpu": config["gpu"],
        "launch_args": config["launch_args"],
        "baseline_ref": config["baseline_ref"],
        "preset": config["preset"],
        "targets_json": json.dumps(config["targets"], separators=(",", ":")),
        "args": config["args"],
        "slices_json": json.dumps({"slices": config["slices"]}, separators=(",", ":")),
        "comment": str(config["comment"]).lower(),
        "artifact_retention_days": config["artifact_retention_days"],
        "comment_header": f"compile-time-bench-{config_id}",
    }


def parse_matrix(path: Path, workflow: str) -> dict[str, Any]:
    try:
        if yaml is not None:
            with path.open(encoding="utf-8") as f:
                matrix = yaml.safe_load(f) or {}
        else:
            completed = subprocess.run(
                ["yq", "-o=json", ".", path.as_posix()],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            matrix = json.loads(completed.stdout or "{}")
    except OSError as e:
        die(f"failed to read {path}: {e}")
    except subprocess.CalledProcessError as e:
        die(f"failed to parse {path} with yq: {e.stderr.strip()}")
    except json.JSONDecodeError as e:
        die(f"failed to decode {path} as JSON: {e}")
    except YAML_ERROR_TYPES as e:
        die(f"failed to parse {path}: {e}")

    compile_time = matrix.get("compile_time")
    if compile_time is None:
        return {"include": []}
    compile_time = require_mapping(compile_time, "compile_time")
    configs = compile_time.get(workflow, [])
    if configs is None:
        configs = []
    if not isinstance(configs, list):
        die(f"compile_time.{workflow} must be a list")
    if not configs:
        return {"include": []}

    seen_ids: set[str] = set()
    return {
        "include": [
            matrix_entry(
                validate_config(
                    config,
                    where=f"compile_time.{workflow}[{index}]",
                    seen_ids=seen_ids,
                )
            )
            for index, config in enumerate(configs)
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse ci/matrix.yaml compile_time entries for GitHub Actions."
    )
    parser.add_argument("matrix_yaml", type=Path)
    parser.add_argument("--workflow", default="pull_request")
    args = parser.parse_args()

    json.dump(parse_matrix(args.matrix_yaml, args.workflow), sys.stdout)
    print()


if __name__ == "__main__":
    main()
