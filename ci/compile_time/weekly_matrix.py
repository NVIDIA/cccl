#!/usr/bin/env python3

"""Build third-party compile-time jobs against the previous weekly run."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def previous_weekly_run(
    runs: list[dict[str, Any]], current_run: dict[str, Any], branch: str
) -> dict[str, Any]:
    current_time = timestamp(current_run["created_at"])
    candidates = [
        run
        for run in runs
        if run.get("event") == "schedule"
        and run.get("head_branch") == branch
        and run.get("id") != current_run["id"]
        and timestamp(run["created_at"]) < current_time
    ]
    if not candidates:
        raise ValueError(f"no earlier scheduled weekly run found for {branch}")
    result = max(candidates, key=lambda run: (timestamp(run["created_at"]), run["id"]))
    if not SHA_RE.fullmatch(result.get("head_sha", "")):
        raise ValueError("previous weekly run has no full commit SHA")
    return result


def expand_matrix(
    matrix: dict[str, Any],
    previous_sha: str,
    *,
    include_cccl: bool,
) -> dict[str, Any]:
    include = matrix["include"]
    expanded = (
        [config for config in include if config["project"] == "cccl"]
        if include_cccl
        else []
    )
    for config in include:
        if config["project"] == "cccl":
            continue
        expanded.append(
            {
                **config,
                "id": f"{config['id']}-previous-weekly",
                "name": f"{config['name']} vs previous weekly run",
                "baseline_ref": previous_sha,
            }
        )
    return {"include": expanded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--run-id", type=int, default=os.getenv("GITHUB_RUN_ID"))
    parser.add_argument("--branch", required=True)
    parser.add_argument("--include-cccl", action="store_true")
    args = parser.parse_args()

    matrix = json.load(sys.stdin)
    if not any(config["project"] != "cccl" for config in matrix["include"]):
        json.dump(
            expand_matrix(matrix, "", include_cccl=args.include_cccl),
            sys.stdout,
        )
        print()
        return

    if not args.repository or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository
    ):
        parser.error("--repository must be an owner/repository name")
    if not args.run_id:
        parser.error("--run-id is required")
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required")

    api_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")

    def get_json(path: str) -> Any:
        request = Request(
            f"{api_url}/{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "cccl-compile-time-bench",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
            except URLError:
                if attempt == 2:
                    raise
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    repository = args.repository
    current_run = get_json(f"repos/{repository}/actions/runs/{args.run_id}")
    weekly_runs = get_json(
        f"repos/{repository}/actions/workflows/ci-workflow-weekly.yml/runs?"
        + urlencode({"event": "schedule", "per_page": 100})
    )["workflow_runs"]
    previous = previous_weekly_run(weekly_runs, current_run, args.branch)
    result = expand_matrix(
        matrix,
        previous["head_sha"],
        include_cccl=args.include_cccl,
    )
    json.dump(result, sys.stdout, separators=(",", ":"))
    print()


if __name__ == "__main__":
    main()
