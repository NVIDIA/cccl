#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path


def extract_jobs(workflow):
    jobs = []
    for group in workflow.values():
        if "standalone" in group:
            jobs += group["standalone"]
        if "two_stage" in group:
            for two_stage in group["two_stage"]:
                jobs += two_stage["producers"]
                jobs += two_stage["consumers"]
    return jobs


def _resolve_job_name_and_url(name: str, job_urls: dict[str, str]) -> tuple[str, str]:
    """Return a display name and URL for a job.

    Attempts exact match by name; otherwise, chooses the longest GitHub job
    name that contains the provided name (to include matrix details).
    """

    if name in job_urls:
        return name, job_urls[name]
    candidates = [n for n in job_urls.keys() if n.startswith(name) or name in n]
    if candidates:
        gh_name = max(candidates, key=len)
        return gh_name, job_urls.get(gh_name, "")
    return name, ""


def _first_match_via_parser(parser: Path, log_path: Path) -> dict | None:
    """Run parse_error.py on a log and return the first JSON match, if any."""

    if not log_path.exists():
        return None
    cmd = [
        sys.executable,
        str(parser),
        "-n",
        "1",
        "--format",
        "json",
        str(log_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data:
        return data[0]
    return None


def _generate_job_id_map(workflow: dict) -> dict[str, str]:
    """Map full GitHub job name to custom job id from workflow.json.

    Mirrors logic in parse-job-times to keep name construction consistent.
    """
    job_id_map: dict[str, str] = {}
    for group_name, group_json in workflow.items():
        standalone = group_json.get("standalone", [])
        for job in standalone:
            name = f"{group_name} / {job['name']}"
            job_id_map[name] = job["id"]
        for pc in group_json.get("two_stage", []):
            for job in pc.get("producers", []) + pc.get("consumers", []):
                name = f"{group_name} / {pc['id']} / {job['name']}"
                job_id_map[name] = job["id"]
    return job_id_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_json")
    parser.add_argument("jobs_json")
    parser.add_argument("jobs_dir")
    args = parser.parse_args()

    with open(args.workflow_json) as f:
        workflow = json.load(f)
    with open(args.jobs_json) as f:
        gha_jobs = json.load(f)

    # Map GH job name (includes matrix) to job info and URL
    job_by_name = {job["name"]: job for job in gha_jobs}

    # Build mapping from job id -> info
    jobs_info = {}
    id_to_display: dict[str, str] = {}
    id_to_url: dict[str, str] = {}
    name_to_id = _generate_job_id_map(workflow)

    # Build reverse maps using GH names → ids
    for gh_name, gh_job in job_by_name.items():
        if gh_name in name_to_id:
            cid = name_to_id[gh_name]
            id_to_display[cid] = gh_name
            id_to_url[cid] = gh_job.get("html_url", "")

    for job in extract_jobs(workflow):
        matrix = job["origin"]["matrix_job"]
        name = matrix.get("job_name", job["id"])
        project = matrix.get("project", "unknown")
        # Prefer mapping via custom id; fallback to fuzzy name match.
        disp_name = id_to_display.get(job["id"]) or name
        url = (
            id_to_url.get(job["id"])
            or _resolve_job_name_and_url(
                name, {k: v.get("html_url", "") for k, v in job_by_name.items()}
            )[1]
        )
        jobs_info[job["id"]] = {
            "name": name,  # matrix job name from workflow
            "display": disp_name,  # GH job name with matrix
            "project": project,
            "url": url,
        }

    # errors[project][summary] -> { full, jobs:set(), location }
    errors = defaultdict(
        lambda: defaultdict(lambda: {"full": "", "jobs": set(), "location": ""})
    )
    unmatched = defaultdict(set)

    for job_id, info in jobs_info.items():
        job_dir = os.path.join(args.jobs_dir, job_id)
        if not os.path.isdir(job_dir):
            continue
        if os.path.exists(os.path.join(job_dir, "success")):
            continue

        found = False
        parser = Path(__file__).resolve().parents[3] / "ci" / "util" / "parse_error.py"
        for log_name in ["configure.log", "build.log", "test.log"]:
            log_path = os.path.join(job_dir, log_name)
            match = _first_match_via_parser(parser, Path(log_path))
            if match:
                found = True
                filepath = match.get("rel_filepath") or match.get("file", "")
                line_no = match.get("line", "")
                summary = (match.get("summary") or "").strip()
                location = f"{filepath}:{line_no}".strip(":")
                entry = errors[info["project"]][summary]
                if not entry["full"]:
                    entry["full"] = match.get("full", "")
                if not entry["location"]:
                    entry["location"] = location
                entry["jobs"].add((info.get("display", info["name"]), info["url"]))
                break
            if found:
                break
        if not found:
            unmatched[info["project"]].add(
                (info.get("display", info["name"]), info["url"])
            )

    if not errors and not unmatched:
        return

    print("<details><summary><h3>🚨 Failure log</h3></summary>\n")
    for project in sorted(errors):
        print(f"<details><summary><h4>📄 {project}</h4></summary>\n")
        for summary in sorted(errors[project]):
            data = errors[project][summary]
            heading = textwrap.shorten(summary, width=120, placeholder="...")
            print(f"<details><summary><h5>⚠️ {heading}</h5></summary>\n")
            if data.get("location"):
                print(f"<p><code>{data['location']}</code></p>")
            print("<pre>")
            print(data["full"])
            print("</pre>\n")
            for job_name, url in sorted(data["jobs"], key=lambda x: x[0]):
                if url:
                    print(f"- [{job_name}]({url})")
                else:
                    print(f"- {job_name}")
            print("\n</details>\n")
        print("</details>\n")
    if unmatched:
        for project in sorted(unmatched):
            print(f"<details><summary><h4>{project}</h4></summary>")
            for job_name, url in sorted(unmatched[project], key=lambda x: x[0]):
                if url:
                    print(f"- [{job_name}]({url})")
                else:
                    print(f"- {job_name}")
            print("\n</details>\n")
    print("</details>")


if __name__ == "__main__":
    main()
