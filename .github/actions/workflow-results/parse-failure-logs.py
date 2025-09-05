#!/usr/bin/env python3

import argparse
import json
import os

# Allow importing utilities from ci/util
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3] / "ci" / "util"))
from parse_error import find_match  # noqa: E402


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

    # Map job name to log URL
    job_urls = {job["name"]: job.get("html_url", "") for job in gha_jobs}

    # Build mapping from job id -> info
    jobs_info = {}
    for job in extract_jobs(workflow):
        matrix = job["origin"]["matrix_job"]
        name = matrix.get("job_name", job["id"])
        project = matrix.get("project", "unknown")
        jobs_info[job["id"]] = {
            "name": name,
            "project": project,
            "url": job_urls.get(name, ""),
        }

    errors = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"full": "", "jobs": set()}))
    )
    unmatched = defaultdict(set)

    for job_id, info in jobs_info.items():
        job_dir = os.path.join(args.jobs_dir, job_id)
        if not os.path.isdir(job_dir):
            continue
        if os.path.exists(os.path.join(job_dir, "success")):
            continue

        found = False
        for log_name in ["configure.log", "build.log", "test.log"]:
            log_path = os.path.join(job_dir, log_name)
            if not os.path.exists(log_path):
                continue
            with open(log_path, "r", errors="ignore") as f:
                for line in f:
                    line = line.rstrip()
                    match = find_match(line)
                    if match:
                        found = True
                        filepath = match.group("file")
                        line_no = match.groupdict().get("line", "")
                        msg = match.group("msg").strip()
                        location = f"{filepath}:{line_no}".strip(":")
                        entry = errors[info["project"]][location][msg]
                        entry["full"] = line
                        entry["jobs"].add((info["name"], info["url"]))
                        break
            if found:
                break
        if not found:
            unmatched[info["project"]].add((info["name"], info["url"]))

    if not errors and not unmatched:
        return

    print("<details><summary><h3>🚨 Failure log</h3></summary>\n")
    for project in sorted(errors):
        for location in sorted(errors[project]):
            print(f"<details><summary><h4>📄 {project}: {location}</h4></summary>\n")
            for msg in sorted(errors[project][location]):
                data = errors[project][location][msg]
                summary = textwrap.shorten(msg, width=120, placeholder="...")
                print(f"<details><summary><h5>⚠️ {summary}</h5></summary>\n")
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
