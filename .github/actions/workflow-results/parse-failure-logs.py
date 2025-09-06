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

    # errors[project][summary] -> { full, context, jobs:set(), location, file, abs }
    errors = defaultdict(
        lambda: defaultdict(lambda: {"full": "", "context": "", "jobs": set(), "location": "", "file": "", "abs": ""})
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
                filepath = (match.get("rel_filepath") or match.get("file", "") or "").strip()
                abs_path = (match.get("abs_filepath") or "").strip()
                line_no = (match.get("line", "") or "").strip()
                summary = (match.get("summary") or "").strip()
                location = f"{filepath}:{line_no}".strip(":")
                entry = errors[info["project"]][summary]
                if not entry["full"]:
                    entry["full"] = match.get("full", "")
                if not entry["context"]:
                    entry["context"] = match.get("context", "") or match.get("full", "")
                if not entry["location"]:
                    entry["location"] = location
                if not entry["file"]:
                    entry["file"] = filepath
                if not entry["abs"]:
                    entry["abs"] = abs_path
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

    print("<details><summary><h2>🚨 Failure Log</h2></summary>\n")
    # Build repo/SHA context for deep links to file locations in GitHub UI.
    repo = (os.environ.get("GITHUB_REPOSITORY") or "NVIDIA/cccl").strip()
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    # Collect compact error rows for PR comment
    compact_rows: list[str] = []
    for project in sorted(errors):
        print(f"<details><summary><h3>📄 {project}</h3></summary>\n")
        for summary in sorted(errors[project]):
            data = errors[project][summary]
            heading = textwrap.shorten(summary, width=120, placeholder="...")
            print(f"<details><summary><h4>⚠️ {heading}</h4></summary>\n")
            if data.get("location"):
                # Try to build a deep link to the file and line on GitHub.
                loc_disp = data["location"].strip()
                loc_file = data.get("file", "").strip()
                # Extract line from the displayed location if possible.
                line = ""
                if ":" in loc_disp:
                    try:
                        line = loc_disp.rsplit(":", 1)[1].strip()
                    except Exception:
                        line = ""
                if repo and sha and loc_file and line:
                    url = f"https://github.com/{repo}/blob/{sha}/{loc_file}#L{line}"
                    print(f"📍 Location: [`{loc_disp}`]({url})\n")
                else:
                    # Fallback to plain text if any component is missing.
                    print(f"📍 Location: `{loc_disp}`\n")

            # Nested details for Full Error context
            print("<details><summary>🔍 Full Error</summary>\n")
            print("<pre>")
            print(data.get("context") or data.get("full") or "")
            print("</pre>\n</details>\n")

            # Non-collapsed Links section (placed after Full Error)
            print("🔗 Links:")
            for job_name, url in sorted(data["jobs"], key=lambda x: x[0]):
                if url:
                    print(f"- [{job_name}]({url})")
                else:
                    print(f"- {job_name}")
            print("")
            print("\n</details>\n")

            # Build compact row: "<project>: <summary> [log] [loc]"
            # Choose last job alphabetically by display name
            job_link = "[log]"
            if data["jobs"]:
                last_job = sorted(data["jobs"], key=lambda x: x[0])[-1]
                _, job_url = last_job
                if job_url:
                    job_link = f"[log]({job_url})"
            # Deep link for loc
            loc_disp = data.get("location", "").strip()
            loc_file = data.get("file", "").strip()
            line = ""
            if ":" in loc_disp:
                try:
                    line = loc_disp.rsplit(":", 1)[1].strip()
                except Exception:
                    line = ""
            loc_link = "[loc]"
            if repo and sha and loc_file and line:
                # Verify the file exists in the source tree (not build tree)
                workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
                candidate = os.path.join(workspace, loc_file.lstrip("/"))
                is_build_path = loc_file.split("/", 1)[0] == "build"
                if os.path.isfile(candidate) and not is_build_path:
                    loc_url = f"https://github.com/{repo}/blob/{sha}/{loc_file}#L{line}"
                    loc_link = f"[loc]({loc_url})"
            compact_rows.append(f"{project}: {summary} {job_link} {loc_link}")
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

    # Write compact rows for PR comment consumption
    try:
        os.makedirs("workflow", exist_ok=True)
        with open("workflow/errors_list.md", "w") as f:
            for row in compact_rows:
                f.write(row + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
