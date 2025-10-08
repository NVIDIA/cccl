#!/usr/bin/env python3
"""Summarize CI failure logs for the workflow-results action.

Inputs
------
- workflow_json: Structured workflow description (groups, jobs, ids) produced
  by prepare-execution-summary.py.
- jobs_json: GitHub API job list for the current run (via actions/github-script).
- jobs_dir: Directory containing per-job artifacts (configure.log, build.log,
  test.log, success flag, etc.).

Outputs
-------
- Prints an HTML+Markdown hybrid to stdout, intended to be included in the
  GHA step summary. It includes a collapsible "Failure Log" section with
  per-error blocks and stable anchors.
- Writes a compact list of per-error rows to workflow/errors_list.md for PR
  comments, using abbreviated messages and deep links to the summary anchors.

Special Cases & Behavior
------------------------
- Deep-links to source locations are generated only if the referenced file
  exists inside the current repo checkout (GITHUB_WORKSPACE); build paths are
  never linked.
- Each error is keyed by its parse_error "summary" and deduplicated per
  project; we aggregate job display names and target names.
- Headings include stable ids: the wrapper uses id="failure-log" and each
  error uses id="error-N" to enable PR comment links to exact blocks.
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TRUNC_MSG_WIDTH = 60
HEADING_SUMMARY_WIDTH = 120
MAX_ERRORS_SHOWN = 50


@dataclass
class ErrorEntry:
    error_line: str = ""
    error_context: str = ""
    error_short: str = ""
    location: str = ""
    filepath: str = ""       # repo-relative path (rel_filepath)
    filename: str = ""
    line: str = ""
    jobs: set = field(default_factory=set)  # set[(display_name, url)]
    targets_all: set = field(default_factory=set)
    targets_by_job: dict = field(default_factory=lambda: defaultdict(set))
    malformed: bool = False
    raw_result: Optional[str] = None  # raw stdout on malformed parse

    def apply_parse_result(self, match: dict):
        # parse_error and this script are version-locked; rely on new names.
        if not self.error_line:
            self.error_line = (match.get("error_line") or "")
        if not self.error_context:
            self.error_context = (match.get("error_context") or "") or self.error_line
        if not self.error_short:
            self.error_short = (match.get("error_short") or "").strip()
        if not self.filename:
            self.filename = (match.get("filename") or "").strip()
        if not self.filepath:
            self.filepath = (match.get("rel_filepath") or "").strip()
        if not self.line:
            self.line = (match.get("line") or "").strip()
        if not self.location:
            loc = (match.get("location") or "").strip()
            if loc:
                self.location = loc
            else:
                base = self.filepath or self.filename
                self.location = f"{base}:{self.line}".strip(":")


def _natural_sort_key(text: str) -> List[str]:
    import re
    return [t.zfill(10) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def _render_targets_line(entry: ErrorEntry) -> str:
    targets = sorted(t for t in entry.targets_all if t)
    if not targets:
        return ""
    esc = lambda s: s.replace("`", "\\`")
    targets_md = ", ".join(f"`{esc(t)}`" for t in targets)
    return f"- 🎯 **Target(s)**: {targets_md}\n"


def _render_error_block(project: str, summary: str, index: int, entry: ErrorEntry,
                        repo: str, sha: str, workspace: str, summary_url: str) -> str:
    # Heading and anchor
    short_summary = textwrap.shorten(summary, width=HEADING_SUMMARY_WIDTH, placeholder="...")
    heading = f"{project}: {short_summary}"
    out = []
    out.append(f"<details><summary>{_make_heading(3, f'⚠️ {heading}', f'error-{index}')}</summary>\n\n")

    # Shareable anchor (link to jobs sub-anchor to scroll appropriately)
    if summary_url:
        anchor_url = f"{summary_url}#user-content-error-{index}-jobs"
        out.append(f"- 🔗 **Shareable Link**: {_make_link('here', anchor_url)}\n")
    else:
        anchor_url = ""

    # Error message
    if entry.error_short:
        safe_msg = entry.error_short.replace("`", "\\`")
        out.append(f"- 💬 **Error Message**: `{safe_msg}`\n")
    if entry.malformed and entry.raw_result:
        raw = str(entry.raw_result).replace("`", "\\`")
        out.append(f"- ⚠️ parse_error returned a malformed result: `{raw}`\n")

    # Location (link if exists in repo)
    if entry.location:
        loc_disp = entry.location.strip()
        url = _deep_link_for_location(repo, sha, workspace, loc_disp, (entry.filepath or "").strip())
        out.append(f"- 📍 **Location**: {_make_link(f'`{loc_disp}`', url)}\n")

    # Targets line
    out.append(_render_targets_line(entry))

    out.append("\n")
    out.append("<ul>\n")

    # Full Error With Context
    out.append("<li><details><summary>🔍 <b>Full Error With Context</b></summary>\n\n")
    out.append("<pre>\n")
    out.append(entry.error_context or entry.error_line or "")
    out.append("\n</pre>\n</details>\n</li>\n")

    # Jobs section with pluralization and bullet links
    jobs_label = _pluralize(len(entry.jobs), 'Job')
    out.append(f"<li><details><summary><span id=\"error-{index}-jobs\">🧰 <b>{jobs_label}</b></span></summary>\n\n")
    for job_name, url in sorted(entry.jobs, key=lambda x: x[0]):
        out.append(f"- {_make_link(job_name, url)}\n")
    out.append("</details>\n")
    out.append("</li>\n")
    out.append("</ul>\n")
    out.append("</details>\n")

    return "".join(out)


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


def _resolve_job_name_and_url(name: str, job_urls: Dict[str, str]) -> Tuple[str, str]:
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


def _all_matches_via_parser(parser: Path, log_path: Path) -> List[dict]:
    """Run parse_error.py on a log and return all JSON matches (deduped by parser)."""

    if not log_path.exists():
        return []
    cmd = [
        sys.executable,
        str(parser),
        "-n",
        "0",  # all matches
        "--format",
        "json",
        str(log_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _generate_job_id_map(workflow: dict) -> Dict[str, str]:
    """Map full GitHub job name to custom job id from workflow.json.

    Mirrors logic in parse-job-times to keep name construction consistent.
    """
    job_id_map: Dict[str, str] = {}
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


def _deep_link_for_location(repo: str, sha: str, workspace: str, location_disp: str, file_path: str) -> str:
    """Return a GitHub blob URL for a location if it exists in the repo checkout.

    - Ensures the path resolves within `workspace` and is a file.
    - Extracts the trailing line number from `location_disp` of the form
      "path:line". If no line can be parsed, returns empty string.
    """
    if not (repo and sha and file_path and location_disp):
        return ""
    # Extract line number from the display string (use right-most colon)
    line = ""
    if ":" in location_disp:
        try:
            candidate_line = location_disp.rsplit(":", 1)[1].strip()
            # Validate numeric line
            int(candidate_line)
            line = candidate_line
        except Exception:
            line = ""
    if not line:
        return ""
    try:
        ws_path = Path(workspace).resolve()
        candidate_path = (ws_path / file_path.lstrip("/")).resolve()
        # Ensure candidate is within workspace
        candidate_path.relative_to(ws_path)
        if not candidate_path.is_file():
            return ""
        repo_rel = candidate_path.relative_to(ws_path).as_posix()
        return f"https://github.com/{repo}/blob/{sha}/{repo_rel}#L{line}"
    except Exception:
        return ""


def _make_link(text, url):
    # Need to make html link in list item to avoid markdown parsing issues
    return f"<a href=\"{url}\">{text}</a>" if url else text


def _make_heading(level, text, anchor=None):
    # Same for heading markup
    if anchor:
        return f"<h{level} id=\"{anchor}\">{text}</h{level}>"
    else:
        return f"<h{level}>{text}</h{level}>"


def _pluralize(n: int, singular: str, plural: str | None = None) -> str:
    if n == 1:
        return f"{n} {singular}"
    return f"{n} {plural or singular + 's'}"

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
    jobs_info: Dict[str, dict] = {}
    id_to_display: Dict[str, str] = {}
    id_to_url: Dict[str, str] = {}
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

    # errors[project][summary] -> {
    #   full, context,
    #   jobs:set[(display_name, url)],
    #   location, file, abs,
    #   targets_all:set[str],
    #   targets_by_job: dict[display_name, set[str]]
    # }
    # errors[project][(summary, pattern_type, rel_filepath, filename, target_name)] -> ErrorEntry
    errors: Dict[str, Dict[tuple, ErrorEntry]] = defaultdict(dict)
    unmatched = defaultdict(set)
    # Track first error per target per project, keep the first one only
    seen_targets_by_project: Dict[str, set] = defaultdict(set)

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
            matches = _all_matches_via_parser(parser, Path(log_path))
            if matches:
                found = True
            for match in matches:
                if not isinstance(match, dict):
                    # Malformed parser output; record under a synthetic summary
                    summary = "[malformed parse_error result]"
                    key = (summary, "", "", "", "")
                    entry = errors[info["project"]].setdefault(key, ErrorEntry())
                    entry.malformed = True
                    entry.raw_result = str(match)
                    disp = info.get("display", info["name"])
                    entry.jobs.add((disp, info["url"]))
                    continue
                # Per-target dedupe across all errors for this project
                target = (match.get("target_name") or "").strip()
                if target:
                    seen = seen_targets_by_project[info["project"]]
                    if target in seen:
                        continue
                    seen.add(target)
                summary = (match.get("summary") or "").strip() or "[unknown error]"
                key = (
                    summary,
                    (match.get("pattern_type") or "").strip(),
                    (match.get("rel_filepath") or "").strip(),
                    (match.get("filename") or "").strip(),
                    target,
                )
                entry = errors[info["project"]].setdefault(key, ErrorEntry())
                entry.apply_parse_result(match)
                disp = info.get("display", info["name"])
                entry.jobs.add((disp, info["url"]))
                if target:
                    entry.targets_all.add(target)
                    entry.targets_by_job[disp].add(target)
        if not found:
            unmatched[info["project"]].add(
                (info.get("display", info["name"]), info["url"])
            )

    if not errors and not unmatched:
        return

    print(f"<details><summary>{_make_heading(2, '🚨 Failure Log', 'failure-log')}</summary>\n")
    # Build repo/SHA context for deep links to file locations in GitHub UI.
    repo = (os.environ.get("GITHUB_REPOSITORY") or "NVIDIA/cccl").strip()
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    workspace = os.environ.get("GITHUB_WORKSPACE") or os.getcwd()
    # Summary page URL (if available) for linking PR comment elements
    summary_url = ""
    try:
        summary_url_path = Path("workflow/summary_url.txt")
        if summary_url_path.exists():
            summary_url = summary_url_path.read_text(encoding="utf-8").strip()
    except Exception:
        summary_url = ""
    # Collect compact error rows for PR comment
    compact_rows: list[str] = []
    error_counter = 0
    total_errors = sum(len(group) for group in errors.values())
    truncated = False
    for project in sorted(errors):
        # Sort by display summary (tuple[0])
        for key in sorted(errors[project], key=lambda k: _natural_sort_key(k[0])):
            summary = key[0]
            data = errors[project][key]
            error_counter += 1
            if error_counter <= MAX_ERRORS_SHOWN:
                print(_render_error_block(project, summary, error_counter, data, repo, sha, workspace, summary_url))
            else:
                truncated = True

            # Build compact PR comment row in the format:
            # - **project** (N job(s)): <filename:line (linked if valid)>: `<full untruncated message>` [full log]
            # Choose last job alphabetically by display name for the log link
            job_url = ""
            if data.jobs:
                last_job = sorted(data.jobs, key=lambda x: x[0])[-1]
                _, job_url = last_job
            # Display only filename:line (no path) using values from JSON
            filename = (data.filename or "").strip()
            line_no = (data.line or "").strip()
            loc_file = (data.filepath or "").strip()  # repo-relative path for linking
            loc_disp_short = f"{filename}:{line_no}".strip(":") if filename or line_no else ""
            loc_url = _deep_link_for_location(repo, sha, workspace, loc_disp_short, loc_file)
            loc_md = (
                f"[`{loc_disp_short}`]({loc_url})" if (loc_disp_short and loc_url)
                else f"`{loc_disp_short}`" if loc_disp_short else ""
            )
            # Use truncated message for compact PR comment; collapse whitespace
            msg_raw = (data.error_short or summary or "").strip()
            msg_trunc = textwrap.shorten(msg_raw, width=TRUNC_MSG_WIDTH, placeholder="...")
            msg_one_line = " ".join(msg_trunc.split())
            msg_one_line = msg_one_line.replace("`", "\\`")
            jobs_text = _pluralize(len(data.jobs), 'job')
            # Anchor for the corresponding error block
            anchor = f"{summary_url}#user-content-error-{error_counter}-jobs" if summary_url else ""
            details_link = f" [details]({anchor})" if anchor else ""
            row = f"**{project}**: ({jobs_text}): {loc_md} `{msg_one_line}`{details_link}"
            if job_url:
                row += f" [full log]({job_url})"
            compact_rows.append(row)
    if truncated:
        remaining = total_errors - MAX_ERRORS_SHOWN
        print(f"\n> Note: Showing only the first {MAX_ERRORS_SHOWN} errors out of {total_errors}. {remaining} more not shown to keep this summary readable.\n")
    # end for project/errors
    if unmatched:
        for project in sorted(unmatched):
            print(f"<details><summary>{_make_heading(3, project)}</summary>")
            for job_name, url in sorted(unmatched[project], key=lambda x: x[0]):
                print(f"- {_make_link(job_name, url)}")
            print("\n</details>\n")
    print("</details>")

    # Write compact rows for PR comment consumption (truncate after 10)
    try:
        os.makedirs("workflow", exist_ok=True)
        with open("workflow/errors_list.md", "w") as f:
            limit = 10
            more = max(0, len(compact_rows) - limit)
            rows = compact_rows[:limit]
            for row in rows:
                f.write(row + "\n")
            if more > 0:
                anchor = "#user-content-failure-log"
                url = (Path("workflow/summary_url.txt").read_text().strip() + anchor) if os.path.exists("workflow/summary_url.txt") else anchor
                f.write(_make_link(f"...plus {more} more\n", url))
    except Exception:
        pass


if __name__ == "__main__":
    main()
