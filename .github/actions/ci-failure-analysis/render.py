#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import quote

GITHUB_REPORT_LIMIT = 60000
SLACK_SUMMARY_LIMIT = 3500
FAILED_CONCLUSIONS = {
    "action_required",
    "failure",
    "startup_failure",
    "timed_out",
}
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
INLINE_MARKDOWN = re.compile(r"([\\`*_\[\]])")
URL_SCHEME = re.compile(r"(?i)\b(https?):/{2}")
WWW_LINK = re.compile(r"(?i)\bwww\.")


# Input validation and shared data.


class ValidationError(ValueError):
    pass


def load_json(path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"could not read JSON from {path}: {error}") from error


def validate_schema(value, schema, location="$"):
    expected_type = schema.get("type")
    checks = {
        "array": lambda item: isinstance(item, list),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "string": lambda item: isinstance(item, str),
    }
    if expected_type not in checks or not checks[expected_type](value):
        raise ValidationError(f"{location}: expected {expected_type}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{location}: value is not one of {schema['enum']}")

    if expected_type == "object":
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ValidationError(f"{location}: missing required fields {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValidationError(f"{location}: unexpected fields {extra}")
        for name, item in value.items():
            if name in properties:
                validate_schema(item, properties[name], f"{location}.{name}")

    if expected_type == "array":
        if len(value) < schema.get("minItems", 0):
            raise ValidationError(
                f"{location}: expected at least {schema['minItems']} items"
            )
        for index, item in enumerate(value):
            validate_schema(item, schema["items"], f"{location}[{index}]")


def load_job_manifest(path):
    job_items = load_json(path)
    if not isinstance(job_items, list):
        raise ValidationError("job manifest must be an array")

    jobs = {}
    step_numbers = {}
    failed_job_ids = set()
    for index, job in enumerate(job_items):
        location = f"job manifest[{index}]"
        if not isinstance(job, dict):
            raise ValidationError(f"{location} must be an object")
        job_id = job.get("id")
        name = job.get("name")
        conclusion = job.get("conclusion")
        steps = job.get("steps", [])
        if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
            raise ValidationError(f"{location}.id must be a positive integer")
        if job_id in jobs:
            raise ValidationError(f"{location}.id duplicates job {job_id}")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{location}.name must be a non-empty string")
        if conclusion is not None and not isinstance(conclusion, str):
            raise ValidationError(f"{location}.conclusion must be a string or null")
        if conclusion not in FAILED_CONCLUSIONS:
            raise ValidationError(f"{location}.conclusion is not a failed conclusion")
        if not isinstance(steps, list):
            raise ValidationError(f"{location}.steps must be an array")

        numbers = set()
        for step_index, step in enumerate(steps):
            number = step.get("number") if isinstance(step, dict) else None
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                raise ValidationError(
                    f"{location}.steps[{step_index}].number must be positive"
                )
            numbers.add(number)

        jobs[job_id] = name
        step_numbers[job_id] = numbers
        failed_job_ids.add(job_id)

    return jobs, step_numbers, failed_job_ids


def validate_job_references(analysis, step_numbers, failed_job_ids):
    grouped_job_ids = set()
    for group_index, group in enumerate(analysis["groups"]):
        location = f"$.groups[{group_index}]"
        job_ids = group["job_ids"]
        if not job_ids:
            raise ValidationError(f"{location}.job_ids must not be empty")
        if len(job_ids) != len(set(job_ids)):
            raise ValidationError(f"{location}.job_ids contains duplicates")
        for job_id in job_ids:
            if job_id not in failed_job_ids:
                raise ValidationError(
                    f"{location}.job_ids references non-failed job {job_id}"
                )
            if job_id in grouped_job_ids:
                raise ValidationError(
                    f"job {job_id} appears in more than one failure group"
                )
            grouped_job_ids.add(job_id)

        group_job_ids = set(job_ids)
        for evidence_index, evidence in enumerate(group["evidence"]):
            evidence_location = f"{location}.evidence[{evidence_index}]"
            job_id = evidence["job_id"]
            step_number = evidence["step_number"]
            if job_id not in group_job_ids:
                raise ValidationError(
                    f"{evidence_location}.job_id is not in the failure group"
                )
            if step_number != 0 and step_number not in step_numbers[job_id]:
                raise ValidationError(
                    f"{evidence_location}.step_number does not exist for job {job_id}"
                )


def validate_run(run):
    if not isinstance(run, dict):
        raise ValidationError("workflow run metadata must be an object")

    for field, label in (("id", "id"), ("run_number", "number")):
        value = run.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValidationError(f"workflow run {label} must be a positive integer")

    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValidationError(
            "workflow run head SHA must be 40 lowercase hexadecimal characters"
        )


def load_analysis_context(analysis_dir):
    schema = load_json(Path(__file__).with_name("output.schema.json"))
    analysis = load_json(analysis_dir / "analysis.json")
    validate_schema(analysis, schema)

    jobs, step_numbers, failed_job_ids = load_job_manifest(analysis_dir / "jobs.json")
    validate_job_references(analysis, step_numbers, failed_job_ids)

    run = load_json(analysis_dir / "run.json")
    validate_run(run)
    return analysis, jobs, run


def validate_repository(repository):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValidationError("repository must have the form owner/name")


def clean_text(value, limit=None):
    if limit is not None:
        value = value[:limit]
    value = CONTROL_CHARACTER.sub("", value)
    return "".join(
        character
        for character in value
        if unicodedata.category(character) not in {"Cf", "Cs"}
    )


# GitHub report.


def sanitize_html(value, limit=1200):
    value = " ".join(clean_text(value, limit).split())
    value = html.escape(value, quote=True)
    value = re.sub(r"#(?=\d)", "&#35;", value)
    value = URL_SCHEME.sub(lambda match: f"{match.group(1)}&#58;//", value)
    value = WWW_LINK.sub(lambda match: f"{match.group(0)[:-1]}&#46;", value)
    value = value.replace("@", "&#64;")
    return value


def sanitize_inline(value, limit=1200):
    value = sanitize_html(value, limit)
    return INLINE_MARKDOWN.sub(r"\\\1", value)


def code_block(value, limit=None):
    value = clean_text(value, limit).strip()
    longest_run = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{value}\n{fence}"


def job_url(repository, run_id, job_id):
    return f"https://github.com/{repository}/actions/runs/{run_id}/job/{job_id}"


def source_url(repository, head_sha, path, line):
    path = clean_text(path)
    if not path:
        return None
    path = PurePosixPath(path)
    if path.is_absolute() or ".." in path.parts or line <= 0:
        return None
    encoded_path = quote(path.as_posix(), safe="/")
    return f"https://github.com/{repository}/blob/{head_sha}/{encoded_path}#L{line}"


def job_link(job_id, jobs, repository, run_id, step_number=0):
    label = sanitize_inline(jobs[job_id], limit=300)
    url = job_url(repository, run_id, job_id)
    if step_number > 0:
        label += f", step {step_number}"
        url += f"#step:{step_number}:1"
    return f"[{label}]({url})"


def render_github_evidence(group, jobs, repository, run_id):
    rendered = []
    remaining_lines = 3
    for evidence in group["evidence"][:3]:
        selected_lines = evidence["lines"][:remaining_lines]
        if not selected_lines:
            continue
        rendered.extend(
            [
                job_link(
                    evidence["job_id"],
                    jobs,
                    repository,
                    run_id,
                    evidence["step_number"],
                ),
                "",
                code_block("\n".join(selected_lines), limit=1800),
            ]
        )
        remaining_lines -= len(selected_lines)
        if remaining_lines == 0:
            break
        rendered.append("")
    return rendered


def render_github_group(index, group, jobs, repository, run_id, head_sha):
    job_ids = group["job_ids"]
    job_label = "job" if len(job_ids) == 1 else "jobs"
    lines = [
        "<details>",
        (
            "<summary><strong>"
            f"{index}. {sanitize_html(group['title'], limit=100)}"
            "</strong>"
            f" &middot; {len(job_ids)} {job_label}"
            "</summary>"
        ),
        "",
        f"**Explanation:** {sanitize_inline(group['explanation'])}",
    ]

    evidence = render_github_evidence(group, jobs, repository, run_id)
    if evidence:
        lines.extend(["", "**Evidence:**", "", *evidence])

    root_cause = f"**Root cause:** {sanitize_inline(group['root_cause'])}"
    source_links = []
    for source in group["source_locations"][:5]:
        url = source_url(repository, head_sha, source["path"], source["line"])
        if url:
            label = sanitize_inline(f"{source['path']}:{source['line']}", limit=500)
            source_links.append(f"[{label}]({url})")
    if source_links:
        root_cause += f" Sources: {', '.join(source_links)}."

    lines.extend(
        [
            "",
            root_cause,
            "",
            f"**Suggested next steps:** {sanitize_inline(group['next_steps'])}",
        ]
    )

    prompt_lines = [
        f"Repository: https://github.com/{repository}",
        f"Workflow run: https://github.com/{repository}/actions/runs/{run_id}",
        f"Failure group: {group['title']}",
        "Affected jobs:",
        *[
            f"- {jobs[job_id]}: {job_url(repository, run_id, job_id)}"
            for job_id in job_ids
        ],
        "",
        group["agent_prompt"],
    ]
    lines.extend(
        [
            "",
            "<details>",
            "<summary><strong>Copy this prompt into a coding agent</strong></summary>",
            "",
            code_block("\n".join(prompt_lines)),
            "",
            "</details>",
            "",
            "**Jobs:**",
        ]
    )
    lines.extend(
        f"- {job_link(job_id, jobs, repository, run_id)}" for job_id in job_ids
    )
    lines.extend(
        [
            "",
            "</details>",
        ]
    )
    return lines


def render_github_report(analysis, jobs, repository, run_id, head_sha):
    lines = ["### AI failure analysis", ""]
    for index, group in enumerate(analysis["groups"], start=1):
        lines.extend(
            render_github_group(
                index,
                group,
                jobs,
                repository,
                run_id,
                head_sha,
            )
        )
        lines.append("")

    report = "\n".join(lines) + "\n"
    if len(report.encode("utf-8")) > GITHUB_REPORT_LIMIT:
        raise ValidationError(
            f"rendered GitHub report exceeds {GITHUB_REPORT_LIMIT:,} bytes"
        )
    return report


# Slack summary.


def sanitize_slack(value, limit=1200):
    value = " ".join(clean_text(value, limit).split())
    value = value.replace("&", "&amp;")
    value = value.replace("<", "&lt;").replace(">", "&gt;")
    return value.replace("`", "'")


def render_slack_summary_group(index, group):
    job_ids = group["job_ids"]
    job_label = "job" if len(job_ids) == 1 else "jobs"
    return [
        (
            f"*{index}.* `{sanitize_slack(group['title'], limit=100)}` — "
            f"{len(job_ids)} {job_label}"
        ),
        f"Root cause: `{sanitize_slack(group['root_cause'], limit=360)}`",
        f"Next: `{sanitize_slack(group['next_steps'], limit=360)}`",
    ]


def compose_slack_summary(header, sections, omitted, footer):
    lines = list(header)
    for section in sections:
        lines.extend(["", *section])
    if omitted:
        group_label = "group" if omitted == 1 else "groups"
        lines.extend(
            ["", f"_{omitted} more {group_label} omitted from this Slack message._"]
        )
    lines.extend(["", footer])
    return "\n".join(lines)


def render_slack_summary(
    analysis,
    repository,
    run_id,
    run_number,
    limit=SLACK_SUMMARY_LIMIT,
):
    failed_job_count = sum(len(group["job_ids"]) for group in analysis["groups"])
    group_count = len(analysis["groups"])
    job_label = "job" if failed_job_count == 1 else "jobs"
    group_label = "group" if group_count == 1 else "groups"
    header = [
        f":rotating_light: *AI failure analysis — workflow run #{run_number}*",
        (
            f"{group_count} failure {group_label} covering "
            f"{failed_job_count} primary failed {job_label}."
        ),
    ]
    all_sections = [
        render_slack_summary_group(index, group)
        for index, group in enumerate(analysis["groups"], start=1)
    ]
    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    footer = f"GitHub report: <{run_url}|GitHub Actions>"
    complete_summary = compose_slack_summary(header, all_sections, 0, footer)
    if len(complete_summary) + 1 <= limit:
        return complete_summary + "\n"

    included_sections = []
    for section in all_sections:
        candidate_sections = [*included_sections, section]
        omitted = len(all_sections) - len(candidate_sections)
        candidate = compose_slack_summary(header, candidate_sections, omitted, footer)
        if len(candidate) + 1 > limit:
            break
        included_sections = candidate_sections

    omitted = len(all_sections) - len(included_sections)
    summary = compose_slack_summary(header, included_sections, omitted, footer)
    if len(summary) + 1 > limit:
        raise ValidationError(f"rendered Slack summary exceeds {limit:,} characters")
    return summary + "\n"


# Command-line entry point.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and render CI failure analysis outputs."
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--github-report-output", type=Path, required=True)
    parser.add_argument("--slack-summary-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_repository(args.repository)
    analysis, jobs, run = load_analysis_context(args.analysis_dir)
    run_id = str(run["id"])

    github_report = render_github_report(
        analysis,
        jobs,
        args.repository,
        run_id,
        run["head_sha"],
    )
    slack_summary = None
    if args.slack_summary_output:
        slack_summary = render_slack_summary(
            analysis,
            args.repository,
            run_id,
            str(run["run_number"]),
        )

    args.github_report_output.write_text(github_report, encoding="utf-8")
    if args.slack_summary_output:
        args.slack_summary_output.write_text(slack_summary, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        safe_error = str(error).encode("unicode_escape").decode("ascii")
        print(
            f"error: could not render CI triage output: {safe_error}", file=sys.stderr
        )
        raise SystemExit(1)
