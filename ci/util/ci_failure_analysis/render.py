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
SLACK_PARENT_LIMIT = 39000
SLACK_REPLY_LIMIT = 3500
SLACK_REPLY_COUNT_LIMIT = 256
SLACK_THREAD_LIMIT = 60000
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


def validate_job_manifest(job_items):
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

    for field, label in (("id", "id"), ("number", "number")):
        value = run.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValidationError(f"workflow run {label} must be a positive integer")

    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValidationError(
            "workflow run head SHA must be 40 lowercase hexadecimal characters"
        )


def load_analysis_context(analysis_file):
    context = load_json(analysis_file)
    if not isinstance(context, dict):
        raise ValidationError("analysis context must be an object")

    expected_fields = {"run", "jobs", "groups"}
    missing = sorted(expected_fields - set(context))
    if missing:
        raise ValidationError(f"analysis context is missing fields {missing}")
    extra = sorted(set(context) - expected_fields)
    if extra:
        raise ValidationError(f"analysis context has unexpected fields {extra}")

    schema = load_json(Path(__file__).with_name("output.schema.json"))
    analysis = {"groups": context["groups"]}
    validate_schema(analysis, schema)

    jobs, step_numbers, failed_job_ids = validate_job_manifest(context["jobs"])
    validate_job_references(analysis, step_numbers, failed_job_ids)

    run = context["run"]
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


# Slack thread.


def sanitize_slack(value, limit=1200):
    value = " ".join(clean_text(value, limit).split())
    value = value.replace("&", "&amp;")
    value = value.replace("<", "&lt;").replace(">", "&gt;")
    return value.replace("`", "'").replace("*", "∗")


def sanitize_slack_title(value):
    return sanitize_slack(value, limit=100)


def sanitize_slack_link_label(value, limit):
    return sanitize_slack(value, limit).replace("|", "¦")


def render_slack_job_link(job_id, jobs, repository, run_id):
    label = sanitize_slack_link_label(jobs[job_id], limit=300)
    return f"<{job_url(repository, run_id, job_id)}|{label}>"


def render_slack_evidence(group):
    for evidence in group["evidence"]:
        for line in evidence["lines"]:
            rendered_line = sanitize_slack(line, limit=700)
            if not rendered_line:
                continue
            return f"*Evidence:* {rendered_line}"
    return None


def render_slack_sources(group, repository, head_sha):
    links = []
    for source in group["source_locations"]:
        url = source_url(repository, head_sha, source["path"], source["line"])
        if not url:
            continue
        label = sanitize_slack_link_label(
            f"{source['path']}:{source['line']}",
            limit=500,
        )
        links.append(f"<{url}|{label}>")
        if len(links) == 2:
            break
    if not links:
        return None
    return f"*Sources:* {', '.join(links)}"


def render_slack_thread_reply(
    index,
    group,
    jobs,
    repository,
    run_id,
    head_sha,
):
    job_ids = group["job_ids"]
    job_label = "job" if len(job_ids) == 1 else "jobs"
    lines = [
        (
            f"*{index}. {sanitize_slack_title(group['title'])}* — "
            f"{len(job_ids)} {job_label}"
        ),
        f"*Root cause:* {sanitize_slack(group['root_cause'], limit=700)}",
    ]
    evidence = render_slack_evidence(group)
    if evidence:
        lines.append(evidence)
    sources = render_slack_sources(group, repository, head_sha)
    if sources:
        lines.append(sources)
    lines.append(
        f"*Suggested next steps:* {sanitize_slack(group['next_steps'], limit=700)}"
    )
    lines.append("*Jobs:*")
    lines.extend(
        f"• {render_slack_job_link(job_id, jobs, repository, run_id)}"
        for job_id in job_ids
    )

    reply = "\n".join(lines) + "\n"
    if len(reply) > SLACK_REPLY_LIMIT:
        raise ValidationError(
            f"rendered Slack reply exceeds {SLACK_REPLY_LIMIT:,} characters"
        )
    return reply


def render_slack_thread(
    analysis,
    jobs,
    repository,
    run_id,
    run_number,
    head_sha,
):
    failed_job_count = sum(len(group["job_ids"]) for group in analysis["groups"])
    group_count = len(analysis["groups"])
    if group_count > SLACK_REPLY_COUNT_LIMIT:
        raise ValidationError(
            f"Slack thread has more than {SLACK_REPLY_COUNT_LIMIT} failure groups"
        )
    job_label = "job" if failed_job_count == 1 else "jobs"
    group_label = "group" if group_count == 1 else "groups"
    parent_lines = [
        f":rotating_light: *AI failure analysis — workflow run #{run_number}*",
        (
            f"{group_count} failure {group_label} covering "
            f"{failed_job_count} primary failed {job_label}."
        ),
        "",
    ]
    for index, group in enumerate(analysis["groups"], start=1):
        job_ids = group["job_ids"]
        job_label = "job" if len(job_ids) == 1 else "jobs"
        parent_lines.append(
            f"{index}. {sanitize_slack_title(group['title'])} "
            f"· {len(job_ids)} {job_label}"
        )

    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    parent_lines.extend(["", f"<{run_url}|GitHub Actions>"])
    parent = "\n".join(parent_lines) + "\n"
    if len(parent) > SLACK_PARENT_LIMIT:
        raise ValidationError(
            f"rendered Slack parent exceeds {SLACK_PARENT_LIMIT:,} characters"
        )

    replies = [
        render_slack_thread_reply(
            index,
            group,
            jobs,
            repository,
            run_id,
            head_sha,
        )
        for index, group in enumerate(analysis["groups"], start=1)
    ]
    return {"parent": parent, "replies": replies}


# Command-line entry point.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and render a structured CI failure analysis."
    )
    parser.add_argument("--analysis-file", type=Path, required=True)
    parser.add_argument("--format", choices=("github", "slack"), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_repository(args.repository)
    analysis, jobs, run = load_analysis_context(args.analysis_file)
    run_id = str(run["id"])

    if args.format == "github":
        rendered = render_github_report(
            analysis,
            jobs,
            args.repository,
            run_id,
            run["head_sha"],
        )
    else:
        slack_thread = render_slack_thread(
            analysis,
            jobs,
            args.repository,
            run_id,
            str(run["number"]),
            run["head_sha"],
        )
        rendered = (
            json.dumps(
                slack_thread,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        if len(rendered.encode("utf-8")) > SLACK_THREAD_LIMIT:
            raise ValidationError(
                f"rendered Slack thread exceeds {SLACK_THREAD_LIMIT:,} bytes"
            )

    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        safe_error = str(error).encode("unicode_escape").decode("ascii")
        print(
            f"error: could not render CI triage output: {safe_error}", file=sys.stderr
        )
        raise SystemExit(1)
