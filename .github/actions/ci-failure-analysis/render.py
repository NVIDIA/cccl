#!/usr/bin/env python3

import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import quote

REPORT_LIMIT = 60000
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
INLINE_MARKDOWN = re.compile(r"([\\`*_\[\]])")
URL_SCHEME = re.compile(r"(?i)\b(https?):/{2}")
FAILED_CONCLUSIONS = {
    "action_required",
    "failure",
    "startup_failure",
    "timed_out",
}


class ValidationError(ValueError):
    pass


def fail(message):
    raise ValidationError(message)


def load_json(path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"could not read JSON from {path}: {error}")


def validate_schema(value, schema, location="$"):
    expected_type = schema.get("type")
    checks = {
        "array": lambda item: isinstance(item, list),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "object": lambda item: isinstance(item, dict),
        "string": lambda item: isinstance(item, str),
    }
    if expected_type not in checks or not checks[expected_type](value):
        fail(f"{location}: expected {expected_type}")
    if "enum" in schema and value not in schema["enum"]:
        fail(f"{location}: value is not one of {schema['enum']}")

    if expected_type == "object":
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            fail(f"{location}: missing required fields {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                fail(f"{location}: unexpected fields {extra}")
        for name, item in value.items():
            if name in properties:
                validate_schema(item, properties[name], f"{location}.{name}")

    if expected_type == "array":
        for index, item in enumerate(value):
            validate_schema(item, schema["items"], f"{location}[{index}]")


def clean_text(value, limit=None):
    if limit is not None:
        value = value[:limit]
    value = CONTROL_CHARACTER.sub("", value)
    return "".join(
        character
        for character in value
        if unicodedata.category(character) not in {"Cf", "Cs"}
    )


def sanitize_html(value, limit=1200):
    value = " ".join(clean_text(value, limit).split())
    value = html.escape(value, quote=True)
    value = re.sub(r"#(?=\d)", "&#35;", value)
    value = URL_SCHEME.sub(lambda match: f"{match.group(1)}&#58;//", value)
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


def unique(values):
    return list(dict.fromkeys(values))


def plural(count, singular, plural_form=None):
    return singular if count == 1 else (plural_form or f"{singular}s")


def job_url(repository, run_id, job_id):
    if job_id <= 0:
        return None
    return f"https://github.com/{repository}/actions/runs/{run_id}/job/{job_id}"


def source_url(repository, head_sha, path, line):
    path = PurePosixPath(path)
    if path.is_absolute() or ".." in path.parts or line <= 0:
        return None
    encoded_path = quote(path.as_posix(), safe="/")
    return f"https://github.com/{repository}/blob/{head_sha}/{encoded_path}#L{line}"


def load_job_manifest(path):
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        fail("job manifest must be an object")
    job_items = manifest.get("jobs")
    total_count = manifest.get("total_count")
    if not isinstance(job_items, list):
        fail("job manifest must contain a jobs array")
    if (
        not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count != len(job_items)
    ):
        fail("job manifest total_count does not match its jobs array")

    jobs = {}
    step_numbers = {}
    failed_job_ids = set()
    cancelled_job_ids = []
    for index, job in enumerate(job_items):
        location = f"job manifest jobs[{index}]"
        if not isinstance(job, dict):
            fail(f"{location} must be an object")
        job_id = job.get("id")
        name = job.get("name")
        conclusion = job.get("conclusion")
        steps = job.get("steps", [])
        if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
            fail(f"{location}.id must be a positive integer")
        if job_id in jobs:
            fail(f"{location}.id duplicates job {job_id}")
        if not isinstance(name, str) or not name:
            fail(f"{location}.name must be a non-empty string")
        if conclusion is not None and not isinstance(conclusion, str):
            fail(f"{location}.conclusion must be a string or null")
        if not isinstance(steps, list):
            fail(f"{location}.steps must be an array")

        numbers = set()
        for step_index, step in enumerate(steps):
            number = step.get("number") if isinstance(step, dict) else None
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                fail(f"{location}.steps[{step_index}].number must be positive")
            numbers.add(number)

        jobs[job_id] = name
        step_numbers[job_id] = numbers
        if conclusion in FAILED_CONCLUSIONS:
            failed_job_ids.add(job_id)
        elif conclusion == "cancelled":
            cancelled_job_ids.append(job_id)

    return jobs, step_numbers, failed_job_ids, cancelled_job_ids


def validate_job_references(analysis, step_numbers, failed_job_ids):
    grouped_job_ids = set()
    for group_index, group in enumerate(analysis["groups"]):
        location = f"$.groups[{group_index}]"
        job_ids = group["job_ids"]
        if not job_ids:
            fail(f"{location}.job_ids must not be empty")
        if len(job_ids) != len(set(job_ids)):
            fail(f"{location}.job_ids contains duplicates")
        for job_id in job_ids:
            if job_id not in failed_job_ids:
                fail(f"{location}.job_ids references non-failed job {job_id}")
            if job_id in grouped_job_ids:
                fail(f"job {job_id} appears in more than one failure group")
            grouped_job_ids.add(job_id)

        group_job_ids = set(job_ids)
        for evidence_index, evidence in enumerate(group["evidence"]):
            evidence_location = f"{location}.evidence[{evidence_index}]"
            job_id = evidence["job_id"]
            step_number = evidence["step_number"]
            if job_id not in group_job_ids:
                fail(f"{evidence_location}.job_id is not in the failure group")
            if step_number != 0 and step_number not in step_numbers[job_id]:
                fail(f"{evidence_location}.step_number does not exist for job {job_id}")


def job_link(job_id, jobs, repository, run_id, step_number=0):
    label = sanitize_inline(jobs.get(job_id, f"Job {job_id}"), limit=300)
    url = job_url(repository, run_id, job_id)
    if not url:
        return label
    if step_number > 0:
        label += f", step {step_number}"
        url += f"#step:{step_number}:1"
    return f"[{label}]({url})"


def render_evidence(group, jobs, repository, run_id):
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


def render_group(index, group, jobs, repository, run_id, head_sha):
    job_ids = unique(group["job_ids"])
    open_attribute = " open" if index == 1 else ""
    lines = [
        f"<details{open_attribute}>",
        (
            "<summary><strong>"
            f"{index}. {sanitize_html(group['title'], limit=100)}"
            "</strong>"
            f" &middot; {len(job_ids)} {plural(len(job_ids), 'job')}"
            "</summary>"
        ),
        "",
        f"**Explanation:** {sanitize_inline(group['explanation'])}",
    ]

    evidence = render_evidence(group, jobs, repository, run_id)
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
            f"- {jobs.get(job_id, f'Job {job_id}')}: "
            f"{job_url(repository, run_id, job_id) or '(invalid job ID)'}"
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


def render_report(
    analysis,
    jobs,
    cancelled_job_ids,
    repository,
    run_id,
    head_sha,
):
    lines = []

    for index, group in enumerate(analysis["groups"], start=1):
        lines.extend(render_group(index, group, jobs, repository, run_id, head_sha))
        lines.append("")

    if cancelled_job_ids:
        lines.extend(
            [
                "<details>",
                "<summary><strong>Cancelled jobs</strong></summary>",
                "",
                *[
                    f"- {job_link(job_id, jobs, repository, run_id)}"
                    for job_id in cancelled_job_ids
                ],
                "",
                "</details>",
                "",
            ]
        )

    report = "\n".join(lines) + "\n"
    if len(report.encode("utf-8")) > REPORT_LIMIT:
        fail(f"rendered report exceeds {REPORT_LIMIT:,} bytes")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render schema-constrained CI triage output as safe Markdown."
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        fail("repository must have the form owner/name")
    if not re.fullmatch(r"[1-9][0-9]*", args.run_id):
        fail("run id must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{40}", args.head_sha):
        fail("head SHA must be 40 lowercase hexadecimal characters")

    schema = load_json(args.schema)
    analysis = load_json(args.analysis)
    validate_schema(analysis, schema)
    jobs, step_numbers, failed_job_ids, cancelled_job_ids = load_job_manifest(args.jobs)
    validate_job_references(analysis, step_numbers, failed_job_ids)
    report = render_report(
        analysis,
        jobs,
        cancelled_job_ids,
        args.repository,
        args.run_id,
        args.head_sha,
    )
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        safe_error = str(error).replace("\r", "\\r").replace("\n", "\\n")
        print(f"error: invalid CI triage output: {safe_error}", file=sys.stderr)
        raise SystemExit(1)
