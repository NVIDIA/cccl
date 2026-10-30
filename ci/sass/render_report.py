#!/usr/bin/env python3
"""Make the SASS comparison PR comment from the results of `sass_diff.sh`."""

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_sass import Status

# A header change can touch every target on every architecture, and a GitHub
# comment holds 65536 characters. Thus the comment names this many targets and
# shows the diff of this many, one architecture each. The rest are counted.
_MAX_LISTED_TARGETS = 25
_MAX_DIFF_BLOCKS = 10

# The sort rank and the label of each classification.
_CLASSIFICATIONS = {
    "significant": (0, "⚠️ Significant"),
    "unclear": (1, "❓ Unclear"),
    "benign": (2, "✅ Potentially Benign"),
}

# The model controls how much it writes, and the comment holds 65536 characters.
_MAX_TITLE_CHARS = 120
_MAX_EXPLANATION_CHARS = 600

# GitHub rejects a comment over 65536 characters so we need a max bound on the size of the
# comment.
_MAX_COMMENT_BYTES = 60_000

_MATRIX_PATH = Path(__file__).resolve().parent.parent / "matrix.yaml"

_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# The markdown that can escape one line: a code span, emphasis, or a link. An
# `_` between two alphanumerics opens no emphasis in CommonMark, and escaping
# it would put a backslash in the middle of every `sm_90`.
_INLINE_MARKDOWN = re.compile(r"[\\`*\[\]]|(?<![0-9A-Za-z])_|_(?![0-9A-Za-z])")


@dataclass(frozen=True)
class Comparison:
    meta: dict[str, Any]
    report: dict[str, Any]
    analysis: dict[str, dict[str, Any]] | None = None


@cache
def _matrix() -> dict[str, dict[str, Any]]:
    with _MATRIX_PATH.open() as f:
        return yaml.safe_load(f)


@cache
def _project_name(project: str) -> str:
    return _matrix()["projects"][project]["name"]


def _code(value: object) -> str:
    """Wrap a value in a markdown code span."""
    return f"`{str(value).replace('`', '')}`"


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _rank(group: dict[str, Any] | None) -> int:
    return _CLASSIFICATIONS[group["classification"] if group else "unclear"][0]


def _render_how_to_benchmark(
    changed: list[tuple[Comparison, list[dict[str, Any]]]],
) -> list[str]:
    """One fenced block that says how to request a benchmark run."""
    filters = []
    for comparison, targets in changed:
        # A target that this PR removed cannot be benchmarked.
        names = [
            target["target"]
            for target in targets
            if Status(target["status"]) is not Status.REMOVED
        ]
        if len(names) > _MAX_LISTED_TARGETS:
            project = comparison.meta["project"]
            filters.append(
                f"      - '^{project}\\.bench\\.'   # all {len(names)} changed targets"
            )
        else:
            filters.extend(f"      - '^{re.escape(name)}$'" for name in names)

    return [
        "<details>",
        "<summary><strong>How to request a benchmark run</strong></summary>",
        "",
        "```",
        "Request a benchmark run for this PR:",
        "",
        "1. Replace the `benchmarks:` block of ci/bench.yaml with exactly this:",
        "",
        "benchmarks:",
        "  filters:",
        "    cpp:",
        *filters,
        "  gpus:",
        '    - "h100"   # pick the GPUs that this change can affect',
        "",
        "2. Commit with `[bench-only]` at the end of the commit summary, so that",
        "   the unrelated CI jobs are skipped. Then push.",
        "",
        "ci/bench.yaml must match ci/bench.template.yaml before the PR can merge.",
        "Reset it once the measurement is done.",
        "```",
        "",
        "</details>",
    ]


def _rejected(reason: str) -> dict[str, Any]:
    """The group that stands in for one the model got wrong."""
    return {
        "classification": "unclear",
        "title": "No usable classification",
        "explanation": f"The renderer rejected the model output: {reason}.",
    }


def _fault(group: Any) -> str | None:
    """What makes `group` unusable, or None when nothing does.

    `codex-action` holds the model to `model-output.schema.json`, but the
    renderer reads a file, not a promise.
    """
    if not isinstance(group, dict):
        return "the group was not an object"
    if (value := group.get("classification")) not in _CLASSIFICATIONS:
        return f"the classification was {json.dumps(value)}"
    if not all(
        isinstance(group.get(field), str) and group[field].strip()
        for field in ("title", "explanation")
    ):
        return "the title or the explanation was empty"
    if not isinstance(group.get("diffs"), list):
        return "the diff list was not a list"
    return None


def load_analysis(path: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map every changed diff to the group that classifies it.

    A diff that the model got wrong or left out still gets a group, so one bad
    answer costs its own diffs and no others. This never raises: the comparison
    result must reach the pull request even when the model output is garbage.
    """
    # A target name holds dots, so a key is compared whole and never split apart.
    known = {
        f"{target['target']}.{arch['arch']}"
        for target in report["targets"]
        for arch in target["archs"]
        if arch["changed"]
    }

    try:
        with path.open() as fd:
            groups = json.load(fd)["groups"]
        if not isinstance(groups, list):
            raise TypeError("groups is not a list")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as error:
        print(f"warning: {path} is unusable: {error}", file=sys.stderr)
        return dict.fromkeys(known, _rejected("the output file was unreadable"))

    by_diff: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(groups):
        location = f"$.groups[{index}]"
        if fault := _fault(group):
            print(f"warning: {location} is unusable: {fault}", file=sys.stderr)
            # A rejected group still names the diffs it meant to cover, so they
            # can carry the real reason instead of a bare "not classified".
            named = group.get("diffs") if isinstance(group, dict) else None
            group = _rejected(fault)
            if not isinstance(named, list):
                continue
            group["diffs"] = named

        for key in group["diffs"]:
            if key not in known:
                print(
                    f"warning: {location} names unknown diff {key!r}", file=sys.stderr
                )
            elif key in by_diff:
                print(f"warning: {location} repeats diff {key!r}", file=sys.stderr)
            else:
                by_diff[key] = group

    for key in known - set(by_diff):
        by_diff[key] = _rejected("the model did not classify this difference")
    return by_diff


def load_comparison(directory: Path) -> Comparison:
    with (directory / "report.json").open() as fd:
        report = json.load(fd)
    with (directory / "meta.json").open() as fd:
        meta = json.load(fd)

    analysis_path = directory / "analysis.json"
    analysis = load_analysis(analysis_path, report) if analysis_path.exists() else None
    return Comparison(meta=meta, report=report, analysis=analysis)


def _text(value: str, limit: int) -> str:
    """Make model text safe for one line of markdown.

    The model quotes the diffs, which hold whatever the compiler emitted, so
    every construct that can escape its line is neutralized, not trusted.
    """
    value = _CONTROL_CHARACTER.sub("", value)
    value = " ".join(value.split())
    if len(value) > limit:
        value = value[:limit].rstrip() + "..."
    value = html.escape(value, quote=True).replace("|", "&#124;")
    return _INLINE_MARKDOWN.sub(lambda m: "\\" + m.group(0), value)


def _render_diff_block(
    target: dict[str, Any],
    arch: dict[str, Any],
    group: dict[str, Any] | None,
    artifacts_url: str,
) -> list[str]:
    diff = arch["diff"]
    excerpt = diff["excerpt"]
    label = explanation = ""
    if group:
        label = f" - {_CLASSIFICATIONS[group['classification']][1]}"
        explanation = (
            f"**{_text(group['title'], _MAX_TITLE_CHARS)}** "
            f"{_text(group['explanation'], _MAX_EXPLANATION_CHARS)}\n"
        )

    return [
        "<details>",
        f"<summary><code>{target['target']} - {arch['arch']}</code>{label}</summary>",
        "",
        explanation,
        f"_Showing {len(excerpt)}/{diff['total_lines']} diff lines, "
        f"{diff['changed_lines']} changes._ - "
        f"[⬇️ Full diff]({artifacts_url})",
        "",
        # ```diff makes GitHub colour the `-` and `+` lines.
        "```diff",
        *excerpt,
        "```",
        "</details>",
    ]


def _render_diffs(
    changed: list[tuple[Comparison, list[dict[str, Any]]]], budget: int
) -> list[str]:
    """Render one collapsed diff excerpt per changed target.

    Architectures can have different classifications, so show the one with the
    lowest classification rank. Keep the first diff when ranks tie. A target
    that this PR added or removed has no diff at all.

    A header change makes hundreds of diffs, so the classification decides
    which ones keep a block. `budget` is the bytes left for this section, and a
    block that does not fit it is dropped with the ones behind it.
    """
    with_diff = []
    for comparison, targets in changed:
        analysis = comparison.analysis or {}
        artifacts_url = comparison.meta["artifacts_url"]
        for target in targets:
            candidates = [
                (target, arch, analysis.get(f"{target['target']}.{arch['arch']}"))
                for arch in target["archs"]
                if arch["diff"]
            ]
            if candidates:
                with_diff.append(
                    (*min(candidates, key=lambda e: _rank(e[2])), artifacts_url)
                )

    if not with_diff:
        return []

    with_diff.sort(key=lambda e: _rank(e[2]))

    header = [
        "",
        "## ‼️  Summary of Differences ‼️ ",
        "",
        f"Showing {len(with_diff)}/{len(with_diff)} summaries.",
    ]
    budget -= len("\n".join(header).encode())

    blocks: list[list[str]] = []
    for target, arch, group, artifacts_url in with_diff[:_MAX_DIFF_BLOCKS]:
        block = _render_diff_block(target, arch, group, artifacts_url)
        budget -= len("\n".join(block).encode()) + 1
        if budget < 0:
            break
        blocks.append(block)

    header[-1] = f"Showing {len(blocks)}/{len(with_diff)} summaries."
    return [*header, *(line for block in blocks for line in block)]


def _render_changed_table(targets: list[dict[str, Any]]) -> list[str]:
    lines = [
        "<details>",
        "<summary><strong>Targets with a SASS change</strong></summary>",
        "",
        "| Target | Architectures with a SASS change |",
        "| --- | --- |",
    ]
    for target in targets[:_MAX_LISTED_TARGETS]:
        status = Status(target["status"])
        if status is Status.ADDED:
            detail = "target added by this PR"
        elif status is Status.REMOVED:
            detail = "target removed by this PR"
        else:
            detail = ", ".join(
                _code(a["arch"]) for a in target["archs"] if a["changed"]
            )
        lines.append(f"| {_code(target['target'])} | {detail} |")

    if (remainder := len(targets) - _MAX_LISTED_TARGETS) > 0:
        lines.append(f"| _... and {remainder} more_ | |")

    lines.extend(
        [
            "",
            "</details>",
        ]
    )
    return lines


def render(comparisons: list[Comparison]) -> str:
    """Render the markdown for the PR comment."""
    runs = {
        (c.meta["base_ref"], c.meta["test_ref"], c.meta["arch"]) for c in comparisons
    }
    if len(runs) != 1:
        raise ValueError(
            f"The comparisons used different refs or architectures: {sorted(runs)}"
        )
    ((base_ref, test_ref, arch),) = runs

    # The comparison reports in whatever order it walked the dumps. Sort here,
    # so that the table is stable between runs and two comments can be compared,
    # and so that the truncation below always drops the same targets.
    changed = [
        (
            comparison,
            sorted(
                (t for t in comparison.report["targets"] if t["changed"]),
                key=lambda t: t["target"],
            ),
        )
        for comparison in comparisons
    ]
    all_changed = [target for _, targets in changed for target in targets]

    if all_changed:
        counts = _join(
            [
                f"{len(targets)} of {len(c.report['targets'])} "
                f"{_project_name(c.meta['project'])}"
                for c, targets in changed
            ]
        )
        headline = [
            (
                f"⚠️ **The SASS changed for {counts} benchmark target(s). "
                "A benchmark run may be necessary**"
            ),
            "",
            *_render_how_to_benchmark([entry for entry in changed if entry[1]]),
        ]
    else:
        counts = _join(
            [
                f"{len(c.report['targets'])} {_project_name(c.meta['project'])}"
                for c in comparisons
            ]
        )
        headline = [
            f"✅ **No SASS change in any of the {counts} benchmark target(s).**"
        ]

    lines = [
        "<!-- cccl-sass-diff -->",
        "## 🔬 Benchmark SASS comparison",
        "",
        *headline,
        "",
        "| Run | Value |",
        "| --- | --- |",
        f"| Baseline | {_code(base_ref)} |",
        f"| Tested | {_code(test_ref)} |",
        f"| Architectures | {_code(arch)} |",
        "",
    ]

    if all_changed:
        lines.extend(_render_changed_table(all_changed))
        budget = _MAX_COMMENT_BYTES - len("\n".join(lines).encode())
        lines.extend([*_render_diffs(changed, budget), ""])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make the PR comment from the results of `sass_diff.sh`.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help=(
            "Directory with report.json, meta.json and an optional analysis.json. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the markdown comment.",
    )
    args = parser.parse_args()

    text = render([load_comparison(directory) for directory in args.input])
    args.output.write_text(text)


if __name__ == "__main__":
    main()
