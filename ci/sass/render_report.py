#!/usr/bin/env python3
"""Make the SASS comparison PR comment from the report of `compare_sass.py`.

The comment says which targets changed, shows the first lines of each diff, and
tells the author how to request a benchmark run. It does not run the benchmarks
and does not say that the performance changed.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_sass import Status  # noqa: E402

# A header change can touch every target on every architecture, and a GitHub
# comment holds 65536 characters. Thus the comment names this many targets and
# shows the diff of this many, one architecture each. The rest are counted.
_MAX_LISTED_TARGETS = 25
_MAX_DIFF_BLOCKS = 10


def _code(value: object) -> str:
    """Wrap a value in a markdown code span."""
    return f"`{str(value).replace('`', '')}`"


def _render_how_to_benchmark(targets: list[dict[str, Any]]) -> list[str]:
    """One fenced block that says how to request a benchmark run.

    Fenced, because GitHub puts a copy button on every fenced block: the reader
    pastes the whole task into an agent with one click. The filters are anchored
    regexes on the target names, and a list longer than `_MAX_LISTED_TARGETS`
    becomes one regex over the benchmark tree. A truncated list is neither short
    nor complete.
    """
    # A target that this PR removed cannot be benchmarked.
    names = [
        target["target"]
        for target in targets
        if Status(target["status"]) is not Status.REMOVED
    ]
    if len(names) > _MAX_LISTED_TARGETS:
        filters = [f"      - '^cub\\.bench\\.'   # all {len(names)} changed targets"]
    else:
        filters = [f"      - '^{re.escape(name)}$'" for name in names]

    return [
        "<details>",
        "<summary><strong>How to request a benchmark run</strong></summary>",
        "",
        "```",
        "Request a CUB benchmark run for this PR:",
        "",
        "1. Replace the `benchmarks:` block of ci/bench.yaml with exactly this:",
        "",
        "benchmarks:",
        "  filters:",
        "    cub:",
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


def _render_diffs(targets: list[dict[str, Any]], artifacts_url: str) -> list[str]:
    """Render one collapsed diff excerpt per changed target.

    A code change usually affects every architecture in the same way, thus the
    first architecture with a diff is enough to show. A target that this PR
    added or removed has no diff at all.
    """
    with_diff = []
    for target in targets:
        for arch in target["archs"]:
            if arch["diff"]:
                with_diff.append((target, arch))
                break

    if not with_diff:
        return []

    shown = with_diff[:_MAX_DIFF_BLOCKS]

    lines = [
        "",
        "## ‼️  Summary of Differences ‼️ ",
        "",
        f"Showing {len(shown)}/{len(with_diff)} summaries.",
    ]
    for target, arch in shown:
        diff = arch["diff"]
        excerpt = diff["excerpt"]

        lines.extend(
            [
                "<details>",
                f"<summary><code>{target['target']} - {arch['arch']}</code></summary>",
                "",
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
        )

    return lines


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


def render(
    report: dict[str, Any],
    *,
    base_ref: str,
    test_ref: str,
    arch: str,
    artifacts_url: str,
) -> str:
    """Render the markdown fragment for the PR comment."""
    targets = report["targets"]
    # The comparison reports in whatever order it walked the dumps. Sort here,
    # so that the table is stable between runs and two comments can be compared,
    # and so that the truncation below always drops the same targets.
    changed = sorted(
        (target for target in targets if target["changed"]),
        key=lambda target: target["target"],
    )

    if changed:
        headline = [
            f"⚠️ **The SASS changed for {len(changed)} of "
            f"{len(targets)} CUB benchmark target(s). A benchmark run "
            "may be necessary**",
            "",
            *_render_how_to_benchmark(changed),
        ]
    else:
        headline = [
            f"✅ **No SASS change in any of the {len(targets)} CUB benchmark "
            "target(s).**",
        ]

    lines = [
        "<!-- cccl-sass-diff -->",
        "## 🔬 CUB benchmark SASS comparison",
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

    if changed:
        lines.extend(
            [
                *_render_changed_table(changed),
                *_render_diffs(changed, artifacts_url),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make the PR comment from the report that `compare_sass.py` wrote "
            "and the metadata that `sass_diff.sh` wrote."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="The report.json that `compare_sass.py` wrote.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="The meta.json that `sass_diff.sh` wrote.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the markdown fragment.",
    )
    parser.add_argument(
        "--artifacts-url",
        default="www.example.com",
        help="Download URL of the uploaded dumps.",
    )
    args = parser.parse_args()

    with args.report.open() as fd:
        report = json.load(fd)
    with args.meta.open() as fd:
        meta = json.load(fd)

    text = render(
        report,
        base_ref=meta["base_ref"],
        test_ref=meta["test_ref"],
        arch=meta["arch"],
        artifacts_url=args.artifacts_url,
    )
    args.output.write_text(text)


if __name__ == "__main__":
    main()
