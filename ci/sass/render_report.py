#!/usr/bin/env python3
"""Make the SASS comparison PR comment from the report of `compare_sass.py`.

The comment says which targets changed, shows the first lines of each diff, and
tells the author how to request a benchmark run. It does not run the benchmarks
and does not say that the performance changed.

With `--analysis` it also shows how a model classified each difference.
"""

import argparse
import html
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

#: The sort rank and the label of each classification.
_CLASSIFICATIONS = {
    "significant": (0, "⚠️ Significant"),
    "unclear": (1, "❓ Unclear"),
    "benign": (2, "✅ Potentially Benign"),
}

# The model controls how much it writes, and the comment holds 65536 characters.
_MAX_TITLE_CHARS = 120
_MAX_EXPLANATION_CHARS = 600

#: What GitHub accepts in one comment. The renderer stays under it, because a
#: comment over the limit is rejected whole.
_MAX_COMMENT_BYTES = 60000

_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: The markdown that can escape one line: a code span, emphasis, or a link. An
#: `_` between two alphanumerics opens no emphasis in CommonMark, and escaping
#: it would put a backslash in the middle of every `sm_90`.
_INLINE_MARKDOWN = re.compile(r"[\\`*\[\]]|(?<![0-9A-Za-z])_|_(?![0-9A-Za-z])")


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
    targets: list[dict[str, Any]],
    artifacts_url: str,
    analysis: dict[str, dict[str, Any]],
    budget: int,
) -> list[str]:
    """Render one collapsed diff excerpt per changed target.

    Architectures can have different classifications, so show the one with the
    lowest classification rank. Keep the first diff when ranks tie or analysis
    is absent. A target that this PR added or removed has no diff at all.

    A header change makes hundreds of diffs, so the classification decides
    which ones keep a block. `budget` is the bytes left for this section, and a
    block that does not fit it is dropped with the ones behind it.
    """
    with_diff = []
    for target in targets:
        candidates = []
        for arch in target["archs"]:
            if arch["diff"]:
                key = f"{target['target']}.{arch['arch']}"
                candidates.append((target, arch, analysis.get(key)))
        if candidates:
            with_diff.append(
                min(
                    candidates,
                    key=lambda e: _CLASSIFICATIONS[e[2]["classification"]][0],
                )
                if analysis
                else candidates[0]
            )

    if not with_diff:
        return []

    if analysis:
        with_diff.sort(key=lambda e: _CLASSIFICATIONS[e[2]["classification"]][0])

    header = [
        "",
        "## ‼️  Summary of Differences ‼️ ",
        "",
        f"Showing {len(with_diff)}/{len(with_diff)} summaries.",
    ]
    budget -= len("\n".join(header).encode())

    blocks: list[list[str]] = []
    for target, arch, group in with_diff[:_MAX_DIFF_BLOCKS]:
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


def render(
    report: dict[str, Any],
    *,
    base_ref: str,
    test_ref: str,
    arch: str,
    artifacts_url: str,
    analysis: dict[str, dict[str, Any]] | None = None,
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
        lines.extend(_render_changed_table(changed))
        budget = _MAX_COMMENT_BYTES - len("\n".join(lines).encode())
        lines.extend(
            [*_render_diffs(changed, artifacts_url, analysis or {}, budget), ""]
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
    parser.add_argument(
        "--analysis",
        type=Path,
        help=(
            "Optional model classification of the diffs. Omit it to render the "
            "comment without the triage section."
        ),
    )
    args = parser.parse_args()

    with args.report.open() as fd:
        report = json.load(fd)
    with args.meta.open() as fd:
        meta = json.load(fd)

    analysis = load_analysis(args.analysis, report) if args.analysis else None

    text = render(
        report,
        base_ref=meta["base_ref"],
        test_ref=meta["test_ref"],
        arch=meta["arch"],
        artifacts_url=args.artifacts_url,
        analysis=analysis,
    )
    args.output.write_text(text)


if __name__ == "__main__":
    main()
