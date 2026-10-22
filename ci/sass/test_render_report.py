#!/usr/bin/env python3
"""Tests for ci/sass/render_report.py.

Run with: python3 -m pytest ci/sass/test_render_report.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_report import (  # noqa: E402
    _MAX_COMMENT_BYTES,  # noqa: E402
    load_analysis,
    render,
)
from render_report import _MAX_DIFF_BLOCKS as _MAX  # noqa: E402
from render_report import _MAX_LISTED_TARGETS as _MAX_TARGETS  # noqa: E402

#: A diff of the shape that `compare_sass.py` writes.
DIFF = {
    "excerpt": [
        "--- base/demo.sm_90",
        "+++ test/demo.sm_90",
        "@@ -1,3 +1,3 @@",
        "-MOV R0, 0x3f800000",
        "+MOV R0, 0x40000000",
    ],
    "changed_lines": 2,
    "total_lines": 5,
    "path": "demo.sm_90.diff",
}


def _report(targets: list[dict[str, object]]) -> dict[str, object]:
    """Build a report of the shape that `compare_sass.py` writes."""
    changed = [target for target in targets if target["changed"]]
    return {
        "summary": {
            "targets_compared": len(targets),
            "targets_changed": len(changed),
            "changed": bool(changed),
        },
        "targets": targets,
    }


def _target(
    name: str,
    *,
    changed: bool = True,
    status: str = "compared",
    archs: list[str] | None = None,
    diff: dict[str, object] | None = None,
) -> dict[str, object]:
    # An explicit empty list must survive: `compare_sass.py` gives an added or
    # removed target no per-architecture results at all. Only an omitted `archs`
    # takes the default.
    if archs is None:
        archs = ["sm_90"]
    return {
        "target": name,
        "status": status,
        "changed": changed,
        "archs": [
            {
                "arch": arch,
                "changed": changed,
                "status": status,
                "diff": diff if changed else None,
            }
            for arch in archs
        ],
    }


def _render(targets: list[dict[str, object]], **kwargs: Any) -> str:
    return render(
        _report(targets),
        base_ref="origin/main",
        test_ref="HEAD",
        arch="all-major-cccl",
        **{"artifacts_url": "", **kwargs},
    )


def _group(
    classification: str = "benign",
    *,
    title: str = "Register numbers differ in the agent kernel",
    explanation: str = "The instruction mix is the same on both sides.",
    diffs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a group of the shape that the model answers with."""
    return {
        "classification": classification,
        "title": title,
        "explanation": explanation,
        "diffs": ["cub.bench.scan.base.sm_90"] if diffs is None else diffs,
    }


def test_the_comment_offers_ready_to_paste_bench_filters() -> None:
    """The filters are regexes, thus the dots must be escaped and anchored."""
    text = _render([_target("cub.bench.reduce.sum.base")])
    assert r"- '^cub\.bench\.reduce\.sum\.base$'" in text
    assert "ci/bench.yaml" in text
    assert "[bench-only]" in text


def test_a_removed_target_is_not_offered_as_a_benchmark() -> None:
    """A target this PR deleted cannot be benchmarked."""
    text = _render([_target("cub.bench.gone.base", status="removed", archs=[])])
    assert "target removed by this PR" in text
    assert r"^cub\.bench\.gone\.base$" not in text


def test_no_benchmark_advice_when_nothing_changed() -> None:
    text = _render([_target("cub.bench.reduce.sum.base", changed=False)])
    assert "How to request a benchmark run" not in text
    assert "No SASS change" in text


def test_only_the_changed_architectures_are_named() -> None:
    text = _render([_target("cub.bench.scan.base", archs=["sm_90", "sm_100"])])
    assert "| `cub.bench.scan.base` | `sm_90`, `sm_100` |" in text


def test_the_targets_are_listed_in_a_stable_order() -> None:
    """The table truncates, so the order decides which targets are dropped."""
    names = ["cub.bench.scan.base", "cub.bench.adjacent.base", "cub.bench.merge.base"]
    text = _render([_target(name) for name in names])
    positions = [text.index(name) for name in sorted(names)]
    assert positions == sorted(positions)


def test_the_fragment_carries_the_sticky_comment_marker() -> None:
    """`bench-results` merges this fragment into one comment, so it must not
    look like a comment of its own."""
    text = _render([_target("cub.bench.reduce.sum.base")])
    assert text.startswith("<!-- cccl-sass-diff -->")


# ============================================================================
# The diff excerpt
# ============================================================================


def test_the_comment_shows_the_diff() -> None:
    text = _render([_target("cub.bench.scan.base", diff=DIFF)])
    assert "```diff" in text
    assert "-MOV R0, 0x3f800000" in text
    assert "+MOV R0, 0x40000000" in text
    assert "<code>cub.bench.scan.base - sm_90</code>" in text
    assert "2 changes." in text


def test_the_diff_carries_a_download_link() -> None:
    with_url = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        artifacts_url="https://example.invalid/run#artifacts",
    )
    assert "[⬇️ Full diff](https://example.invalid/run#artifacts)" in with_url


def test_a_truncated_diff_says_how_much_it_left_out() -> None:
    text = _render(
        [_target("cub.bench.scan.base", diff={**DIFF, "total_lines": 900})],
        artifacts_url="https://example.invalid/run#artifacts",
    )
    assert "Showing 5/900 diff lines, 2 changes." in text


def test_the_diff_blocks_are_capped() -> None:
    """A header change touches every target, so the comment must stay small."""
    targets = [_target(f"cub.bench.t{i}.base", diff=DIFF) for i in range(_MAX + 4)]
    text = _render(targets)
    assert text.count("```diff") == _MAX
    # The count tells the reader that the list is not complete.
    assert f"Showing {_MAX}/{_MAX + 4} summaries." in text


def test_a_target_without_a_diff_gets_no_block() -> None:
    """An added or removed target has no second side, so it has no diff.

    The whole section goes away, not only the fenced block, thus the heading
    must be absent too.
    """
    text = _render([_target("cub.bench.gone.base", status="removed", archs=[])])
    assert "```diff" not in text
    assert "Summary of Differences" not in text


# ============================================================================
# How to request a benchmark run
# ============================================================================


def test_the_instructions_are_one_copyable_block() -> None:
    """GitHub puts a copy button on a fenced block, which is the whole point."""
    text = _render([_target("cub.bench.scan.base")])
    body = text.split("How to request a benchmark run", 1)[1].split("```")[1]
    # Everything the reader must do is inside the one block.
    assert "ci/bench.yaml" in body
    assert "'^cub\\.bench\\.scan\\.base$'" in body
    assert "[bench-only]" in body


def test_a_short_target_list_is_named_in_full() -> None:
    names = ["cub.bench.scan.base", "cub.bench.merge.base"]
    text = _render([_target(name) for name in names])
    for name in names:
        assert f"'^{name.replace('.', chr(92) + '.')}$'" in text
    assert "'^cub\\.bench\\.'" not in text


def test_a_long_target_list_becomes_one_regex() -> None:
    """A truncated list is neither short nor complete, so it is not used."""
    targets = [_target(f"cub.bench.t{i}.base") for i in range(_MAX_TARGETS + 1)]
    text = _render(targets)
    assert f"'^cub\\.bench\\.'   # all {_MAX_TARGETS + 1} changed targets" in text
    assert "'^cub\\.bench\\.t0\\.base$'" not in text


# ============================================================================
# The triage of the differences
# ============================================================================


def _load(
    groups: list[dict[str, Any]],
    targets: list[dict[str, object]],
    tmp_path: Path,
) -> dict[str, dict[str, Any]]:
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps({"groups": groups}))
    return load_analysis(path, _report(targets))


def _classified(names: dict[str, str]) -> dict[str, dict[str, Any]]:
    """An analysis of the shape `load_analysis` returns, target name to class."""
    return {
        f"{name}.sm_90": _group(classification, title=f"{classification} change")
        for name, classification in names.items()
    }


def test_a_block_holds_no_classification_without_an_analysis() -> None:
    """`--analysis` is optional, and every existing caller omits it."""
    text = _render([_target("cub.bench.scan.base", diff=DIFF)])
    assert "<summary><code>cub.bench.scan.base - sm_90</code></summary>" in text


def test_a_block_carries_its_own_classification() -> None:
    text = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        analysis={
            "cub.bench.scan.base.sm_90": _group(
                "significant",
                title="LDG widened to LDG.128",
                explanation="The load moves 128 bits where it moved 32.",
            )
        },
    )
    assert (
        "<summary><code>cub.bench.scan.base - sm_90</code>"
        " - ⚠️ Significant</summary>" in text
    )
    assert (
        "**LDG widened to LDG.128** The load moves 128 bits where it moved 32." in text
    )


@pytest.mark.parametrize(
    "classifications, expected_arch",
    [
        (["benign", "unclear", "significant"], "sm_100"),
        (["significant", "unclear", "benign"], "sm_80"),
        (["benign", "unclear", "benign"], "sm_90"),
        (["benign", "significant", "significant"], "sm_90"),
        (None, "sm_80"),
    ],
)
def test_the_highest_priority_architecture_represents_the_target(
    classifications: list[str] | None, expected_arch: str
) -> None:
    archs = ["sm_80", "sm_90", "sm_100"]
    target = _target("cub.bench.scan.base", archs=archs, diff=DIFF)
    for arch in target["archs"]:
        arch["diff"] = {**DIFF, "excerpt": [f"+Evidence for {arch['arch']}"]}
    analysis = (
        {
            f"cub.bench.scan.base.{arch}": _group(classification)
            for arch, classification in zip(archs, classifications)
        }
        if classifications
        else None
    )

    text = _render([target], analysis=analysis)
    assert text.count("```diff") == 1
    assert f"<code>cub.bench.scan.base - {expected_arch}</code>" in text
    assert f"+Evidence for {expected_arch}" in text
    for arch in archs:
        if arch != expected_arch:
            assert f"+Evidence for {arch}" not in text


def test_a_significant_diff_is_never_dropped_for_a_benign_one() -> None:
    """A header change makes hundreds of diffs, and only 10 get a block."""
    names = {f"cub.bench.t{i}.base": "benign" for i in range(_MAX + 4)}
    names["cub.bench.zzz.base"] = "significant"
    targets = [_target(name, diff=DIFF) for name in names]
    targets[-1] = _target("cub.bench.zzz.base", archs=["sm_80", "sm_90"], diff=DIFF)
    analysis = _classified(names)
    analysis["cub.bench.zzz.base.sm_80"] = _group("benign")

    text = _render(targets, analysis=analysis)
    assert text.count("```diff") == _MAX
    assert "cub.bench.zzz.base - sm_90</code> - ⚠️ Significant" in text


def test_the_blocks_are_ordered_by_classification() -> None:
    names = {
        "cub.bench.a.base": "benign",
        "cub.bench.b.base": "unclear",
        "cub.bench.c.base": "significant",
    }
    targets = [_target(name, diff=DIFF) for name in names]

    diffs = _render(targets, analysis=_classified(names)).split(
        "Summary of Differences"
    )[1]
    positions = [diffs.index(name) for name in ("c.base", "b.base", "a.base")]
    assert positions == sorted(positions)


def test_the_classification_text_cannot_break_the_comment() -> None:
    """The model quotes disassembly, so its text is data, not markup."""
    text = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        analysis={
            "cub.bench.scan.base.sm_90": _group(
                title="<script>x</script>", explanation="a | b"
            )
        },
    )
    assert "<script>x" not in text
    assert "&lt;script&gt;" in text
    assert "a &#124; b" in text


def _classes(analysis: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {key: group["classification"] for key, group in analysis.items()}


def test_every_changed_diff_gets_a_group(tmp_path: Path) -> None:
    """The renderer keys off the map, so a gap in it would drop a block."""
    analysis = _load(
        [_group("benign", diffs=["cub.bench.a.base.sm_90"])],
        [
            _target("cub.bench.a.base", diff=DIFF),
            _target("cub.bench.b.base", diff=DIFF),
            _target("cub.bench.same.base", changed=False),
        ],
        tmp_path,
    )
    assert _classes(analysis) == {
        "cub.bench.a.base.sm_90": "benign",
        "cub.bench.b.base.sm_90": "unclear",
    }


def test_an_invented_diff_name_is_dropped(tmp_path: Path) -> None:
    analysis = _load(
        [_group(diffs=["cub.bench.invented.base.sm_90", "cub.bench.scan.base.sm_90"])],
        [_target("cub.bench.scan.base", diff=DIFF)],
        tmp_path,
    )
    assert _classes(analysis) == {"cub.bench.scan.base.sm_90": "benign"}


def test_the_first_group_keeps_a_repeated_diff(tmp_path: Path) -> None:
    analysis = _load(
        [_group("benign"), _group("significant")],
        [_target("cub.bench.scan.base", diff=DIFF)],
        tmp_path,
    )
    assert _classes(analysis) == {"cub.bench.scan.base.sm_90": "benign"}


@pytest.mark.parametrize(
    "group",
    [
        "not an object",
        {**_group(), "classification": "catastrophic"},
        {**_group(), "title": "  "},
        {**_group(), "explanation": ""},
        {**_group(), "diffs": "cub.bench.scan.base.sm_90"},
        {k: v for k, v in _group().items() if k != "explanation"},
    ],
    ids=[
        "not-object",
        "bad-class",
        "blank-title",
        "no-explanation",
        "diffs-str",
        "missing",
    ],
)
def test_a_bad_group_is_unclear(group: Any, tmp_path: Path) -> None:
    """`codex-action` checks the schema, but the renderer reads a file."""
    analysis = _load([group], [_target("cub.bench.scan.base", diff=DIFF)], tmp_path)
    assert _classes(analysis) == {"cub.bench.scan.base.sm_90": "unclear"}
    assert "renderer rejected" in analysis["cub.bench.scan.base.sm_90"]["explanation"]


@pytest.mark.parametrize(
    "content", ["not json", "[]", '{"groups": {}}', '{"other": []}'], ids=str
)
def test_an_unusable_file_marks_every_diff_unclear(
    content: str, tmp_path: Path
) -> None:
    path = tmp_path / "analysis.json"
    path.write_text(content)
    analysis = load_analysis(
        path,
        _report(
            [
                _target("cub.bench.a.base", diff=DIFF),
                _target("cub.bench.b.base", diff=DIFF),
            ]
        ),
    )
    assert set(_classes(analysis).values()) == {"unclear"}
    assert len(analysis) == 2


def test_a_missing_file_marks_every_diff_unclear(tmp_path: Path) -> None:
    analysis = load_analysis(
        tmp_path / "absent.json", _report([_target("cub.bench.a.base", diff=DIFF)])
    )
    assert _classes(analysis) == {"cub.bench.a.base.sm_90": "unclear"}


# ============================================================================
# The model text is data, not markup
# ============================================================================


def test_a_backtick_cannot_open_a_code_span() -> None:
    text = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        analysis={
            "cub.bench.scan.base.sm_90": _group(title="`x` *y* [z](http://a) _w_")
        },
    )
    assert "\\`x\\` \\*y\\* \\[z\\]" in text


def test_an_arch_name_keeps_its_underscore() -> None:
    """`sm_90` opens no emphasis in CommonMark, so it needs no backslash."""
    text = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        analysis={
            "cub.bench.scan.base.sm_90": _group(
                title="A change on sm_90 and sm_100",
                explanation="It also shows _emphasis_ and a _ on its own.",
            )
        },
    )
    assert "sm_90 and sm_100" in text
    assert "\\_emphasis\\_" in text
    assert "a \\_ on its own" in text


def test_a_control_character_is_removed() -> None:
    """A disassembly holds whatever the compiler emitted."""
    text = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        analysis={"cub.bench.scan.base.sm_90": _group(title="a\x00b\x1bc")},
    )
    assert "\x00" not in text
    assert "\x1b" not in text
    assert "abc" in text


def test_long_model_text_is_cut_and_marked() -> None:
    text = _render(
        [_target("cub.bench.scan.base", diff=DIFF)],
        analysis={"cub.bench.scan.base.sm_90": _group(explanation="w " * 900)},
    )
    assert "w w..." in text
    assert len(text) < 3000


def test_the_comment_stays_under_the_github_limit() -> None:
    """A model that writes to its cap on every block must not lose the comment."""
    names = {f"cub.bench.t{i}.base": "significant" for i in range(_MAX)}
    targets = [
        _target(name, diff={**DIFF, "excerpt": ["+X" * 400] * 40}) for name in names
    ]
    analysis = {
        f"{name}.sm_90": _group("significant", title="t" * 500, explanation="e" * 5000)
        for name in names
    }

    text = _render(targets, analysis=analysis)
    assert len(text.encode()) <= _MAX_COMMENT_BYTES
    assert "Summary of Differences" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
