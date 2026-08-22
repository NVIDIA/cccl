#!/usr/bin/env python3
"""Tests for ci/sass/render_report.py.

Run with: python3 -m pytest ci/sass/test_render_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_report import _MAX_DIFF_BLOCKS as _MAX  # noqa: E402
from render_report import _MAX_LISTED_TARGETS as _MAX_TARGETS  # noqa: E402
from render_report import render  # noqa: E402

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


def _render(targets: list[dict[str, object]], **kwargs: str) -> str:
    return render(
        _report(targets),
        base_ref="origin/main",
        test_ref="HEAD",
        arch="all-major-cccl",
        **{"artifacts_url": "", **kwargs},
    )


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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
