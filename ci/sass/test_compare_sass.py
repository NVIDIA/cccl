#!/usr/bin/env python3
"""Tests for ci/sass/compare_sass.py.

Run with: python3 -m pytest ci/sass/test_compare_sass.py

The samples below are real `cuobjdump -sass` output from CUDA 13.3, cut down to
a size that can be read. Do not replace them with hand-written text: the exact
spelling of the headers, the placement of the encoded instruction words and the
form of the branch targets are the parts that a hand-written sample gets wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from compare_sass import (  # noqa: E402
    _MAX_EXCERPT_LINES,
    Status,
    TargetResult,
    compare,
    compare_target,
    normalized_text,
)


@pytest.fixture
def compared(tmp_path: Path):
    """Call `compare_target` with a scratch work directory."""

    def call(base_dump: str, test_dump: str, target: str = "cub.bench.demo"):
        for name in ("base", "test", "diff"):
            (tmp_path / name).mkdir(exist_ok=True)
        paths = []
        for side, dump in (("base", base_dump), ("test", test_dump)):
            path = tmp_path / f"{side}.sass"
            path.write_text(dump)
            paths.append(path)
        return compare_target(*paths, target, tmp_path).archs

    return call


# Two architectures in one dump, as `cuobjdump -sass` prints them.
BASELINE = """
Fatbin elf code:
================
arch = sm_75
code version = [1,8]
host = linux
compile_size = 64bit

\tcode for sm_75
\t.target\tsm_75

\t\tFunction : _Z5otherPf
\t.headerflags\t@"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
        /*0000*/                   MOV R1, c[0x0][0x28] ;      /* 0x00000a0000017a02 */
                                                               /* 0x000fe40000000f00 */
        /*0010*/                   MOV R0, 0x3f800000 ;        /* 0x3f80000000007802 */
                                                               /* 0x000fe20000000f00 */
        /*0020*/                   EXIT ;                      /* 0x000000000000794d */
                                                               /* 0x000fea0003800000 */
        /*0030*/                   BRA 0x30;                   /* 0xfffffff000007947 */
                                                               /* 0x000fc0000383ffff */
        /*0040*/                   NOP;                        /* 0x0000000000007918 */
                                                               /* 0x000fc00000000000 */
\t\t..........


\t\tFunction : _Z6kernelIiEvPT_PKS0_i
\t.headerflags\t@"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
        /*0000*/                   S2R R4, SR_CTAID.X ;        /* 0x0000000000047919 */
                                                               /* 0x000e280000002500 */
        /*0010*/                   ISETP.GE.AND P0, PT, R4, c[0x0][0x170], PT ;
                                                               /* 0x000fd80003f06270 */
        /*0020*/              @!P0 BRA 0x50 ;                  /* 0x0000002000008947 */
                                                               /* 0x000fea0003800000 */
        /*0030*/                   IMAD R7, R2, 0x6, RZ ;      /* 0x0000000602077824 */
                                                               /* 0x004fd000078e02ff */
        /*0040*/                   EXIT ;                      /* 0x000000000000794d */
                                                               /* 0x000fea0003800000 */
        /*0050*/                   BRA 0x50;                   /* 0xfffffff000007947 */
                                                               /* 0x000fc0000383ffff */
\t\t..........



Fatbin elf code:
================
arch = sm_90
code version = [1,8]
host = linux
compile_size = 64bit

\tcode for sm_90
\t.target\tsm_90

\t\tFunction : _Z5otherPf
\t.headerflags\t@"EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)"
        /*0000*/                   LDC R1, c[0x0][0x28] ;      /* 0x00000a00ff017b82 */
                                                               /* 0x000ff00000000800 */
        /*0010*/                   EXIT ;                      /* 0x000000000000794d */
                                                               /* 0x000fea0003800000 */
        /*0020*/                   BRA 0x20;                   /* 0xfffffffc00fc7947 */
                                                               /* 0x000fc0000383ffff */
\t\t..........


"""

# The same code, moved to a higher address. Every instruction address, every
# branch target and every encoded word differs, and the count of the trailing
# `NOP` padding differs. None of that is a change of the generated code.
SHIFTED = """
Fatbin elf code:
================
arch = sm_75
code version = [1,8]
host = linux
compile_size = 64bit

\tcode for sm_75
\t.target\tsm_75

\t\tFunction : _Z5otherPf
\t.headerflags\t@"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
        /*2000*/                   MOV R1, c[0x0][0x28] ;      /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2010*/                   MOV R0, 0x3f800000 ;        /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2020*/                   EXIT ;                      /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2030*/                   BRA 0x2030;                 /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2040*/                   NOP;                        /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2050*/                   NOP;                        /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2060*/                   NOP;                        /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
\t\t..........


\t\tFunction : _Z6kernelIiEvPT_PKS0_i
\t.headerflags\t@"EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
        /*2000*/                   S2R R4, SR_CTAID.X ;        /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2010*/                   ISETP.GE.AND P0, PT, R4, c[0x0][0x170], PT ;
                                                               /* 0xdeadbeefdeadbeef */
        /*2020*/              @!P0 BRA 0x2050 ;                /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2030*/                   IMAD R7, R2, 0x6, RZ ;      /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2040*/                   EXIT ;                      /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2050*/                   BRA 0x2050;                 /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
\t\t..........



Fatbin elf code:
================
arch = sm_90
code version = [1,8]
host = linux
compile_size = 64bit

\tcode for sm_90
\t.target\tsm_90

\t\tFunction : _Z5otherPf
\t.headerflags\t@"EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)"
        /*2000*/                   LDC R1, c[0x0][0x28] ;      /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2010*/                   EXIT ;                      /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
        /*2020*/                   BRA 0x2020;                 /* 0xdeadbeefdeadbeef */
                                                               /* 0xdeadbeefdeadbeef */
\t\t..........


"""


def test_architectures_are_separated() -> None:
    assert list(normalized_text(BASELINE)) == ["sm_75", "sm_90"]


def test_kernels_are_assigned_to_their_architecture() -> None:
    text = normalized_text(BASELINE)
    assert text["sm_75"].count("Function : ") == 2
    assert text["sm_90"].count("Function : ") == 1


def test_container_metadata_is_not_a_kernel() -> None:
    """`code for sm_75` and `.target` must not be parsed as a function header."""
    names = {
        line.removeprefix("Function : ")
        for text in normalized_text(BASELINE).values()
        for line in text.splitlines()
        if line.startswith("Function : ")
    }
    assert names == {"_Z5otherPf", "_Z6kernelIiEvPT_PKS0_i"}


#: A dump as `sass_diff.sh` writes it: piped through `cu++filt`, thus the kernel
#: name is demangled and holds spaces, parentheses and angle brackets.
DEMANGLED = """
Fatbin elf code:
arch = sm_90
\t\tFunction : void kernel<int>(T1 *, const T1 *, int)
        /*0000*/                   MOV R1, c[0x0][0x28] ;   /* 0x00000a0000017a02 */
        /*0010*/                   IMAD R7, R2, 0x6, RZ ;   /* 0x0000000602077824 */
        /*0020*/                   EXIT ;                   /* 0x000000000000794d */
"""


def test_a_demangled_kernel_name_is_parsed() -> None:
    """`cu++filt` puts spaces in the name, which a `\\S+` pattern would reject.

    A rejected name drops every instruction of that kernel, both sides normalize
    to the same empty text, and every target compares as unchanged.
    """
    text = normalized_text(DEMANGLED)["sm_90"]
    assert "Function : void kernel<int>(T1 *, const T1 *, int)" in text
    assert "IMAD R7, R2, 0x6, RZ ;" in text


def test_a_dump_with_no_recognized_kernel_is_a_fault() -> None:
    """Silently dropping the instructions would read as "nothing changed"."""
    broken = DEMANGLED.replace("Function :", "aasdasdasd :")
    with pytest.raises(ValueError, match="no `Function :` line"):
        normalized_text(broken)


def test_addresses_encoded_words_and_padding_are_noise() -> None:
    assert normalized_text(BASELINE) == normalized_text(SHIFTED)


def test_shifted_code_is_not_reported_as_a_change(compared) -> None:
    """SHIFTED moves the code without changing it, so it must compare equal."""
    results = {entry.arch: entry for entry in compared(BASELINE, SHIFTED)}
    assert not results["sm_75"].changed
    assert results["sm_75"].diff is None
    # The raw text really does differ, so the test is not trivially true.
    assert BASELINE != SHIFTED


def test_branch_targets_become_deltas() -> None:
    text = normalized_text(BASELINE)["sm_75"]
    assert "@!P0 BRA <+0x30> ;" in text
    assert "BRA 0x50" not in text


def test_encoded_words_are_removed() -> None:
    assert "0x000fe40000000f00" not in normalized_text(BASELINE)["sm_75"]


def test_immediates_and_bank_offsets_are_kept() -> None:
    text = normalized_text(BASELINE)["sm_75"]
    assert "c[0x0][0x28]" in text
    assert "0x3f800000" in text
    assert "IMAD R7, R2, 0x6, RZ ;" in text


def test_kernel_order_is_noise() -> None:
    reordered = """
arch = sm_80
\t\tFunction : _Z1bv
        /*0000*/                   NOP ;
\t\tFunction : _Z1av
        /*0000*/                   EXIT ;
"""
    original = """
arch = sm_80
\t\tFunction : _Z1av
        /*0000*/                   EXIT ;
\t\tFunction : _Z1bv
        /*0000*/                   NOP ;
"""
    assert normalized_text(original) == normalized_text(reordered)


def test_opcode_change_is_detected() -> None:
    changed = BASELINE.replace("IMAD R7, R2, 0x6, RZ", "IADD3 R7, R2, 0x6, RZ")
    assert normalized_text(BASELINE) != normalized_text(changed)


def test_register_change_is_detected() -> None:
    changed = BASELINE.replace("IMAD R7, R2, 0x6, RZ", "IMAD R9, R2, 0x6, RZ")
    assert normalized_text(BASELINE) != normalized_text(changed)


def test_immediate_change_is_detected() -> None:
    """A changed immediate must not be normalized away with the addresses."""
    changed = BASELINE.replace("IMAD R7, R2, 0x6, RZ", "IMAD R7, R2, 0x8, RZ")
    assert normalized_text(BASELINE) != normalized_text(changed)


def test_control_flow_change_is_detected() -> None:
    changed = BASELINE.replace("@!P0 BRA 0x50 ;", "@!P0 BRA 0x40 ;")
    assert normalized_text(BASELINE) != normalized_text(changed)


def test_predicate_change_is_detected() -> None:
    changed = BASELINE.replace("@!P0 BRA 0x50 ;", "@P0 BRA 0x50 ;")
    assert normalized_text(BASELINE) != normalized_text(changed)


def test_added_architecture_is_reported(compared) -> None:
    # Keep only the first `Fatbin elf code:` block, which holds sm_75.
    one_arch = BASELINE.split("Fatbin elf code:")[1]
    results = {entry.arch: entry for entry in compared(one_arch, BASELINE)}
    assert set(results) == {"sm_75", "sm_90"}
    assert results["sm_90"].status is Status.ADDED
    assert results["sm_90"].changed
    assert results["sm_75"].status is Status.COMPARED


def test_removed_architecture_is_reported(compared) -> None:
    one_arch = BASELINE.split("Fatbin elf code:")[1]
    results = {entry.arch: entry for entry in compared(BASELINE, one_arch)}
    assert results["sm_90"].status is Status.REMOVED
    assert results["sm_90"].changed


def test_a_target_on_one_side_only_counts_as_changed() -> None:
    """An added or removed target has no per-architecture results of its own."""
    for status in (Status.ADDED, Status.REMOVED):
        assert TargetResult(target="cub.bench.new", status=status).changed


def test_the_work_dir_holds_the_compared_text(tmp_path: Path) -> None:
    """The written text must be what the comparison acted on, not the raw dump."""
    base_dir = tmp_path / "base"
    test_dir = tmp_path / "test"
    for directory, dump in ((base_dir, BASELINE), (test_dir, SHIFTED)):
        directory.mkdir()
        (directory / "cub.bench.demo.sass").write_text(dump)

    work = tmp_path / "result"
    report = compare(base_dir, test_dir, work)

    # One file per target and architecture, on both sides.
    written = sorted(path.name for path in (work / "base").iterdir())
    assert written == ["cub.bench.demo.sm_75.sass", "cub.bench.demo.sm_90.sass"]

    # BASELINE and SHIFTED differ only in noise, so the normalized text of the
    # two sides must be equal, and the report must agree.
    for name in written:
        assert (work / "base" / name).read_text() == (work / "test" / name).read_text()
    assert not any(target["changed"] for target in report["targets"])
    # Nothing changed, so no diff was written.
    assert list((work / "diff").iterdir()) == []


# ============================================================================
# The diff that the PR comment shows
# ============================================================================

# An opcode change on sm_75 only. `SHIFTED` moves the code without changing it,
# so it is the wrong sample here: the diff must have real content.
CHANGED = BASELINE.replace("MOV R0, 0x3f800000", "MOV R0, 0x40000000")


def test_an_unchanged_architecture_gets_no_diff(compared) -> None:
    for entry in compared(BASELINE, BASELINE):
        assert entry.diff is None


def test_a_changed_architecture_carries_the_changed_lines(compared) -> None:
    results = {entry.arch: entry for entry in compared(BASELINE, CHANGED)}
    diff = results["sm_75"].diff
    assert diff is not None
    text = "\n".join(diff.excerpt)
    assert "-MOV R0, 0x3f800000" in text
    assert "+MOV R0, 0x40000000" in text
    # One line on each side, and the file headers are not counted.
    assert diff.changed_lines == 2
    # The other architecture is untouched, so it has no diff of its own.
    assert results["sm_90"].diff is None


def test_the_diff_names_both_sides(compared) -> None:
    results = compared(BASELINE, CHANGED)
    diff = next(entry.diff for entry in results if entry.diff is not None)
    assert diff.excerpt[0] == "--- base/cub.bench.demo.sm_75"
    assert diff.excerpt[1] == "+++ test/cub.bench.demo.sm_75"


def test_a_long_diff_is_truncated_for_the_comment(compared) -> None:
    """The comment shows an excerpt; the whole diff goes to the artifacts."""

    # One kernel of many instructions, so the diff is longer than the excerpt.
    def dump(opcode: str) -> str:
        body = "\n".join(
            f"        /*{i * 16:04x}*/                   {opcode} R{i}, R1 ;"
            for i in range(_MAX_EXCERPT_LINES * 2)
        )
        return f"Fatbin elf code:\narch = sm_90\n\t\tFunction : _Z4longv\n{body}\n"

    results = compared(dump("MOV"), dump("IADD3"))
    diff = next(entry.diff for entry in results if entry.diff is not None)
    assert len(diff.excerpt) == _MAX_EXCERPT_LINES
    assert diff.total_lines > len(diff.excerpt)


def test_the_whole_diff_is_written_to_the_artifacts(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    test_dir = tmp_path / "test"
    for directory, dump in ((base_dir, BASELINE), (test_dir, CHANGED)):
        directory.mkdir()
        (directory / "cub.bench.demo.sass").write_text(dump)

    work = tmp_path / "result"
    report = compare(base_dir, test_dir, work)

    # Only the architecture that changed gets a file.
    written = sorted(path.name for path in (work / "diff").iterdir())
    assert written == ["cub.bench.demo.sm_75.diff"]

    text = (work / "diff" / written[0]).read_text()
    assert "-MOV R0, 0x3f800000" in text
    assert "+MOV R0, 0x40000000" in text

    # The report must name the file, so that the comment can point at it.
    archs = {entry["arch"]: entry for entry in report["targets"][0]["archs"]}
    assert archs["sm_75"]["diff"]["path"] == "cub.bench.demo.sm_75.diff"
    assert archs["sm_90"]["diff"] is None


def test_the_summary_says_whether_the_sass_changed(compared) -> None:
    """`main` turns this field into the exit code, so it must be exact."""
    assert not any(entry.changed for entry in compared(BASELINE, BASELINE))
    assert any(entry.changed for entry in compared(BASELINE, CHANGED))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
