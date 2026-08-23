#!/usr/bin/env python3
"""Compare the SASS of two builds and report what changed.

`sass_diff.sh` builds the CUB benchmarks from two git refs and dumps the
disassembly with `cuobjdump -sass`. This script normalizes those dumps and
compares them. `render_report.py` makes the PR comment from the result.

The raw disassembly contains data that changes between builds when the machine
code does not change. The normalizer removes it:

  * instruction addresses (`/*0a30*/`),
  * the encoded instruction words (`/* 0x000fe200078e0203 */`),
  * the order in which the kernels are emitted,
  * the `NOP` padding at the end of each kernel.

Branch targets become a signed delta from the branch address, thus code that
only moved is equal. Opcodes, modifiers, predicates, registers, immediates,
constant-bank offsets and the control flow are all compared.

`cuobjdump -sass` prints every architecture into one stream. Each architecture
is split out and compared on its own. A fatbin names it in an `arch =` line, and
a raw cubin in a `code for` line.

This script gives status 1 when the SASS changed. This is for use from a shell.
CI does not read that status. CI reads `changed` from report.json, because
`sass_diff.sh` runs with `set -e` and cannot tell status 1 from a failed build.
"""

import argparse
import difflib
import itertools
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# ============================================================================
# Normalization
# ============================================================================

_FATBIN_RE = re.compile(r"^\s*Fatbin elf code\s*:")

# `arch = sm_90`
_ARCH_RE = re.compile(r"^\s*arch\s*=\s*(?P<arch>\S+)\s*$")

# `	code for sm_90`. A raw cubin has no `Fatbin elf code:` container, thus no
# `arch =` line, and starts here. In a fatbin this line follows the `arch =` line
# of the same architecture, so it reopens the block that is already open.
_CODE_FOR_RE = re.compile(r"^\s*code for\s+(?P<arch>\S+)\s*$")

# `                Function : void kernel<int>(T1 *, const T1 *, int)`
# The dump goes through `cu++filt`, thus the name is demangled and holds spaces
# and parentheses. The whole rest of the line is the name.
_FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(?P<name>\S.*?)\s*$")

# `        /*0000*/                   MOV R1, c[0x0][0x28] ;   /* 0x00000a0000017a02 */`
_INSTRUCTION_RE = re.compile(
    r"""^\s*
        /\*(?P<address>[0-9a-fA-F]+)\*/    # the address comment is mandatory
        \s*
        (?P<text>\S.*?)                    # the instruction itself
        \s*
        (?:/\*\s*0x[0-9a-fA-F]+\s*\*/)?    # the encoded word, when it is present
        \s*$""",
    re.VERBOSE,
)

# Opcodes with a code address as an operand. Only these get their hex operand
# rewritten. Every other hex value is an immediate or a constant-bank offset.
_BRANCH_OPCODES = (
    "BRA",
    "BRX",
    "BRXU",
    "JMP",
    "JMX",
    "JMXU",
    "CALL",
    "CAL",
    "BSSY",
    "SSY",
    "PBK",
    "BRK",
    "PCNT",
    "CONT",
)

_BRANCH_RE = re.compile(
    r"^(?P<head>(?:@!?\w+\s+)?(?:" + "|".join(_BRANCH_OPCODES) + r")[\w.]*\s+"
    r"(?:\w+,\s*)?)"
    r"(?P<target>0x[0-9a-fA-F]+)"
    r"(?P<tail>.*)$"
)

_NOP_RE = re.compile(r"^NOP\s*;?\s*$")

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Kernel:
    """One disassembled kernel."""

    name: str
    instructions: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join([f"Function : {self.name}", *self.instructions])


@dataclass
class Listing:
    """Every kernel that `cuobjdump -sass` printed for one architecture."""

    arch: str = ""
    kernels: list[Kernel] = field(default_factory=list)

    def text(self) -> str:
        # Sorted, thus a change of the emission order alone is not a difference.
        return "\n".join(
            [
                f"arch = {self.arch}",
                *(
                    kernel.text()
                    for kernel in sorted(self.kernels, key=lambda k: k.name)
                ),
                "",
            ]
        )


def _kernels(lines: list[str]) -> list[Kernel]:
    """Parse and normalize the body of one architecture block.

    Raises if the block holds instructions but no kernel was recognized. That
    means `_FUNCTION_RE` no longer matches what `cuobjdump` prints, and every
    instruction would be dropped. Both sides would then normalize to the same
    empty text and every target would compare as unchanged, which reads as
    "nothing changed" instead of as a fault.
    """
    kernels: list[Kernel] = []
    kernel: Kernel | None = None
    saw_instruction = False

    for line in filter(None, map(str.strip, lines)):
        # Over 99% of the lines are instructions or encoded words, and only those
        # start with `/*`. This test keeps the other patterns off the hot path.
        if not line.startswith("/*"):
            if function_match := _FUNCTION_RE.match(line):
                kernel = Kernel(name=function_match.group("name"))
                kernels.append(kernel)
            continue

        if not (instruction_match := _INSTRUCTION_RE.match(line)):
            continue

        saw_instruction = True
        if kernel is None:
            continue

        address = int(instruction_match.group("address"), 16)
        text = _WHITESPACE_RE.sub(" ", instruction_match.group("text")).strip()

        if branch_match := _BRANCH_RE.match(text):
            delta = int(branch_match.group("target"), 16) - address
            sign = "+" if delta >= 0 else "-"
            text = (
                f"{branch_match.group('head')}<{sign}{abs(delta):#x}>"
                f"{branch_match.group('tail')}"
            )

        kernel.instructions.append(text)

    if saw_instruction and not kernels:
        raise AssertionError(
            "The dump holds instructions but no `Function :` line was "
            "recognized. `_FUNCTION_RE` does not match what cuobjdump prints."
        )

    # ptxas pads each kernel with `NOP` up to its alignment. The number of them
    # is a result of the kernel size, not of the generated code.
    for entry in kernels:
        end = len(entry.instructions)
        if end == 0:
            raise AssertionError(
                f"Kernel {entry} has no instructions. This should never happen."
            )
        while end > 0 and _NOP_RE.match(entry.instructions[end - 1]):
            end -= 1
        del entry.instructions[end:]

    return kernels


def normalized_text(raw: str) -> dict[str, str]:
    """Return the normalized text of each architecture in a dump.

    The same architecture can occur in more than one `Fatbin elf code:` block,
    because the binary can contain more than one linked object. The blocks are
    merged by architecture name.

    Raises on an instruction that belongs to no architecture. Dropping it would
    let both sides normalize to the same result and compare as unchanged.
    """
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None

    for line in filter(None, map(str.strip, raw.splitlines())):
        if _FATBIN_RE.match(line):
            current = None
            continue

        if arch_match := _ARCH_RE.match(line):
            current = blocks.setdefault(arch_match.group("arch"), [])
            continue

        if code_for_match := _CODE_FOR_RE.match(line):
            current = blocks.setdefault(code_for_match.group("arch"), [])
            continue

        if current is None:
            if line.startswith("/*"):
                raise AssertionError(
                    f"Current arch for {line} is none. We have failed "
                    "to detect the architecture for this kernel."
                )
            continue

        current.append(line)

    return {
        arch: Listing(arch=arch, kernels=_kernels(lines)).text()
        for arch, lines in blocks.items()
    }


# ============================================================================
# Comparison
# ============================================================================

# The diff of a complete kernel can be thousands of lines, and a GitHub comment
# holds 65536 characters. Thus the comment shows only this many lines and links
# to the complete diff in the artifacts.
_MAX_EXCERPT_LINES = 40


class Status(StrEnum):
    """Whether an item was compared, or exists on only one side."""

    COMPARED = "compared"
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class Diff:
    """The unified diff of one target and architecture."""

    #: The first `_MAX_EXCERPT_LINES` lines, for the PR comment.
    excerpt: list[str]
    changed_lines: int
    total_lines: int
    #: The file with the complete diff.
    path: str


@dataclass
class ArchResult:
    """The result for one architecture of one target."""

    arch: str
    status: Status = Status.COMPARED
    diff: Diff | None = None

    @property
    def changed(self) -> bool:
        # An architecture on one side only is a change, and has no diff.
        return self.status is not Status.COMPARED or self.diff is not None


@dataclass
class TargetResult:
    """The result for one benchmark target."""

    target: str
    status: Status = Status.COMPARED
    archs: list[ArchResult] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        # A target on one side only is a change, and has no per-arch results.
        if self.status is not Status.COMPARED:
            return True
        return any(arch.changed for arch in self.archs)


def compare_target(
    base_path: Path, test_path: Path, target: str, work_dir: Path
) -> TargetResult:
    """Compare one target, architecture by architecture.

    The normalized text of both sides goes under `base/` and `test/` of
    `work_dir`, and the complete diff of each changed architecture under `diff/`.
    """
    base = normalized_text(base_path.read_text(errors="replace"))
    test = normalized_text(test_path.read_text(errors="replace"))

    diff_dir = work_dir / "diff"

    for side, texts in (("base", base), ("test", test)):
        side_dir = work_dir / side
        for arch, text in texts.items():
            (side_dir / f"{target}.{arch}.sass").write_text(text)

    results: list[ArchResult] = []
    for arch in base.keys() | test.keys():
        if arch not in base or arch not in test:
            status = Status.ADDED if arch not in base else Status.REMOVED
            results.append(ArchResult(arch=arch, status=status))
            continue

        if base[arch] == test[arch]:
            results.append(ArchResult(arch=arch))
            continue

        lines = list(
            difflib.unified_diff(
                base[arch].splitlines(),
                test[arch].splitlines(),
                fromfile=f"base/{target}.{arch}",
                tofile=f"test/{target}.{arch}",
                n=3,
                lineterm="",
            )
        )
        diff = Diff(
            excerpt=lines[:_MAX_EXCERPT_LINES],
            # Skip the `---` and `+++` headers, which are not changed content.
            changed_lines=sum(
                1 for line in itertools.islice(lines, 2, None) if line[0] in "-+"
            ),
            total_lines=len(lines),
            path=f"{target}.{arch}.diff",
        )
        lines.append("")
        (diff_dir / diff.path).write_text("\n".join(lines))

        results.append(ArchResult(arch=arch, diff=diff))
    return TargetResult(target=target, archs=results)


def compare(
    base_dir: Path, test_dir: Path, work_dir: Path, verbose: bool = False
) -> dict[str, Any]:
    """Compare every `<target>.sass` file that both directories hold.

    The normalized text of both sides goes under `base/` and `test/` of
    `work_dir`, and the complete diffs under `diff/`. Those files show what the
    comparison used, and the PR comment links to the diffs.
    """

    def dumps(directory: Path) -> dict[str, Path]:
        return {path.stem: path for path in directory.glob("*.sass")}

    base_dumps = dumps(base_dir)
    test_dumps = dumps(test_dir)

    for name in ("base", "test", "diff"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)

    results = [
        TargetResult(
            target=target,
            status=Status.ADDED if target not in base_dumps else Status.REMOVED,
        )
        for target in (base_dumps.keys() ^ test_dumps.keys())
    ]

    # Each target is independent and costs seconds of pure CPU, so the work is
    # spread over the cores of the runner. The workers read the dumps themselves,
    # because the dumps are tens of megabytes each.
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(
                compare_target, base_dumps[target], test_dumps[target], target, work_dir
            )
            for target in (base_dumps.keys() & test_dumps.keys())
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            entry = future.result()
            results.append(entry)
            if verbose:
                state = "changed" if entry.changed else "same"
                print(f"[{idx}/{len(futures)}] {entry.target}: {state}")

    if verbose:
        changed = sum(1 for entry in results if entry.changed)
        print(f"{changed} of {len(results)} target(s) changed")

    return {
        "targets": [
            {
                "target": entry.target,
                "status": entry.status,
                # `changed` is a property, and `asdict` emits fields only.
                "changed": entry.changed,
                "archs": [
                    {**asdict(arch), "changed": arch.changed} for arch in entry.archs
                ],
            }
            for entry in results
        ],
    }


# ============================================================================
# Command line
# ============================================================================

#: `render_report.py` reads this file from the output directory.
REPORT_NAME = "report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every `<target>.sass` file that both directories hold, per "
            "target and per architecture. Ignores instruction addresses, "
            "encoded instruction words, absolute branch targets, kernel "
            "emission order and trailing NOP padding, because those change "
            f"between builds when the code does not. Writes {REPORT_NAME}, the "
            "normalized text of both sides under base/ and test/, and the diff "
            "of each changed architecture under diff/. Exits 0 when the SASS is "
            "unchanged, 1 when it changed, and 2 on any fault."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Directory of baseline dumps.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory of the dumps to test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the report and the diffs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each target as it is compared, and a count at the end.",
    )
    args = parser.parse_args()

    for directory in (args.base_dir, args.test_dir):
        if not directory.is_dir():
            raise ValueError(f"not a directory: {directory}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = compare(args.base_dir, args.test_dir, args.output_dir, args.verbose)
    (args.output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2))

    changed = any(target["changed"] for target in report["targets"])
    return 1 if changed else 0


if __name__ == "__main__":
    # Only a return of 1 from `main` means "the SASS changed". Python exits 1 on
    # an unhandled exception too, thus every fault is forced to 2 here.
    try:
        ret = main()
    except SystemExit as e:
        if e.code == 1:
            raise SystemExit(2)
        raise  # reraise 0, or anything other number as-is
    except BaseException as e:
        raise SystemExit(2) from e

    sys.exit(ret)
