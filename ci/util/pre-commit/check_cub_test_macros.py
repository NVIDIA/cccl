#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import sys

RAW_TEST_MACROS = (
    "C2H_TEST",
    "C2H_TEST_LIST",
    "C2H_TEST_WITH_FIXTURE",
    "C2H_TEST_LIST_WITH_FIXTURE",
    "TEST_CASE",
    "TEST_CASE_METHOD",
    "SCENARIO",
    "SCENARIO_METHOD",
    "TEMPLATE_TEST_CASE",
    "TEMPLATE_TEST_CASE_SIG",
    "TEMPLATE_TEST_CASE_METHOD",
    "TEMPLATE_TEST_CASE_METHOD_SIG",
    "TEMPLATE_PRODUCT_TEST_CASE",
    "TEMPLATE_PRODUCT_TEST_CASE_SIG",
    "TEMPLATE_PRODUCT_TEST_CASE_METHOD",
    "TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG",
    "TEMPLATE_LIST_TEST_CASE",
    "TEMPLATE_LIST_TEST_CASE_METHOD",
)

RAW_TEST_MACRO_RE = re.compile(
    r"^[ \t]*(?P<macro>"
    + "|".join(
        re.escape(macro) for macro in sorted(RAW_TEST_MACROS, key=len, reverse=True)
    )
    + r")[ \t]*\(",
    re.MULTILINE,
)


def remove_comments(source: str) -> str:
    """Blank C++ comments while preserving line and column positions."""
    result = list(source)
    index = 0

    while index < len(source):
        if source.startswith("//", index):
            end = source.find("\n", index)
            if end == -1:
                end = len(source)
            for comment_index in range(index, end):
                result[comment_index] = " "
            index = end
            continue

        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end == -1:
                end = len(source) - 2
            for comment_index in range(index, min(end + 2, len(source))):
                if source[comment_index] not in "\r\n":
                    result[comment_index] = " "
            index = end + 2
            continue

        if source[index] in {'"', "'"}:
            quote = source[index]
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            continue

        index += 1

    return "".join(result)


def self_test() -> bool:
    fixtures = [
        ('// C2H_TEST("x")', False),
        ('/*\nTEST_CASE("x")\n*/', False),
        ('const char* value = "TEST_CASE(";', False),
        ('CUB_TEST("x", "[y]", CUB_SMALL)', False),
        ('CUB_TEST_CASE("x", "[y]", CUB_LARGE)', False),
        ('CUB_TEST_LIST("x", "[y]", CUB_SMALL, types)', False),
    ]
    fixtures.extend((f'{macro}("x")', True) for macro in RAW_TEST_MACROS)

    for source, expected in fixtures:
        found = bool(RAW_TEST_MACRO_RE.search(remove_comments(source)))
        if found != expected:
            expected_result = "match" if expected else "no match"
            actual_result = "match" if found else "no match"
            print(
                "internal error: test-registration checker self-test failed for "
                f"{source!r}: expected {expected_result}, found {actual_result}. "
                "This is a problem with the checker, not the files being committed.",
                file=sys.stderr,
            )
            return False

    return True


def check_file(filename: str) -> bool:
    with open(filename, encoding="utf-8", errors="surrogateescape") as source_file:
        source = source_file.read()

    source_without_comments = remove_comments(source)
    found_error = False
    for match in RAW_TEST_MACRO_RE.finditer(source_without_comments):
        line = source.count("\n", 0, match.start()) + 1
        column = match.start("macro") - source.rfind("\n", 0, match.start("macro"))
        print(
            f"{filename}:{line}:{column}: {match.group('macro')} bypasses CUB "
            "memory classification; use CUB_TEST, CUB_TEST_CASE, or CUB_TEST_LIST."
        )
        found_error = True

    return found_error


def main() -> int:
    if not self_test():
        return 2

    found_error = False
    for filename in sys.argv[1:]:
        found_error = check_file(filename) or found_error
    return int(found_error)


if __name__ == "__main__":
    sys.exit(main())
