#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

_BASE_CMAKELISTS_TEMPLATE: Final = """\
list(APPEND CMAKE_MESSAGE_CONTEXT "{underscore_check_name}")

cccl_create_clang_tidy_plugin({underscore_check_name}.cpp)

add_subdirectory(test)
"""

_TEST_CMAKELISTS_TEMPLATE: Final = """\
list(APPEND CMAKE_MESSAGE_CONTEXT "test")

file(GLOB test_srcs CONFIGURE_DEPENDS "*.cpp")

foreach (src IN LISTS test_srcs)
  cccl_create_clang_tidy_test({underscore_check_name} "${{src}}")
endforeach()
"""

_CHECK_TEMPLATE: Final = """\
// SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include <clang-tidy/ClangTidyCheck.h>
#include <clang-tidy/ClangTidyModule.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>

namespace clang::tidy::cccl
{{
namespace
{{
using namespace ast_matchers; // NOLINT(google-build-using-namespace)

class {title_check_name}Check final : public ClangTidyCheck
{{
public:
  [[nodiscard]] std::optional<TraversalKind> getCheckTraversalKind() const override
  {{
    return TK_IgnoreUnlessSpelledInSource;
  }}

  void registerMatchers(MatchFinder* finder) override
  {{
    //
  }}

  void check(const MatchFinder::MatchResult& result) override
  {{
    llvm::reportFatalUsageError("cccl-{dash_check_name}: unhandled node kind, this is a bug in the check");
  }}
}};

class {title_check_name}CheckModule final : public ClangTidyModule
{{
public:
  static constexpr StringRef CHECK_NAME{{"cccl-{dash_check_name}"}};

  void addCheckFactories(ClangTidyCheckFactories& check_factories) override
  {{
    check_factories.registerCheck<{title_check_name}Check>(CHECK_NAME);
  }}
}};

// NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
ClangTidyModuleRegistry::Add<{title_check_name}CheckModule> _{{
  {title_check_name}CheckModule::CHECK_NAME,
  /* you must provide a useful description here! */}};
}} // namespace
}} // namespace clang::tidy::cccl
"""


@dataclass
class Options:
    dash_check_name: str
    underscore_check_name: str
    title_check_name: str

    def __init__(self, base_name: str) -> None:
        if any(map(str.isspace, base_name)):
            raise ValueError(
                "Invalid base check name, should be separated by '-', not ' '"
            )

        self.dash_check_name = base_name
        self.underscore_check_name = base_name.replace("-", "_")
        self.title_check_name = base_name.title().replace("-", "")


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Create a new clang-tidy check by creating/adding all the required boiler-plate."
    )

    parser.add_argument(
        "base_name",
        help=(
            "Base name of clang-tidy check without the cccl- prefix and delimited by '-'. "
            "For example, for a check cccl-foo-bar-baz, this must be 'foo-bar-baz'"
        ),
    )

    args = parser.parse_args()
    return Options(args.base_name)


def main() -> None:
    opts = parse_args()
    cur_dir = Path(__file__).parent

    check_dir = cur_dir / opts.underscore_check_name
    check_dir.mkdir(exist_ok=True)

    (check_dir / "CMakeLists.txt").write_text(
        _BASE_CMAKELISTS_TEMPLATE.format(
            underscore_check_name=opts.underscore_check_name
        )
    )
    (check_dir / f"{opts.underscore_check_name}.cpp").write_text(
        _CHECK_TEMPLATE.format(
            year=datetime.now().year,
            dash_check_name=opts.dash_check_name,
            underscore_check_name=opts.underscore_check_name,
            title_check_name=opts.title_check_name,
        )
    )

    test_dir = check_dir / "test"
    test_dir.mkdir(exist_ok=True)

    (test_dir / "CMakeLists.txt").write_text(
        _TEST_CMAKELISTS_TEMPLATE.format(
            underscore_check_name=opts.underscore_check_name
        )
    )

    # Do this after we have done all the rest correctly
    with (cur_dir / "CMakeLists.txt").open("a") as fd:
        fd.write(f"add_subdirectory({opts.underscore_check_name})\n")


if __name__ == "__main__":
    main()
