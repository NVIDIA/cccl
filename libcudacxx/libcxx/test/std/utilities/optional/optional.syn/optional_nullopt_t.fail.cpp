//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// A program that necessitates the instantiation of template optional for
// (possibly cv-qualified) nullopt_t is ill-formed.

#include <optional>

#include "test_macros.h"

int main(int, char**)
{
    using std::optional;
    using std::nullopt_t;
    using std::nullopt;

    optional<nullopt_t> opt; // expected-note 1 {{requested here}}
    optional<const nullopt_t> opt1; // expected-note 1 {{requested here}}
    optional<nullopt_t &> opt2; // expected-note 1 {{requested here}}
#ifdef TEST_COMPILER_CLANG
    // expected-error-re@optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-object type is undefined behavior}}
#endif // TEST_COMPILER_CLANG
    optional<nullopt_t &&> opt3; // expected-note 1 {{requested here}}
    // expected-error@optional:* 4 {{instantiation of optional with nullopt_t is ill-formed}}
#ifdef TEST_COMPILER_CLANG
    // expected-error-re@optional:* {{{{(static_assert|static assertion)}} failed{{.*}}instantiation of optional with a non-object type is undefined behavior}}
#endif // TEST_COMPILER_CLANG

  return 0;
}
