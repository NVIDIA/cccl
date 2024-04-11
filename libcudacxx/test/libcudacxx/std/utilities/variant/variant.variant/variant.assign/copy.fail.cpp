//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// LWG issue 3024

#include <cuda/std/type_traits>
#include <cuda/std/variant>

struct NotCopyConstructible
{
  NotCopyConstructible()                            = default;
  NotCopyConstructible(NotCopyConstructible const&) = delete;
};

int main(int, char**)
{
  static_assert(!cuda::std::is_copy_constructible_v<NotCopyConstructible>);

  cuda::std::variant<NotCopyConstructible> v;
  cuda::std::variant<NotCopyConstructible> v1;
  cuda::std::variant<NotCopyConstructible> v2(v); // expected-error {{call to implicitly-deleted copy constructor of
                                                  // 'cuda::std::variant<NotCopyConstructible>'}}
  v1 = v; // expected-error-re {{object of type 'std:{{.*}}:variant<NotCopyConstructible>' cannot be assigned because
          // its copy assignment operator is implicitly deleted}}
}
