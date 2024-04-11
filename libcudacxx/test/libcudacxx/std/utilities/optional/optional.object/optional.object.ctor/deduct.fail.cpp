//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <cuda/std/optional>

// template<class T>
//   optional(T) -> optional<T>;

#include <cuda/std/cassert>
#include <cuda/std/optional>

struct A
{};

int main(int, char**)
{
  //  Test the explicit deduction guides

  //  Test the implicit deduction guides
  {
    //  optional()
    cuda::std::optional opt; // expected-error {{no viable constructor or deduction guide for deduction of template
                             // arguments of 'optional'}}
  }

  {
    //  optional(nullopt_t)
    cuda::std::optional opt(cuda::std::nullopt); // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                                 // failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
  }

  return 0;
}
