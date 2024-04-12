//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// template <class ElementType, size_t Extent>
//     span<byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : sizeof(ElementType) * Extent>
//     as_writable_bytes(span<ElementType, Extent> s) noexcept;

#include <cuda/std/span>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int iArr2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

struct A
{};

__host__ __device__ void f()
{
  cuda::std::as_writable_bytes(cuda::std::span<const int>()); // expected-error {{no matching function for call to
                                                              // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const long>()); // expected-error {{no matching function for call to
                                                               // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const double>()); // expected-error {{no matching function for call to
                                                                 // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const A>()); // expected-error {{no matching function for call to
                                                            // 'as_writable_bytes'}}

  cuda::std::as_writable_bytes(cuda::std::span<const int, 0>()); // expected-error {{no matching function for call to
                                                                 // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const long, 0>()); // expected-error {{no matching function for call to
                                                                  // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const double, 0>()); // expected-error {{no matching function for call to
                                                                    // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const A, 0>()); // expected-error {{no matching function for call to
                                                               // 'as_writable_bytes'}}

  cuda::std::as_writable_bytes(cuda::std::span<const int>(iArr2, 1)); // expected-error {{no matching function for call
                                                                      // to 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const int, 1>(iArr2 + 5, 1)); // expected-error {{no matching function
                                                                             // for call to 'as_writable_bytes'}}
}

int main(int, char**)
{
  return 0;
}
