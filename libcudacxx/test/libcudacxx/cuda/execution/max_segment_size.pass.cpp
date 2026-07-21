//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/guarantee.h>
#include <cuda/__execution/max_segment_size.h>
#include <cuda/std/__execution/env.h>

__host__ __device__ void test(size_t dynamic_val)
{
  namespace exec = cuda::execution;
  static_assert(cuda::std::is_base_of_v<exec::__guarantee, exec::max_segment_size<42>>);

  static_assert(cuda::std::is_same_v<decltype(exec::__get_max_segment_size(exec::max_segment_size<42>{})),
                                     exec::max_segment_size<42>>);

  static_assert(cuda::std::is_same_v<decltype(exec::__get_max_segment_size(exec::max_segment_size<>{dynamic_val})),
                                     exec::max_segment_size<exec::dynamic_max_segment_size>>);

  static_assert(
    cuda::std::is_same_v<
      decltype(exec::__get_max_segment_size(exec::__get_guarantees(exec::guarantee(exec::max_segment_size<42>{})))),
      exec::max_segment_size<42>>);

  static_assert(cuda::std::is_same_v<decltype(exec::__get_max_segment_size(
                                       exec::__get_guarantees(exec::guarantee(exec::max_segment_size<>{dynamic_val})))),
                                     exec::max_segment_size<exec::dynamic_max_segment_size>>);

  constexpr exec::max_segment_size<42> static_size{};
  static_assert(static_cast<int>(static_size) == 42);

  constexpr exec::max_segment_size<42> static_size_with_runtime_value{100};
  // ignore runtime value in case of static size
  static_assert(static_cast<int>(static_size_with_runtime_value) == 42);

  exec::max_segment_size dynamic_size{dynamic_val};
  (void) (static_cast<size_t>(dynamic_size) == dynamic_val);

  auto g_env = exec::guarantee(dynamic_size);
  (void) g_env;

  auto dynamic_size_extracted =
    ::cuda::std::execution::__query_or(g_env, exec::__get_max_segment_size_t{}, exec::max_segment_size<0>{});
  (void) (static_cast<size_t>(dynamic_size_extracted) == dynamic_val);
}

int main(int argc, char**)
{
  test(argc);

  return 0;
}
