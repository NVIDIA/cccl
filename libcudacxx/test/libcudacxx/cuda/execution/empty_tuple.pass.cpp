//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__utility/pod_tuple.h>

struct max_segment_size_42
{};

__host__ __device__ void test_tuple_empty()
{
  [[maybe_unused]] ::cuda::std::__tuple<max_segment_size_42> t{};
}

int main(int, char**)
{
  test_tuple_empty();
  return 0;
}
