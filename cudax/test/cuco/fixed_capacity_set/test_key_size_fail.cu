// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/experimental/__cuco/fixed_capacity_set.cuh>

#include <vector_types.h>

int main()
{
  using set_type = cuda::experimental::cuco::fixed_capacity_set<char3>;
  // expected-error {{"key_type size must be a power of two"}}
  static_assert(sizeof(typename set_type::ref_type) > 0);
}
