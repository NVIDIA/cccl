// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/experimental/__cuco/fixed_capacity_set.cuh>

int main()
{
  // The default probing group needs a capacity divisible by four.
  using set_type = cuda::experimental::cuco::fixed_capacity_set<int, 7>;
  // expected-error "Capacity must be a valid open-addressing capacity"
  static_assert(sizeof(typename set_type::ref_type) > 0);
}
