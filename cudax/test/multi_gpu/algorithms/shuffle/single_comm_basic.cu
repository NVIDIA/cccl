// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shuffle_common.cuh"

MULTI_GPU_TEST("HashGraph shuffle threaded single-communicator conservation", shuffle_test::types)
{
  using T = typename c2h::get<0, TestType>;
  shuffle_test::cases<T, true>(this->communicators());
}
