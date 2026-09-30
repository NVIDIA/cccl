//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/mdspan>
#include <cuda/std/cassert>

bool test_mdspan_to_dlpack_wrapper_get_lvalue()
{
  auto tensor = cuda::dlpack_tensor<3>{}.get();
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_wrapper_get_lvalue());))
  return 0;
}
