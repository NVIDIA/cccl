//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/mdspan>

#include "test_macros.h"

using ext_t = cuda::std::extents<int, 4>;

bool host_accessor_test()
{
  int array[] = {1, 2, 3, 4};
  int* h_ptr;
  assert(cudaMallocHost(&h_ptr, 4) == cudaSuccess);
  [[maybe_unused]] cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  [[maybe_unused]] cuda::host_mdspan<int, ext_t> h_md2{h_ptr, ext_t{}};
  using empty_ext_t = cuda::std::dextents<int, 2>;
  cuda::host_mdspan<int, empty_ext_t> empty_md{nullptr, empty_ext_t{0, 3}};
  assert(empty_md.size() == 0);
  assert(empty_md.data_handle() == nullptr);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(host_accessor_test());))
  return 0;
}
