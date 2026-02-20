//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/place_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)
/**
 * @brief Test green context partition and affinity: partition by green_context, push/pop affinity per subplace.
 */
void test_green_ctx_affinity()
{
  async_resources_handle handle;
  for (auto p : place_partition(exec_place::current_device(), handle, place_partition_scope::green_context))
  {
    handle.push_affinity(::std::make_shared<exec_place>(p));
    _CCCL_ASSERT(handle.current_affinity().size() == 1, "invalid value");
    handle.pop_affinity();
  }
}
#endif // _CCCL_CTK_AT_LEAST(12, 4)

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 4) vvv
  test_green_ctx_affinity();
  return 0;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^
}
