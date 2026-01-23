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
 * @brief Test green context affinity with configurable data place type
 *
 * @param use_green_ctx_data_place If true, use green context data place extension.
 *                                  If false, use regular device data place.
 */
void test_green_ctx_affinity(bool use_green_ctx_data_place)
{
  async_resources_handle handle;
  auto gc_helper = handle.get_gc_helper(0, 8); // 8 SMs per green context

  for (size_t i = 0; i < gc_helper->get_count(); i++)
  {
    auto p = exec_place::green_ctx(gc_helper->get_view(i), use_green_ctx_data_place);

    if (use_green_ctx_data_place)
    {
      _CCCL_ASSERT(p.affine_data_place().is_extension(), "expected a green context data place (extension)");
    }
    else
    {
      _CCCL_ASSERT(!p.affine_data_place().is_extension(), "expected a device data place (not extension)");
      _CCCL_ASSERT(device_ordinal(p.affine_data_place()) == 0, "expected device 0");
    }

    handle.push_affinity(::std::make_shared<exec_place>(p));
    _CCCL_ASSERT(handle.current_affinity().size() == 1, "invalid value");

    if (use_green_ctx_data_place)
    {
      _CCCL_ASSERT(handle.current_affinity()[0]->affine_data_place().is_extension(),
                   "expected a green context data place (extension)");
    }
    else
    {
      _CCCL_ASSERT(!handle.current_affinity()[0]->affine_data_place().is_extension(),
                   "expected a device data place (not extension)");
    }

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
  // Test both data place modes
  test_green_ctx_affinity(false); // Device data place (default)
  test_green_ctx_affinity(true); // Green context data place extension
  return 0;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^
}
