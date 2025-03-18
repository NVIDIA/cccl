//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cccl/c/radix_sort.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
struct radix_sort_runtime_tuning_policy
{
  int block_size;
  int items_per_thread;
  int items_per_tile;

  radix_sort_runtime_tuning_policy Histogram() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy ExclusiveSum() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy Onesweep() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy Scan() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy Downsweep() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy AltDownsweep() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy Upsweep() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy AltUpsweep() const
  {
    return *this;
  }

  radix_sort_runtime_tuning_policy SingleTile() const
  {
    return *this;
  }
};

struct radix_sort_tuning_t
{
  int cc;
  int block_size;
  int items_per_thread;
};

radix_sort_runtime_tuning_policy get_policy(int cc, int key_size)
{
  radix_sort_tuning_t chain[] = {
    {60, 256, nominal_4b_items_to_items(17, key_size)}, {35, 256, nominal_4b_items_to_items(11, key_size)}};
  auto [_, block_size, items_per_thread] = find_tuning(cc, chain);
  // TODO: we hardcode this value in order to make sure that the merge_sort test does not fail due to the memory op
  // assertions. This currently happens when we pass in items and keys of type uint8_t or int16_t, and for the custom
  // types test as well. This will be fixed after https://github.com/NVIDIA/cccl/issues/3570 is resolved.
  items_per_thread = 2;

  return {block_size, items_per_thread, block_size * items_per_thread};
}

} // namespace radix_sort

CCCL_C_API CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_type_info key_t,
  cccl_type_info value_t,
  cccl_op_t decomposer,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;
  try
  {
    const char* name = "test";

    const int cc      = cc_major * 10 + cc_minor;
    const auto policy = radix_sort::get_policy(cc, output_keys_it.value_type.size);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CCCL_C_API CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_double_buffer_t d_keys,
  cccl_double_buffer_t d_values,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  CUstream stream);

CCCL_C_API CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* bld_ptr);
