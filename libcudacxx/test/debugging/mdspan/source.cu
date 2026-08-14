// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allow constructing device_mdspan below over ordinary host memory.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <cuda_runtime_api.h>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using inner_mdspan = cuda::std::mdspan<int, cuda::std::extents<int, 2>>;

[[gnu::noinline]] void inspect_rank1_dynamic(const cuda::std::mdspan<int, cuda::std::dextents<int, 1>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_rank2_static(const cuda::std::mdspan<int, cuda::std::extents<int, 3, 4>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_rank2_mixed(const cuda::std::mdspan<int, cuda::std::extents<int, 3, cuda::std::dynamic_extent>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_rank2_dynamic(const cuda::std::mdspan<int, cuda::std::dextents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_rank0(const cuda::std::mdspan<int, cuda::std::extents<int>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::std::mdspan<int, cuda::std::extents<int, 0, 3>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_layout_right(const cuda::std::mdspan<int, cuda::std::extents<int, 3, 3>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_layout_left(const cuda::std::mdspan<int, cuda::std::extents<int, 3, 3>, cuda::std::layout_left>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_layout_stride(const cuda::std::mdspan<int, cuda::std::extents<int, 3>, cuda::std::layout_stride>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_rank0_layout_stride(const cuda::std::mdspan<int, cuda::std::extents<int>, cuda::std::layout_stride>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_const_element(const cuda::std::mdspan<const int, cuda::std::dextents<int, 1>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_pointer_element(const cuda::std::mdspan<int*, cuda::std::dextents<int, 1>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::mdspan<inner_mdspan, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_custom_accessor(
  const cuda::std::
    mdspan<int, cuda::std::dextents<int, 1>, cuda::std::layout_right, cuda::std::aligned_accessor<int, 16>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_host_mdspan(const cuda::host_mdspan<int, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// No elements expected: device memory isn't host-dereferenceable.
[[gnu::noinline]] void inspect_device_mdspan(const cuda::device_mdspan<int, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// Same as above, but backed by a real cudaMalloc'd allocation (not host memory, and not the
// invalidity-detection bypass)
[[gnu::noinline]] void
inspect_device_mdspan_real_memory(const cuda::device_mdspan<int, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// A plain cuda::std::mdspan (default_accessor, no device_accessor marker) whose data pointer
// happens to be a real cudaMalloc'd allocation.
[[gnu::noinline]] void inspect_mdspan_device_backed(const cuda::std::mdspan<int, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// A null data pointer, so the printer's element-copy cudaMemcpy call fails cleanly
// (cudaErrorInvalidValue) without touching real memory.
[[gnu::noinline]] void inspect_mdspan_copy_failure(const cuda::std::mdspan<int, cuda::std::extents<int, 1>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

// Unified memory is host-readable: full element display, like host_mdspan.
[[gnu::noinline]] void inspect_managed_mdspan(const cuda::managed_mdspan<int, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_restrict_mdspan(const cuda::restrict_mdspan<int, cuda::std::extents<int, 2>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void
inspect_array(const cuda::std::array<cuda::std::mdspan<int, cuda::std::extents<int, 2>>, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::mdspan<int, cuda::std::dextents<int, 1>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::mdspan<int, cuda::std::dextents<int, 1>>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

int main()
{
  // Build the CUDA context before the first breakpoint. The printer copies elements
  // with an inferior cudaMemcpy call; if that call is the first CUDA call in the
  // process, a breakpoint hit during context setup interrupts it, LLDB unwinds it,
  // and every later copy fails with cudaErrorContextIsDestroyed.
  if (cudaFree(nullptr) != cudaSuccess)
  {
    return 1;
  }

  int rank1_dynamic_data[4] = {10, -3, 7, 42};
  const cuda::std::mdspan<int, cuda::std::dextents<int, 1>> rank1_dynamic(rank1_dynamic_data, 4);
  inspect_rank1_dynamic(rank1_dynamic);

  int rank2_static_data[12];
  cuda::std::mdspan<int, cuda::std::extents<int, 3, 4>> rank2_static(rank2_static_data);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 4; ++col)
    {
      rank2_static(row, col) = row * 10 + col;
    }
  }
  inspect_rank2_static(rank2_static);

  int rank2_mixed_data[12];
  cuda::std::mdspan<int, cuda::std::extents<int, 3, cuda::std::dynamic_extent>> rank2_mixed(rank2_mixed_data, 4);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 4; ++col)
    {
      rank2_mixed(row, col) = 100 + row * 10 + col;
    }
  }
  inspect_rank2_mixed(rank2_mixed);

  int rank2_dynamic_data[6];
  cuda::std::mdspan<int, cuda::std::dextents<int, 2>> rank2_dynamic(rank2_dynamic_data, 2, 3);
  for (int row = 0; row < 2; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      rank2_dynamic(row, col) = 1000 + row * 10 + col;
    }
  }
  inspect_rank2_dynamic(rank2_dynamic);

  int rank0_data[1] = {77};
  const cuda::std::mdspan<int, cuda::std::extents<int>> rank0(rank0_data);
  inspect_rank0(rank0);

  int empty_data[1] = {0};
  const cuda::std::mdspan<int, cuda::std::extents<int, 0, 3>> empty_span(empty_data);
  inspect_empty(empty_span);

  int layout_shared_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  const cuda::std::mdspan<int, cuda::std::extents<int, 3, 3>> layout_right_span(layout_shared_data);
  inspect_layout_right(layout_right_span);

  const cuda::std::mdspan<int, cuda::std::extents<int, 3, 3>, cuda::std::layout_left> layout_left_span(
    layout_shared_data);
  inspect_layout_left(layout_left_span);

  const cuda::std::extents<int, 3> layout_stride_extents;
  const cuda::std::layout_stride::mapping<cuda::std::extents<int, 3>> layout_stride_mapping(
    layout_stride_extents, cuda::std::array<int, 1>{4});
  const cuda::std::mdspan<int, cuda::std::extents<int, 3>, cuda::std::layout_stride> layout_stride_span(
    layout_shared_data, layout_stride_mapping);
  inspect_layout_stride(layout_stride_span);

  int rank0_stride_data[1] = {55};
  const cuda::std::extents<int> rank0_stride_extents;
  const cuda::std::layout_stride::mapping<cuda::std::extents<int>> rank0_stride_mapping(
    rank0_stride_extents, cuda::std::array<int, 0>{});
  const cuda::std::mdspan<int, cuda::std::extents<int>, cuda::std::layout_stride> rank0_stride_span(
    rank0_stride_data, rank0_stride_mapping);
  inspect_rank0_layout_stride(rank0_stride_span);

  int const_element_data[3] = {5, 6, 7};
  const cuda::std::mdspan<const int, cuda::std::dextents<int, 1>> const_element(const_element_data, 3);
  inspect_const_element(const_element);

  int pointee0                 = 101;
  int pointee1                 = 102;
  int pointee2                 = 103;
  int* pointer_element_data[3] = {&pointee0, &pointee1, &pointee2};
  const cuda::std::mdspan<int*, cuda::std::dextents<int, 1>> pointer_element(pointer_element_data, 3);
  inspect_pointer_element(pointer_element);

  int nested_data0[2]         = {1, 2};
  int nested_data1[2]         = {3, 4};
  inner_mdspan nested_data[2] = {inner_mdspan(nested_data0), inner_mdspan(nested_data1)};
  const cuda::std::mdspan<inner_mdspan, cuda::std::extents<int, 2>> nested(nested_data);
  inspect_nested(nested);

  alignas(16) int custom_accessor_data[4] = {-7, 8, -9, 10};
  const cuda::std::mdspan<int, cuda::std::dextents<int, 1>, cuda::std::layout_right, cuda::std::aligned_accessor<int, 16>>
    custom_accessor(custom_accessor_data, 4);
  inspect_custom_accessor(custom_accessor);

  int host_mdspan_data[2] = {41, 42};
  const cuda::host_mdspan<int, cuda::std::extents<int, 2>> host_mdspan_span(host_mdspan_data);
  inspect_host_mdspan(host_mdspan_span);

  int device_mdspan_data[2] = {51, 52};
  const cuda::device_mdspan<int, cuda::std::extents<int, 2>> device_mdspan_span(device_mdspan_data);
  inspect_device_mdspan(device_mdspan_span);

  int* device_mdspan_real_data = nullptr;
  if (cudaMalloc(&device_mdspan_real_data, 2 * sizeof(int)) != cudaSuccess)
  {
    return 1;
  }
  const int device_mdspan_real_host_data[2] = {91, 92};
  if (cudaMemcpy(device_mdspan_real_data,
                 device_mdspan_real_host_data,
                 sizeof(device_mdspan_real_host_data),
                 cudaMemcpyHostToDevice)
      != cudaSuccess)
  {
    return 1;
  }
  const cuda::device_mdspan<int, cuda::std::extents<int, 2>> device_mdspan_real_span(device_mdspan_real_data);
  inspect_device_mdspan_real_memory(device_mdspan_real_span);
  cudaFree(device_mdspan_real_data);

  int* mdspan_device_backed_data = nullptr;
  if (cudaMalloc(&mdspan_device_backed_data, 2 * sizeof(int)) != cudaSuccess)
  {
    return 1;
  }
  const int mdspan_device_backed_host_data[2] = {93, 94};
  if (cudaMemcpy(mdspan_device_backed_data,
                 mdspan_device_backed_host_data,
                 sizeof(mdspan_device_backed_host_data),
                 cudaMemcpyHostToDevice)
      != cudaSuccess)
  {
    return 1;
  }
  const cuda::std::mdspan<int, cuda::std::extents<int, 2>> mdspan_device_backed_span(mdspan_device_backed_data);
  inspect_mdspan_device_backed(mdspan_device_backed_span);
  cudaFree(mdspan_device_backed_data);

  int* const null_data = nullptr;
  const cuda::std::mdspan<int, cuda::std::extents<int, 1>> copy_failure_span(null_data);
  inspect_mdspan_copy_failure(copy_failure_span);

  int managed_mdspan_data[2] = {61, 62};
  const cuda::managed_mdspan<int, cuda::std::extents<int, 2>> managed_mdspan_span(managed_mdspan_data);
  inspect_managed_mdspan(managed_mdspan_span);

  int restrict_mdspan_data[2] = {71, 72};
  const cuda::restrict_mdspan<int, cuda::std::extents<int, 2>> restrict_mdspan_span(restrict_mdspan_data);
  inspect_restrict_mdspan(restrict_mdspan_span);

  int array_data0[2] = {-2, 4};
  int array_data1[2] = {11, -9};
  const cuda::std::array<cuda::std::mdspan<int, cuda::std::extents<int, 2>>, 2> mdspan_array{
    cuda::std::mdspan<int, cuda::std::extents<int, 2>>(array_data0),
    cuda::std::mdspan<int, cuda::std::extents<int, 2>>(array_data1),
  };
  inspect_array(mdspan_array);

  int mutable_data[3] = {1, 2, 3};
  const cuda::std::mdspan<int, cuda::std::dextents<int, 1>> mutable_values(mutable_data, 3);
  inspect_before_update(mutable_values);
  mutable_data[0] = 9;
  mutable_data[1] = -3;
  mutable_data[2] = 15;
  inspect_after_update(mutable_values);
}
