// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <vector>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

using inner_mdspan = cuda::std::mdspan<int, cuda::std::extents<int, 2>>;

[[gnu::noinline]] void inspect_rank1_dynamic(const cuda::std::mdspan<int, cuda::std::dextents<int, 1>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_rank2_static(const cuda::std::mdspan<int, cuda::std::extents<int, 3, 4>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void
inspect_rank2_mixed(const cuda::std::mdspan<int, cuda::std::extents<int, 3, cuda::std::dynamic_extent>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_rank2_dynamic(const cuda::std::mdspan<int, cuda::std::dextents<int, 2>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_rank0(const cuda::std::mdspan<int, cuda::std::extents<int>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_empty(const cuda::std::mdspan<int, cuda::std::extents<int, 0, 3>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void
inspect_layout_left(const cuda::std::mdspan<int, cuda::std::dextents<int, 2>, cuda::std::layout_left>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void
inspect_layout_stride(const cuda::std::mdspan<int, cuda::std::dextents<int, 2>, cuda::std::layout_stride>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_const_element(const cuda::std::mdspan<const int, cuda::std::dextents<int, 1>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_pointer_element(const cuda::std::mdspan<int*, cuda::std::dextents<int, 1>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::mdspan<inner_mdspan, cuda::std::extents<int, 2>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_custom_accessor(
  const cuda::std::
    mdspan<int, cuda::std::dextents<int, 1>, cuda::std::layout_right, cuda::std::aligned_accessor<int, 16>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_vector(const std::vector<cuda::std::mdspan<int, cuda::std::extents<int, 2>>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::mdspan<int, cuda::std::dextents<int, 1>>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::mdspan<int, cuda::std::dextents<int, 1>>& values)
{
  keep_for_debugger(values);
}

int main()
{
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

  int layout_left_data[6];
  cuda::std::mdspan<int, cuda::std::dextents<int, 2>, cuda::std::layout_left> layout_left_span(layout_left_data, 2, 3);
  for (int row = 0; row < 2; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      layout_left_span(row, col) = 2000 + row * 10 + col;
    }
  }
  inspect_layout_left(layout_left_span);

  int layout_stride_data[6];
  const cuda::std::dextents<int, 2> layout_stride_extents(2, 3);
  const cuda::std::layout_stride::mapping<cuda::std::dextents<int, 2>> layout_stride_mapping(
    layout_stride_extents, cuda::std::array<int, 2>{1, 2});
  cuda::std::mdspan<int, cuda::std::dextents<int, 2>, cuda::std::layout_stride> layout_stride_span(
    layout_stride_data, layout_stride_mapping);
  for (int row = 0; row < 2; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      layout_stride_span(row, col) = 3000 + row * 10 + col;
    }
  }
  inspect_layout_stride(layout_stride_span);

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

  int vector_data0[2] = {-2, 4};
  int vector_data1[2] = {11, -9};
  const std::vector<cuda::std::mdspan<int, cuda::std::extents<int, 2>>> mdspan_vector{
    cuda::std::mdspan<int, cuda::std::extents<int, 2>>(vector_data0),
    cuda::std::mdspan<int, cuda::std::extents<int, 2>>(vector_data1),
  };
  inspect_vector(mdspan_vector);

  int mutable_data[3] = {1, 2, 3};
  const cuda::std::mdspan<int, cuda::std::dextents<int, 1>> mutable_values(mutable_data, 3);
  inspect_before_update(mutable_values);
  mutable_data[0] = 9;
  mutable_data[1] = -3;
  mutable_data[2] = 15;
  inspect_after_update(mutable_values);
}
