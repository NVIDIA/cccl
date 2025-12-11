//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/mdspan>
#include <cuda/std/utility>

#include <cuda/experimental/__container/mdarray_device.cuh>
#include <cuda/experimental/__container/mdarray_host.cuh>

#include "testing.cuh"

template <typename View, typename ElementType>
struct CheckInitOp
{
  View view;
  bool* d_result;

  template <typename IndexType, typename... Indices>
  _CCCL_DEVICE void operator()(IndexType, Indices... indices)
  {
    if (view(indices...) != ElementType{})
    {
      *d_result = false;
    }
  }
};

template <typename View>
struct CehckSequenceOp
{
  View view;
  bool* d_result;

  template <typename IndexType, typename... Indices>
  _CCCL_DEVICE void operator()(IndexType i, Indices... indices)
  {
    if (view(indices...) != i)
    {
      *d_result = false;
    }
  }
};

template <typename View>
struct SequenceOp
{
  View view;

  template <typename IndexType, typename... Indices>
  _CCCL_DEVICE void operator()(IndexType i, Indices... indices)
  {
    view(indices...) = i;
  }
};

template <typename ElementType, typename ExtentsType, typename LayoutPolicy>
bool check_init(const cuda::device_mdspan<ElementType, ExtentsType, LayoutPolicy>& mdspan)
{
  using view_type = cuda::device_mdspan<ElementType, ExtentsType, LayoutPolicy>;
  thrust::device_vector<bool> d_result(1, true);
  CheckInitOp<view_type, ElementType> op{mdspan, thrust::raw_pointer_cast(d_result.data())};
  CUDAX_REQUIRE(cub::DeviceFor::ForEachInExtents(mdspan.extents(), op) == cudaSuccess);
  return d_result[0];
}

template <typename View>
bool check_sequence(const View& view)
{
  thrust::device_vector<bool> d_result(1, true);
  CehckSequenceOp<View> op{view, thrust::raw_pointer_cast(d_result.data())};
  CUDAX_REQUIRE(cub::DeviceFor::ForEachInLayout(view.mapping(), op) == cudaSuccess);
  return d_result[0];
}

template <typename MdspanLeftType, typename MdspanRightType>
bool are_mdspan_equal(const MdspanLeftType& lhs, const MdspanRightType& rhs, bool skip_data_handle = false)
{
  static_assert(cuda::std::is_same_v<typename MdspanLeftType::element_type, typename MdspanRightType::element_type>);
  static_assert(cuda::std::is_same_v<typename MdspanLeftType::size_type, typename MdspanRightType::size_type>);
  static_assert(cuda::std::is_same_v<typename MdspanLeftType::index_type, typename MdspanRightType::index_type>);
  static_assert(cuda::std::is_same_v<typename MdspanLeftType::rank_type, typename MdspanRightType::rank_type>);
  static_assert(
    cuda::std::is_same_v<typename MdspanLeftType::data_handle_type, typename MdspanRightType::data_handle_type>);
  return lhs.mapping() == rhs.mapping() && (skip_data_handle || lhs.data_handle() == rhs.data_handle());
}

/***********************************************************************************************************************
 * Test Cases
 **********************************************************************************************************************/

C2H_TEST("cudax::mdarray", "[mdarray][constructor][default]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  using view_type    = typename d_mdarray_t::view_type;
  d_mdarray_t d_mdarray{};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{}));
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][mapping]")
{
  {
    using extents_type = cuda::std::dims<2>;
    using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
    using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
    using mapping_type = cuda::std::layout_right::mapping<extents_type>;
    mapping_type mapping{extents_type{2, 3}};
    d_mdarray_t d_mdarray{mapping};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, mapping}, true));
    CUDAX_REQUIRE(check_init(d_mdarray.view()));

    auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
    d_mdarray_t d_mdarray2{mapping, allocator};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, mapping}, true));
    CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
    CUDAX_REQUIRE(check_init(d_mdarray2.view()));
  }
  {
    using extents_type = cuda::std::dims<2>;
    using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_stride>;
    using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_stride>;
    using mapping_type = cuda::std::layout_stride::mapping<extents_type>;
    mapping_type mapping{extents_type{2, 3}, cuda::std::array<size_t, 2>{3, 1}};
    d_mdarray_t d_mdarray{mapping};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, mapping}, true));
    CUDAX_REQUIRE(check_init(d_mdarray.view()));

    auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
    d_mdarray_t d_mdarray2{mapping, allocator};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, mapping}, true));
    CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
    CUDAX_REQUIRE(check_init(d_mdarray2.view()));
  }
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][extents]")
{
  using extents_type = cuda::std::extents<int, 3, 2, 4>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_left>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_left>;
  extents_type extents{};
  d_mdarray_t d_mdarray{extents};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, extents}, true));
  CUDAX_REQUIRE(check_init(d_mdarray.view()));

  auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
  d_mdarray_t d_mdarray2{extents, allocator};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, extents}, true));
  CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
  CUDAX_REQUIRE(check_init(d_mdarray2.view()));
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][other index types]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  d_mdarray_t d_mdarray{5, 7};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, 5, 7}, true));
  CUDAX_REQUIRE(check_init(d_mdarray.view()));
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][array]")
{
  {
    using extents_type = cuda::std::dims<3>;
    using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
    using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
    ::cuda::std::array<int, 3> extents{5, 7, 9};
    d_mdarray_t d_mdarray{extents};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(check_init(d_mdarray.view()));

    auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
    d_mdarray_t d_mdarray2{extents, allocator};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
    CUDAX_REQUIRE(check_init(d_mdarray2.view()));
  }
  {
    using extents_type = cuda::std::extents<int, 3, 2, 4>;
    using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
    using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
    ::cuda::std::array<int, 3> extents{3, 2, 4};
    d_mdarray_t d_mdarray{extents};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(check_init(d_mdarray.view()));

    auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
    d_mdarray_t d_mdarray2{extents, allocator};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
    CUDAX_REQUIRE(check_init(d_mdarray2.view()));
  }
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][span]")
{
  {
    using extents_type = cuda::std::dims<3>;
    using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
    using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
    int array[3]       = {34, 7, 3};
    ::cuda::std::span<int, 3> extents{array};
    d_mdarray_t d_mdarray{extents};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(check_init(d_mdarray.view()));

    auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
    d_mdarray_t d_mdarray2{extents, allocator};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
    CUDAX_REQUIRE(check_init(d_mdarray2.view()));
  }
  {
    using extents_type = cuda::std::extents<int, 4, 57, 5>;
    using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
    using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
    int array[3]       = {4, 57, 5};
    ::cuda::std::span<int, 3> extents{array};
    d_mdarray_t d_mdarray{extents};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(check_init(d_mdarray.view()));

    auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
    d_mdarray_t d_mdarray2{extents, allocator};
    CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), mdspan_t{nullptr, extents}, true));
    CUDAX_REQUIRE(d_mdarray2.get_allocator() == allocator);
    CUDAX_REQUIRE(check_init(d_mdarray2.view()));
  }
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][copy]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  d_mdarray_t d_mdarray1{5, 7};
  CUDAX_REQUIRE(
    cub::DeviceFor::ForEachInLayout(d_mdarray1.mapping(), SequenceOp<mdspan_t>{d_mdarray1.view()}) == cudaSuccess);

  d_mdarray_t d_mdarray2{d_mdarray1};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), static_cast<mdspan_t>(d_mdarray1), true));
  CUDAX_REQUIRE(d_mdarray2.get_allocator() == d_mdarray1.get_allocator());
  CUDAX_REQUIRE(check_sequence(d_mdarray2.view()));

  auto allocator = cuda::mr::make_shared_resource<cuda::device_memory_pool>(cuda::device_ref{0});
  d_mdarray_t d_mdarray3{d_mdarray1, allocator};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray3), static_cast<mdspan_t>(d_mdarray1), true));
  CUDAX_REQUIRE(d_mdarray3.get_allocator() == allocator);
  CUDAX_REQUIRE(check_sequence(d_mdarray3.view()));
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][move]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  d_mdarray_t d_mdarray1{5, 7};
  CUDAX_REQUIRE(
    cub::DeviceFor::ForEachInLayout(d_mdarray1.mapping(), SequenceOp<mdspan_t>{d_mdarray1.view()}) == cudaSuccess);
  d_mdarray_t d_mdarray1_copy{d_mdarray1};
  auto data_handle = d_mdarray1_copy.data_handle();

  d_mdarray_t d_mdarray2{cuda::std::move(d_mdarray1)};
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), static_cast<mdspan_t>(d_mdarray1_copy), true));
  CUDAX_REQUIRE(d_mdarray2.data_handle() != data_handle);
  CUDAX_REQUIRE(d_mdarray2.get_allocator() == d_mdarray1_copy.get_allocator());
  CUDAX_REQUIRE(check_sequence(d_mdarray2.view()));

  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray1), mdspan_t{}));
  CUDAX_REQUIRE(d_mdarray1.data_handle() == nullptr);
}

C2H_TEST("cudax::mdarray", "[mdarray][copy assignment]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  d_mdarray_t d_mdarray1{5, 7};
  CUDAX_REQUIRE(
    cub::DeviceFor::ForEachInLayout(d_mdarray1.mapping(), SequenceOp<mdspan_t>{d_mdarray1.view()}) == cudaSuccess);

  d_mdarray_t d_mdarray2{};
  d_mdarray2 = d_mdarray1; // need reallocation
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), static_cast<mdspan_t>(d_mdarray1), true));
  CUDAX_REQUIRE(d_mdarray2.get_allocator() == d_mdarray1.get_allocator());
  CUDAX_REQUIRE(check_sequence(d_mdarray2.view()));

  d_mdarray_t d_mdarray3{5, 7};
  d_mdarray3 = d_mdarray1; // no reallocation
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray3), static_cast<mdspan_t>(d_mdarray1), true));
  CUDAX_REQUIRE(d_mdarray3.get_allocator() == d_mdarray1.get_allocator());
  CUDAX_REQUIRE(check_sequence(d_mdarray3.view()));
}

C2H_TEST("cudax::mdarray", "[mdarray][move assignment]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using mdspan_t     = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  d_mdarray_t d_mdarray1{5, 7};
  CUDAX_REQUIRE(
    cub::DeviceFor::ForEachInLayout(d_mdarray1.mapping(), SequenceOp<mdspan_t>{d_mdarray1.view()}) == cudaSuccess);
  d_mdarray_t d_mdarray1_copy{d_mdarray1};
  auto data_handle = d_mdarray1_copy.data_handle();

  d_mdarray_t d_mdarray2{2, 8};
  d_mdarray2 = cuda::std::move(d_mdarray1);
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray2), static_cast<mdspan_t>(d_mdarray1_copy), true));
  CUDAX_REQUIRE(d_mdarray2.data_handle() != data_handle);
  CUDAX_REQUIRE(d_mdarray2.get_allocator() == d_mdarray1_copy.get_allocator());
  CUDAX_REQUIRE(check_sequence(d_mdarray2.view()));

  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(d_mdarray1), mdspan_t{}));
  CUDAX_REQUIRE(d_mdarray1.data_handle() == nullptr);
}

C2H_TEST("cudax::mdarray", "[mdarray][other methods]")
{
  using extents_type     = cuda::std::dims<2>;
  using d_mdarray_t      = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using d_mdspan_t       = cuda::device_mdspan<int, extents_type, cuda::std::layout_right>;
  using mdspan_t         = cuda::std::mdspan<int, extents_type, cuda::std::layout_right>;
  using const_d_mdspan_t = cuda::device_mdspan<const int, extents_type, cuda::std::layout_right>;
  using const_mdspan_t   = cuda::std::mdspan<const int, extents_type, cuda::std::layout_right>;
  d_mdarray_t d_mdarray1{5, 7};
  const auto& const_d_mdarray1 = d_mdarray1;
  auto view                    = d_mdarray1.view();
  auto const_view              = const_d_mdarray1.view();
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<mdspan_t>(view), mdspan_t{nullptr, 5, 7}, true));
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<d_mdspan_t>(d_mdarray1), mdspan_t{nullptr, 5, 7}, true));
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<const_mdspan_t>(const_view), const_mdspan_t{nullptr, 5, 7}, true));
  CUDAX_REQUIRE(are_mdspan_equal(static_cast<const_d_mdspan_t>(const_d_mdarray1), const_mdspan_t{nullptr, 5, 7}, true));
  static_assert(noexcept(d_mdarray1.view()));
  static_assert(noexcept(const_d_mdarray1.view()));
  static_assert(noexcept(static_cast<d_mdspan_t>(d_mdarray1)));
}
