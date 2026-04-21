//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__multi_gpu/shard.h>

#include <testing.cuh>

namespace
{
using iter_t       = int*;
using const_iter_t = const int*;

template <class Range, class Device, class = void>
struct can_deduce_range_shard : cuda::std::false_type
{};

template <class Range, class Device>
struct can_deduce_range_shard<
  Range,
  Device,
  cuda::std::void_t<decltype(cudax::shard{cuda::std::declval<Range>(), cuda::std::declval<Device>()})>>
    : cuda::std::true_type
{};

template <class Iterator, class Sentinel, class Device, class = void>
struct can_deduce_iterator_shard : cuda::std::false_type
{};

template <class Iterator, class Sentinel, class Device>
struct can_deduce_iterator_shard<
  Iterator,
  Sentinel,
  Device,
  cuda::std::void_t<decltype(cudax::shard{
    cuda::std::declval<Iterator>(), cuda::std::declval<Sentinel>(), cuda::std::declval<Device>()})>>
    : cuda::std::true_type
{};
} // namespace

using iterator_pair_device_ref_deduction_t     = decltype(cudax::shard{
  cuda::std::declval<iter_t>(), cuda::std::declval<iter_t>(), cuda::std::declval<cuda::device_ref>()});
using iterator_pair_logical_device_deduction_t = decltype(cudax::shard{
  cuda::std::declval<iter_t>(), cuda::std::declval<iter_t>(), cuda::std::declval<cudax::logical_device>()});
using range_device_ref_deduction_t =
  decltype(cudax::shard{cuda::std::declval<int (&)[4]>(), cuda::std::declval<cuda::device_ref>()});
using range_logical_device_deduction_t =
  decltype(cudax::shard{cuda::std::declval<int (&)[4]>(), cuda::std::declval<cudax::logical_device>()});

C2H_CCCLRT_TEST("shard static tests", "[multi_gpu][shard]")
{
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::shard<iter_t>::iterator_type, iter_t>);
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::shard<iter_t>::size_type,
                                      cuda::std::make_unsigned_t<cuda::std::iter_difference_t<iter_t>>>);
  STATIC_REQUIRE(cuda::std::is_same_v<cudax::shard<const_iter_t>::iterator_type, const_iter_t>);

  STATIC_REQUIRE(cuda::std::is_constructible_v<cudax::shard<iter_t>, iter_t, iter_t, cuda::device_ref>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cudax::shard<iter_t>, iter_t, iter_t, cudax::logical_device>);
  STATIC_REQUIRE_FALSE(cuda::std::is_constructible_v<cudax::shard<iter_t>, int, cuda::device_ref>);
  STATIC_REQUIRE_FALSE(cuda::std::is_constructible_v<cudax::shard<iter_t>, iter_t, iter_t>);
  STATIC_REQUIRE_FALSE(cuda::std::is_constructible_v<cudax::shard<iter_t>, iter_t, const_iter_t, cuda::device_ref>);
  STATIC_REQUIRE_FALSE(cuda::std::is_constructible_v<cudax::shard<iter_t>, iter_t, iter_t, float*>);

  STATIC_REQUIRE(cuda::std::is_same_v<iterator_pair_device_ref_deduction_t, cudax::shard<iter_t>>);
  STATIC_REQUIRE(cuda::std::is_same_v<iterator_pair_logical_device_deduction_t, cudax::shard<iter_t>>);
  STATIC_REQUIRE(cuda::std::is_same_v<range_device_ref_deduction_t, cudax::shard<iter_t>>);
  STATIC_REQUIRE(cuda::std::is_same_v<range_logical_device_deduction_t, cudax::shard<iter_t>>);

  STATIC_REQUIRE(can_deduce_range_shard<int (&)[4], cuda::device_ref>::value);
  STATIC_REQUIRE(can_deduce_range_shard<const int (&)[4], cuda::device_ref>::value);
  STATIC_REQUIRE(can_deduce_range_shard<cuda::std::span<int>, cuda::device_ref>::value);
  STATIC_REQUIRE_FALSE(can_deduce_range_shard<int, cuda::device_ref>::value);
  STATIC_REQUIRE_FALSE(can_deduce_range_shard<int*, cuda::device_ref>::value);
  STATIC_REQUIRE_FALSE(can_deduce_range_shard<int (&)[4], float*>::value);

  STATIC_REQUIRE(can_deduce_iterator_shard<iter_t, iter_t, cuda::device_ref>::value);
  STATIC_REQUIRE_FALSE(can_deduce_iterator_shard<iter_t, const_iter_t, cuda::device_ref>::value);
  STATIC_REQUIRE_FALSE(can_deduce_iterator_shard<iter_t, iter_t, float*>::value);

  STATIC_REQUIRE(cudax::__range_of_shards<cudax::shard<iter_t>(&)[2]>);
  STATIC_REQUIRE(cudax::__range_of_shards<cudax::shard<const_iter_t>(&)[2]>);
  STATIC_REQUIRE_FALSE(cudax::__range_of_shards<int (&)[4]>);
  STATIC_REQUIRE_FALSE(cudax::__range_of_shards<cudax::shard<iter_t>>);
}

C2H_CCCLRT_TEST("shard stores iterator bounds", "[multi_gpu][shard]")
{
  int data[] = {0, 1, 2, 3};
  auto first = data + 1;
  auto last  = data + 4;
  auto dev   = cuda::device_ref{0};

  auto s = cudax::shard{first, last, dev};

  REQUIRE(s.begin() == first);
  REQUIRE(s.end() == last);
  REQUIRE(s.size() == 3);
  REQUIRE(s.device().kind() == cudax::logical_device::kinds::device);
  REQUIRE(s.device().underlying_device() == dev);
}

C2H_CCCLRT_TEST("shard can be constructed from a range", "[multi_gpu][shard]")
{
  int data[] = {0, 1, 2, 3};
  auto dev   = cuda::device_ref{0};
  auto ldev  = cudax::logical_device{dev};

  auto device_ref_shard     = cudax::shard{data, dev};
  auto logical_device_shard = cudax::shard{data, ldev};

  REQUIRE(device_ref_shard.begin() == data);
  REQUIRE(device_ref_shard.end() == data + 4);
  REQUIRE(device_ref_shard.size() == 4);
  REQUIRE(device_ref_shard.device().underlying_device() == dev);

  REQUIRE(logical_device_shard.begin() == data);
  REQUIRE(logical_device_shard.end() == data + 4);
  REQUIRE(logical_device_shard.size() == 4);
  REQUIRE(logical_device_shard.device() == ldev);
}

C2H_CCCLRT_TEST("shard supports empty ranges", "[multi_gpu][shard]")
{
  int data[] = {0, 1, 2, 3};
  auto dev   = cuda::device_ref{0};

  auto s          = cudax::shard{data, data, dev};
  auto empty_span = cuda::std::span<int>{data, /*__count=*/0};
  auto empty      = cudax::shard{empty_span, dev};

  REQUIRE(s.begin() == data);
  REQUIRE(s.end() == data);
  REQUIRE(s.size() == 0);

  REQUIRE(&*empty.begin() == data);
  REQUIRE(&*empty.end() == data);
  REQUIRE(empty.size() == 0);
}

C2H_CCCLRT_TEST("shard preserves const range iterators", "[multi_gpu][shard]")
{
  const int data[] = {0, 1, 2, 3};
  auto dev         = cuda::device_ref{0};

  auto s = cudax::shard{data, dev};

  STATIC_REQUIRE(cuda::std::is_same_v<decltype(s)::iterator_type, const int*>);
  REQUIRE(s.begin() == data);
  REQUIRE(s.end() == data + 4);
  REQUIRE(s.size() == 4);
  REQUIRE(s.device().underlying_device() == dev);
}

C2H_CCCLRT_TEST("shard supports singleton ranges", "[multi_gpu][shard]")
{
  int data[] = {0, 1, 2, 3};
  auto dev   = cuda::device_ref{0};

  auto s = cudax::shard{data + 2, data + 3, dev};

  REQUIRE(s.begin() == data + 2);
  REQUIRE(s.end() == data + 3);
  REQUIRE(s.size() == 1);
}
