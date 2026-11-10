// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/memory.h>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <unittest/unittest.h>

struct plus_one
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr int operator()(const int val) const noexcept
  {
    return val + 1;
  }
};

using thrust_pointer = thrust::pointer<int, thrust::device_system_tag, int&>;

template <class Iter>
using transform_t = thrust::transform_iterator<plus_one, Iter>;

template <class Iter>
using cuda_transform_t = cuda::transform_iterator<plus_one, Iter>;

template <class Iter>
using transform_output_t = thrust::transform_output_iterator<plus_one, Iter>;

template <class Iter>
using transform_input_output_t = thrust::transform_input_output_iterator<plus_one, plus_one, Iter>;

template <class Iter, class Pointer>
void check_contiguous_storage()
{
  STATIC_REQUIRE(cuda::std::__can_to_address<Iter>);
  STATIC_REQUIRE(cuda::__is_contiguous_iterator_or_adaptor_v<Iter>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, Pointer>);
  STATIC_REQUIRE(cuda::__iter_can_rebind<Iter, cuda::std::remove_pointer_t<Pointer>>);
}

template <class Iter>
void check_contiguous_adaptor()
{
  STATIC_REQUIRE(cuda::__is_contiguous_iterator_or_adaptor_v<Iter>);
  STATIC_REQUIRE(cuda::__iter_can_unwrap<Iter>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, int*>);
  STATIC_REQUIRE(cuda::__iter_can_rebind<Iter, int>);
}

// Output adaptors unwrap to their storage, but cannot be rebuilt over a copy of it
template <class Iter>
void check_output_adaptor()
{
  STATIC_REQUIRE(!cuda::__is_contiguous_iterator_or_adaptor_v<Iter>);
  STATIC_REQUIRE(!cuda::__iter_can_unwrap<Iter>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<Iter>, int*>);
}

TEST_CASE("contiguous iterator adaptor thrust storage", "[iterators]")
{
  // device_ptr storage cannot be rebuilt over a const pointer
  STATIC_REQUIRE(!cuda::__iter_can_rebind<thrust::device_ptr<int>, const int>);
  check_contiguous_storage<thrust_pointer, int*>();
  check_contiguous_storage<thrust::device_ptr<int>, int*>();
  check_contiguous_storage<thrust::host_vector<int>::iterator, int*>();
  check_contiguous_storage<thrust::host_vector<int>::const_iterator, const int*>();
  check_contiguous_storage<thrust::device_vector<int>::iterator, int*>();
  check_contiguous_storage<thrust::device_vector<int>::const_iterator, const int*>();

  int staged[] = {10, 20, 30, 40};

  thrust::device_vector<int> device_values(4);
  int* const storage = thrust::raw_pointer_cast(device_values.data());

  thrust_pointer pointer{storage};
  REQUIRE(cuda::__unwrap_iterator_adaptor(pointer) == storage);
  cuda::__rebind_iterator_adaptor(pointer, staged);
  REQUIRE(cuda::__unwrap_iterator_adaptor(pointer) == staged);

  thrust::device_ptr<int> device_pointer = device_values.data();
  REQUIRE(cuda::__unwrap_iterator_adaptor(device_pointer) == storage);
  cuda::__rebind_iterator_adaptor(device_pointer, staged);
  REQUIRE(cuda::__unwrap_iterator_adaptor(device_pointer) == staged);

  auto device_iter = device_values.begin();
  REQUIRE(cuda::__unwrap_iterator_adaptor(device_iter) == storage);
  cuda::__rebind_iterator_adaptor(device_iter, staged);
  REQUIRE(cuda::__unwrap_iterator_adaptor(device_iter) == staged);

  thrust::host_vector<int> host_values(4);
  auto host_iter = host_values.begin();
  REQUIRE(cuda::__unwrap_iterator_adaptor(host_iter) == host_values.data());
  cuda::__rebind_iterator_adaptor(host_iter, staged);
  REQUIRE(cuda::__unwrap_iterator_adaptor(host_iter) == staged);
}

TEST_CASE("contiguous iterator adaptor transform_iterator", "[iterators]")
{
  check_contiguous_adaptor<transform_t<int*>>();
  check_contiguous_adaptor<transform_t<thrust_pointer>>();
  check_contiguous_adaptor<transform_t<thrust::device_ptr<int>>>();
  check_contiguous_adaptor<transform_t<thrust::host_vector<int>::iterator>>();
  check_contiguous_adaptor<transform_t<thrust::device_vector<int>::iterator>>();

  int input[]  = {1, 2, 3, 4};
  int staged[] = {10, 20, 30, 40};

  auto iter = thrust::make_transform_iterator(input, plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(iter + 2) == input + 2);
  REQUIRE(*iter == 2);

  iter.__rebind(staged);
  REQUIRE(iter.base() == staged);
  REQUIRE(*iter == 11);

  // The rebuilt iterator reads the host buffer instead of device memory
  thrust::device_vector<int> device_values(4);
  auto device_iter = thrust::make_transform_iterator(device_values.begin(), plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(device_iter) == thrust::raw_pointer_cast(device_values.data()));

  device_iter.__rebind(staged);
  REQUIRE(cuda::__unwrap_iterator_adaptor(device_iter) == staged);
}

TEST_CASE("contiguous iterator adaptor nested transform_iterator", "[iterators]")
{
  check_contiguous_adaptor<transform_t<transform_t<int*>>>();
  check_contiguous_adaptor<transform_t<transform_t<thrust::device_ptr<int>>>>();
  check_contiguous_adaptor<transform_t<transform_t<thrust::host_vector<int>::iterator>>>();
  check_contiguous_adaptor<transform_t<transform_t<transform_t<int*>>>>();

  int input[]  = {1, 2, 3, 4};
  int staged[] = {10, 20, 30, 40};

  auto inner = thrust::make_transform_iterator(input, plus_one{});
  auto outer = thrust::make_transform_iterator(inner, plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(outer) == input);
  REQUIRE(*outer == 3);

  outer.__rebind(staged);
  REQUIRE(outer.base().base() == staged);
  REQUIRE(*outer == 12);

  auto outermost = thrust::make_transform_iterator(outer, plus_one{});
  REQUIRE(*outermost == 13);
  cuda::__rebind_iterator_adaptor(outermost, input);
  REQUIRE(cuda::__unwrap_iterator_adaptor(outermost) == input);
  REQUIRE(*outermost == 4);
}

TEST_CASE("contiguous iterator adaptor mixed thrust and libcu++ transform_iterator", "[iterators]")
{
  check_contiguous_adaptor<transform_t<cuda_transform_t<int*>>>();
  check_contiguous_adaptor<cuda_transform_t<transform_t<thrust::device_ptr<int>>>>();
  check_contiguous_adaptor<cuda_transform_t<transform_t<transform_t<thrust::device_ptr<int>>>>>();

  int input[]  = {1, 2, 3, 4};
  int staged[] = {10, 20, 30, 40};

  auto iter = cuda::make_transform_iterator(thrust::make_transform_iterator(input, plus_one{}), plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == input);
  REQUIRE(*iter == 3);

  iter.__rebind(staged);
  REQUIRE(iter.base().base() == staged);
  REQUIRE(*iter == 12);
}

TEST_CASE("contiguous iterator adaptor transform_iterator over non-contiguous iterators", "[iterators]")
{
  // The base of a counting iterator is its counter, so it is not unwrapped any further
  using counting = thrust::counting_iterator<int>;
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<counting>, counting>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<counting>, int>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<transform_t<counting>>, counting>);
  STATIC_REQUIRE(!cuda::__is_contiguous_iterator_or_adaptor_v<transform_t<counting>>);
  STATIC_REQUIRE(!cuda::__is_contiguous_iterator_or_adaptor_v<transform_t<cuda::counting_iterator<int>>>);
  STATIC_REQUIRE(!cuda::__iter_can_unwrap<counting>);
  STATIC_REQUIRE(!cuda::__iter_can_unwrap<transform_t<counting>>);
  STATIC_REQUIRE(!cuda::__iter_can_unwrap<transform_t<transform_t<counting>>>);

  const auto iter = thrust::make_transform_iterator(counting{5}, plus_one{});
  REQUIRE(*cuda::__unwrap_iterator_adaptor(iter) == 5);

  // A reverse iterator unwraps to a pointer, but does not traverse the storage forward
  using reverse = cuda::std::reverse_iterator<int*>;
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::__unwrapped_iterator_t<transform_t<reverse>>, int*>);
  STATIC_REQUIRE(!cuda::__is_contiguous_iterator_or_adaptor_v<transform_t<reverse>>);
  STATIC_REQUIRE(!cuda::__iter_can_unwrap<transform_t<reverse>>);
}

TEST_CASE("contiguous iterator adaptor transform_output_iterator", "[iterators]")
{
  check_output_adaptor<transform_output_t<int*>>();
  check_output_adaptor<transform_output_t<transform_t<thrust::device_ptr<int>>>>();
  STATIC_REQUIRE(!cuda::__is_contiguous_iterator_or_adaptor_v<transform_output_t<thrust::counting_iterator<int>>>);

  int output[] = {1, 2, 3, 4};

  const auto iter = thrust::make_transform_output_iterator(output, plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == output);

  const auto nested =
    thrust::make_transform_output_iterator(thrust::make_transform_iterator(output, plus_one{}), plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(nested) == output);
}

TEST_CASE("contiguous iterator adaptor transform_input_output_iterator", "[iterators]")
{
  check_output_adaptor<transform_input_output_t<int*>>();
  check_output_adaptor<transform_input_output_t<thrust::host_vector<int>::iterator>>();
  STATIC_REQUIRE(
    !cuda::__is_contiguous_iterator_or_adaptor_v<transform_input_output_t<thrust::counting_iterator<int>>>);

  int values[] = {1, 2, 3, 4};

  const auto iter = thrust::make_transform_input_output_iterator(values, plus_one{}, plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == values);
  REQUIRE(*iter == 2);

  thrust::host_vector<int> host_values(4);
  const auto over_vector = thrust::make_transform_input_output_iterator(host_values.begin(), plus_one{}, plus_one{});
  REQUIRE(cuda::__unwrap_iterator_adaptor(over_vector) == host_values.data());
}
