//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class PopulationIterator, class SampleIterator, class Distance,
//           class UniformRandomNumberGenerator>
// SampleIterator sample(PopulationIterator first, PopulationIterator last,
//                       SampleIterator out, Distance n,
//                       UniformRandomNumberGenerator &&g);

#include <cuda/std/__random_>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

constexpr size_t sample_size = 4;

TEST_GLOBAL_VARIABLE constexpr int host_reservoir_sample1[sample_size]   = {10, 5, 9, 4};
TEST_GLOBAL_VARIABLE constexpr int device_reservoir_sample1[sample_size] = {10, 5, 9, 4};
TEST_GLOBAL_VARIABLE constexpr int host_reservoir_sample2[sample_size]   = {5, 2, 10, 4};
TEST_GLOBAL_VARIABLE constexpr int device_reservoir_sample2[sample_size] = {5, 2, 10, 4};

struct ReservoirSampleExpectations
{
  __host__ __device__ static constexpr const int* get_sample1() noexcept
  {
    unused(host_reservoir_sample1, device_reservoir_sample1);
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return device_reservoir_sample1;), (return host_reservoir_sample1;))
  }
  __host__ __device__ static constexpr const int* get_sample2() noexcept
  {
    unused(host_reservoir_sample2, device_reservoir_sample2);
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return device_reservoir_sample2;), (return host_reservoir_sample2;))
  }
};

TEST_GLOBAL_VARIABLE constexpr int host_selection_sample1[sample_size]   = {1, 4, 6, 7};
TEST_GLOBAL_VARIABLE constexpr int device_selection_sample1[sample_size] = {1, 4, 6, 7};
TEST_GLOBAL_VARIABLE constexpr int host_selection_sample2[sample_size]   = {1, 2, 6, 8};
TEST_GLOBAL_VARIABLE constexpr int device_selection_sample2[sample_size] = {1, 2, 6, 8};

struct SelectionSampleExpectations
{
  __host__ __device__ static constexpr const int* get_sample1() noexcept
  {
    unused(device_selection_sample1, host_selection_sample1);
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return device_selection_sample1;), (return host_selection_sample1;))
  }
  __host__ __device__ static constexpr const int* get_sample2() noexcept
  {
    unused(device_selection_sample2, host_selection_sample2);
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return device_selection_sample2;), (return host_selection_sample2;))
  }
};

template <class IteratorCategory>
struct TestExpectations : public SelectionSampleExpectations
{};

template <>
struct TestExpectations<cuda::std::input_iterator_tag> : public ReservoirSampleExpectations
{};

template <template <class...> class PopulationIteratorType,
          class PopulationItem,
          template <class...> class SampleIteratorType,
          class SampleItem>
__host__ __device__ void test()
{
  using PopulationIterator = PopulationIteratorType<PopulationItem*>;
  using SampleIterator     = SampleIteratorType<SampleItem*>;
  PopulationItem ia[]      = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  constexpr size_t is      = sizeof(ia) / sizeof(ia[0]);
  using Expectations = TestExpectations<typename cuda::std::iterator_traits<PopulationIterator>::iterator_category>;
  SampleItem oa[sample_size];
  [[maybe_unused]] const int* oa1 = Expectations::get_sample1();
  [[maybe_unused]] const int* oa2 = Expectations::get_sample2();
  cuda::std::minstd_rand g;
  SampleIterator end =
    cuda::std::sample(PopulationIterator(ia), PopulationIterator(ia + is), SampleIterator(oa), sample_size, g);
  assert(static_cast<cuda::std::size_t>(base(end) - oa) == cuda::std::min(sample_size, is));
  // sample() is deterministic but non-reproducible;
  // its results can vary between implementations.
  _CCCL_ASSERT(cuda::std::equal(oa, oa + sample_size, oa1), "");
  end = cuda::std::sample(
    PopulationIterator(ia), PopulationIterator(ia + is), SampleIterator(oa), sample_size, cuda::std::move(g));
  assert(static_cast<cuda::std::size_t>(base(end) - oa) == cuda::std::min(sample_size, is));
  _CCCL_ASSERT(cuda::std::equal(oa, oa + sample_size, oa2), "");
}

template <template <class...> class PopulationIteratorType,
          class PopulationItem,
          template <class...> class SampleIteratorType,
          class SampleItem>
__host__ __device__ void test_empty_population()
{
  using PopulationIterator = PopulationIteratorType<PopulationItem*>;
  using SampleIterator     = SampleIteratorType<SampleItem*>;
  PopulationItem ia[]      = {42};
  const unsigned os        = 4;
  SampleItem oa[os];
  cuda::std::minstd_rand g;
  SampleIterator end = cuda::std::sample(PopulationIterator(ia), PopulationIterator(ia), SampleIterator(oa), os, g);
  assert(base(end) == oa);
}

template <template <class...> class PopulationIteratorType,
          class PopulationItem,
          template <class...> class SampleIteratorType,
          class SampleItem>
__host__ __device__ void test_empty_sample()
{
  using PopulationIterator = PopulationIteratorType<PopulationItem*>;
  using SampleIterator     = SampleIteratorType<SampleItem*>;
  PopulationItem ia[]      = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const unsigned is        = sizeof(ia) / sizeof(ia[0]);
  SampleItem oa[1];
  cuda::std::minstd_rand g;
  SampleIterator end = cuda::std::sample(PopulationIterator(ia), PopulationIterator(ia + is), SampleIterator(oa), 0, g);
  assert(base(end) == oa);
}

template <template <class...> class PopulationIteratorType,
          class PopulationItem,
          template <class...> class SampleIteratorType,
          class SampleItem>
__host__ __device__ void test_small_population()
{
  // The population size is less than the sample size.
  using PopulationIterator = PopulationIteratorType<PopulationItem*>;
  using SampleIterator     = SampleIteratorType<SampleItem*>;
  PopulationItem ia[]      = {1, 2, 3, 4, 5};
  const unsigned is        = sizeof(ia) / sizeof(ia[0]);
  const unsigned os        = 8;
  SampleItem oa[os];
  const SampleItem oa1[] = {1, 2, 3, 4, 5};
  cuda::std::minstd_rand g;
  SampleIterator end =
    cuda::std::sample(PopulationIterator(ia), PopulationIterator(ia + is), SampleIterator(oa), os, g);
  assert(static_cast<cuda::std::size_t>(base(end) - oa) == cuda::std::min(os, is));
  using PopulationCategory = typename cuda::std::iterator_traits<PopulationIterator>::iterator_category;
  if (cuda::std::is_base_of<cuda::std::forward_iterator_tag, PopulationCategory>::value)
  {
    assert(cuda::std::equal(oa, base(end), oa1));
  }
  else
  {
    assert(cuda::std::is_permutation(oa, base(end), oa1));
  }
}

int main(int, char**)
{
  test<cpp17_input_iterator, int, random_access_iterator, int>();
  test<forward_iterator, int, cpp17_output_iterator, int>();
  test<forward_iterator, int, random_access_iterator, int>();

  test<cpp17_input_iterator, int, random_access_iterator, double>();
  test<forward_iterator, int, cpp17_output_iterator, double>();
  test<forward_iterator, int, random_access_iterator, double>();

  test_empty_population<cpp17_input_iterator, int, random_access_iterator, int>();
  test_empty_population<forward_iterator, int, cpp17_output_iterator, int>();
  test_empty_population<forward_iterator, int, random_access_iterator, int>();

  test_empty_sample<cpp17_input_iterator, int, random_access_iterator, int>();
  test_empty_sample<forward_iterator, int, cpp17_output_iterator, int>();
  test_empty_sample<forward_iterator, int, random_access_iterator, int>();

  test_small_population<cpp17_input_iterator, int, random_access_iterator, int>();
  test_small_population<forward_iterator, int, cpp17_output_iterator, int>();
  test_small_population<forward_iterator, int, random_access_iterator, int>();

  return 0;
}
