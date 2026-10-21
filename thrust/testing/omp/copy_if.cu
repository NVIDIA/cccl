// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <iterator>
#include <new>

#include <unittest/unittest.h>

namespace
{
struct is_even
{
  bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};

// Returns memory 7 bytes past a cache line boundary, which is legal for a char allocator. Every eighth 8-byte word in
// it then crosses a cache line.
struct misaligning_allocator
{
  using value_type = char;

  char* allocate(std::size_t n)
  {
    return static_cast<char*>(::operator new(n + 7, std::align_val_t{64})) + 7;
  }

  void deallocate(char* p, std::size_t)
  {
    ::operator delete(p - 7, std::align_val_t{64});
  }
};

// An output iterator without comparison operators or a difference
struct output_iterator_without_comparisons
{
  using iterator_category = std::output_iterator_tag;
  using value_type        = void;
  using difference_type   = std::ptrdiff_t;
  using pointer           = void;
  using reference         = int&;

  int* ptr;

  int& operator*() const
  {
    return *ptr;
  }

  output_iterator_without_comparisons& operator++()
  {
    ++ptr;
    return *this;
  }

  output_iterator_without_comparisons operator++(int)
  {
    return {ptr++};
  }

  output_iterator_without_comparisons operator+(std::ptrdiff_t i) const
  {
    return {ptr + i};
  }
};
} // namespace

TEST_CASE("TestOmpCopyIfInsideParallelRegion", "[copy_if]")
{
  // Large enough that copy_if splits the work between threads when it is called outside of a parallel region
  const int n = 1 << 20;

  const thrust::host_vector<int> h_input   = unittest::random_integers<int>(n);
  const thrust::device_vector<int> d_input = h_input;

  thrust::host_vector<int> h_result(n, thrust::no_init);
  h_result.resize(thrust::copy_if(h_input.begin(), h_input.end(), h_result.begin(), is_even{}) - h_result.begin());

  thrust::device_vector<int> d_results[2] = {
    thrust::device_vector<int>(n, thrust::no_init), thrust::device_vector<int>(n, thrust::no_init)};
  std::ptrdiff_t d_result_sizes[2] = {};

  // Nested parallelism is disabled by default, so the parallel regions started by copy_if get a single thread each,
  // although omp_get_max_threads() still reports the full number of threads
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 2; ++i)
  {
    d_result_sizes[i] =
      thrust::copy_if(d_input.begin(), d_input.end(), d_results[i].begin(), is_even{}) - d_results[i].begin();
  }

  for (int i = 0; i < 2; ++i)
  {
    d_results[i].resize(d_result_sizes[i]);
    REQUIRE(d_results[i] == h_result);
  }
}

TEST_CASE("TestOmpCopyIfMisalignedTemporaryStorage", "[copy_if]")
{
  const int n = 1 << 14;

  const thrust::host_vector<int> input = unittest::random_integers<int>(n);

  thrust::host_vector<int> expected(n, thrust::no_init);
  expected.resize(thrust::copy_if(input.begin(), input.end(), expected.begin(), is_even{}) - expected.begin());

  misaligning_allocator alloc;
  thrust::host_vector<int> result(n, thrust::no_init);

  // A torn read of a status word only shows up in some runs
  for (int run = 0; run < 100; ++run)
  {
    const int* const end =
      thrust::copy_if(thrust::omp::par(alloc), input.data(), input.data() + n, result.data(), is_even{});
    REQUIRE(end - result.data() == static_cast<std::ptrdiff_t>(expected.size()));
    REQUIRE(thrust::equal(expected.begin(), expected.end(), result.begin()));
  }
}

TEST_CASE("TestOmpCopyIfOutputIteratorWithoutComparisons", "[copy_if]")
{
  for (const int n : {100, 1 << 16})
  {
    thrust::host_vector<int> input(n);
    thrust::sequence(input.begin(), input.end());
    thrust::host_vector<int> result(n);

    const output_iterator_without_comparisons end = thrust::copy_if(
      thrust::omp::par, input.data(), input.data() + n, output_iterator_without_comparisons{result.data()}, is_even{});

    REQUIRE(end.ptr - result.data() == n / 2);
    for (int i = 0; i < n / 2; ++i)
    {
      REQUIRE(result[i] == 2 * i);
    }
  }
}

TEST_CASE("TestOmpCopyIfPredicateTakingLvalueReferenceToZipReference", "[copy_if]")
{
  // Like the generic implementation, copy_if passes elements to pred as const lvalues, also for a zip_iterator, whose
  // reference is a prvalue
  for (const int n : {100, 1 << 16})
  {
    thrust::device_vector<int> a(n);
    thrust::device_vector<int> b(n, n / 2);
    thrust::sequence(a.begin(), a.end());
    thrust::device_vector<int> out_a(n);
    thrust::device_vector<int> out_b(n);

    const auto first = thrust::make_zip_iterator(a.begin(), b.begin());
    const auto end =
      thrust::copy_if(first, first + n, thrust::make_zip_iterator(out_a.begin(), out_b.begin()), [](auto& t) {
        return thrust::get<0>(t) < thrust::get<1>(t);
      });

    REQUIRE(end - thrust::make_zip_iterator(out_a.begin(), out_b.begin()) == n / 2);
    out_a.resize(n / 2);
    a.resize(n / 2);
    REQUIRE(out_a == a);
  }
}

TEST_CASE("TestOmpCopyIfNarrowDifferenceType", "[copy_if]")
{
  // n - tile_begin has type int for a short difference_type
  const int n = 20000;

  const thrust::counting_iterator<int, thrust::use_default, thrust::use_default, short> first(0);
  thrust::device_vector<int> result(n);

  const auto end = thrust::copy_if(first, first + static_cast<short>(n), result.begin(), is_even{});

  REQUIRE(end - result.begin() == n / 2);
  for (int i = 0; i < n / 2; ++i)
  {
    REQUIRE(result[i] == 2 * i);
  }
}
