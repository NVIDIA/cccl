// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Test that sequential algorithms correctly handle predicates
// that require implicit type conversions (e.g., float → const double&).
// This validates that wrapped_function properly handles the conversion chain
// when user-provided callables expect a different type than the iterator's value_type.
// When the sequential backend is invoked from device code (CDP) with device_vector
// iterators, wrapped_function also unwraps proxy references (device_reference<T> → T&)
// before the implicit conversion occurs.

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>

#include "catch2_test_helper.h"

// Predicate that accepts const double& — requires implicit conversion
// from float (the vector's value_type) to double.
// In a CDP context with device_vector, this also requires unwrapping the
// proxy reference: device_reference<float> → float → double
struct double_greater_than_two
{
  _CCCL_HOST_DEVICE bool operator()(const double& x) const
  {
    return x > 2.0;
  }
};

// Binary predicate taking const double& arguments
struct double_less
{
  _CCCL_HOST_DEVICE bool operator()(const double& a, const double& b) const
  {
    return a < b;
  }
};

// Binary predicate for equality via double
struct double_equal
{
  _CCCL_HOST_DEVICE bool operator()(const double& a, const double& b) const
  {
    return a == b;
  }
};

// Unary function taking const double&
struct double_negate
{
  _CCCL_HOST_DEVICE void operator()(const double& x) const
  {
    (void) x; // just verify it compiles and runs
  }
};

// Binary function taking const double& for adjacent_difference
struct double_minus
{
  _CCCL_HOST_DEVICE float operator()(const double& a, const double& b) const
  {
    return static_cast<float>(a - b);
  }
};

// Binary function taking const double& for reduce/scan
struct double_plus
{
  _CCCL_HOST_DEVICE float operator()(const double& a, const double& b) const
  {
    return static_cast<float>(a + b);
  }
};

// Unary transform: negate via double conversion
struct double_negate_transform
{
  _CCCL_HOST_DEVICE float operator()(const double& x) const
  {
    return static_cast<float>(-x);
  }
};

TEST_CASE("SequentialFindIfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // Use thrust::seq to force sequential backend
  const auto result = thrust::find_if(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{});
  CHECK(result - vec.begin() == 2);
}

TEST_CASE("SequentialForEachProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f};

  // Validates type conversion for for_each
  thrust::for_each(thrust::seq, vec.begin(), vec.end(), double_negate{});
}

TEST_CASE("SequentialForEachNProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f};

  const auto result = thrust::for_each_n(thrust::seq, vec.begin(), 3, double_negate{});
  CHECK(result == vec.end());
}

TEST_CASE("SequentialMinElementProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{3.0f, 1.0f, 2.0f, 5.0f, 4.0f};

  const auto result = thrust::min_element(thrust::seq, vec.begin(), vec.end(), double_less{});
  CHECK(result - vec.begin() == 1);
}

TEST_CASE("SequentialMaxElementProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{3.0f, 1.0f, 2.0f, 5.0f, 4.0f};

  const auto result = thrust::max_element(thrust::seq, vec.begin(), vec.end(), double_less{});
  CHECK(result - vec.begin() == 3);
}

TEST_CASE("SequentialMinmaxElementProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{3.0f, 1.0f, 2.0f, 5.0f, 4.0f};

  const auto result = thrust::minmax_element(thrust::seq, vec.begin(), vec.end(), double_less{});
  CHECK(result.first - vec.begin() == 1);
  CHECK(result.second - vec.begin() == 3);
}

TEST_CASE("SequentialLowerBoundProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  const auto result = thrust::lower_bound(thrust::seq, vec.begin(), vec.end(), 3.0f, double_less{});
  CHECK(result - vec.begin() == 2);
}

TEST_CASE("SequentialUpperBoundProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  const auto result = thrust::upper_bound(thrust::seq, vec.begin(), vec.end(), 3.0f, double_less{});
  CHECK(result - vec.begin() == 3);
}

TEST_CASE("SequentialRemoveIfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  const auto new_end = thrust::remove_if(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{});
  CHECK(new_end - vec.begin() == 2);
  CHECK(vec[0] == 1.0f);
  CHECK(vec[1] == 2.0f);
}

TEST_CASE("SequentialUniqueProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 1.0f, 2.0f, 2.0f, 3.0f};

  const auto new_end = thrust::unique(thrust::seq, vec.begin(), vec.end(), double_equal{});
  CHECK(new_end - vec.begin() == 3);
}

TEST_CASE("SequentialUniqueCopyProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 1.0f, 2.0f, 2.0f, 3.0f};
  thrust::host_vector<float> output(5);

  const auto new_end = thrust::unique_copy(thrust::seq, input.begin(), input.end(), output.begin(), double_equal{});
  CHECK(new_end - output.begin() == 3);
  CHECK(output[0] == 1.0f);
  CHECK(output[1] == 2.0f);
  CHECK(output[2] == 3.0f);
}

TEST_CASE("SequentialMergeProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> a{1.0f, 3.0f, 5.0f};
  thrust::host_vector<float> b{2.0f, 4.0f, 6.0f};
  thrust::host_vector<float> out(6);

  thrust::merge(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), double_less{});

  CHECK(out[0] == 1.0f);
  CHECK(out[1] == 2.0f);
  CHECK(out[2] == 3.0f);
  CHECK(out[3] == 4.0f);
  CHECK(out[4] == 5.0f);
  CHECK(out[5] == 6.0f);
}

TEST_CASE("SequentialSetDifferenceProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  thrust::host_vector<float> b{2.0f, 4.0f};
  thrust::host_vector<float> out(5);

  const auto new_end =
    thrust::set_difference(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), double_less{});
  CHECK(new_end - out.begin() == 3);
  CHECK(out[0] == 1.0f);
  CHECK(out[1] == 3.0f);
  CHECK(out[2] == 5.0f);
}

TEST_CASE("SequentialSetIntersectionProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  thrust::host_vector<float> b{2.0f, 4.0f, 6.0f};
  thrust::host_vector<float> out(5);

  const auto new_end =
    thrust::set_intersection(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), double_less{});
  CHECK(new_end - out.begin() == 2);
  CHECK(out[0] == 2.0f);
  CHECK(out[1] == 4.0f);
}

TEST_CASE("SequentialSetUnionProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> a{1.0f, 3.0f, 5.0f};
  thrust::host_vector<float> b{2.0f, 3.0f, 4.0f};
  thrust::host_vector<float> out(6);

  const auto new_end =
    thrust::set_union(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), double_less{});
  CHECK(new_end - out.begin() == 5);
  CHECK(out[0] == 1.0f);
  CHECK(out[1] == 2.0f);
  CHECK(out[2] == 3.0f);
  CHECK(out[3] == 4.0f);
  CHECK(out[4] == 5.0f);
}

TEST_CASE("SequentialRemoveCopyIfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  thrust::host_vector<float> output(5);

  const auto new_end =
    thrust::remove_copy_if(thrust::seq, input.begin(), input.end(), output.begin(), double_greater_than_two{});
  CHECK(new_end - output.begin() == 2);
  CHECK(output[0] == 1.0f);
  CHECK(output[1] == 2.0f);
}

TEST_CASE("SequentialAdjacentDifferenceProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 4.0f, 9.0f, 16.0f};
  thrust::host_vector<float> output(4);

  const auto result =
    thrust::adjacent_difference(thrust::seq, input.begin(), input.end(), output.begin(), double_minus{});
  CHECK(result == output.end());
  CHECK(output[0] == 1.0f); // first element copied
  CHECK(output[1] == 3.0f); // 4 - 1
  CHECK(output[2] == 5.0f); // 9 - 4
  CHECK(output[3] == 7.0f); // 16 - 9
}

TEST_CASE("SequentialReduceProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};

  const float result = thrust::reduce(thrust::seq, vec.begin(), vec.end(), 0.0f, double_plus{});
  CHECK(result == 10.0f);
}

TEST_CASE("SequentialInclusiveScanProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  thrust::host_vector<float> output(4);

  thrust::inclusive_scan(thrust::seq, input.begin(), input.end(), output.begin(), double_plus{});
  CHECK(output[0] == 1.0f);
  CHECK(output[1] == 3.0f);
  CHECK(output[2] == 6.0f);
  CHECK(output[3] == 10.0f);
}

TEST_CASE("SequentialExclusiveScanProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  thrust::host_vector<float> output(4);

  thrust::exclusive_scan(thrust::seq, input.begin(), input.end(), output.begin(), 0.0f, double_plus{});
  CHECK(output[0] == 0.0f);
  CHECK(output[1] == 1.0f);
  CHECK(output[2] == 3.0f);
  CHECK(output[3] == 6.0f);
}

TEST_CASE("SequentialStablePartitionCopyProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  thrust::host_vector<float> out_true(5);
  thrust::host_vector<float> out_false(5);

  const auto result = thrust::stable_partition_copy(
    thrust::seq, input.begin(), input.end(), out_true.begin(), out_false.begin(), double_greater_than_two{});
  const auto n_true  = result.first - out_true.begin();
  const auto n_false = result.second - out_false.begin();
  CHECK(n_true == 3);
  CHECK(n_false == 2);
  CHECK(out_true[0] == 3.0f);
  CHECK(out_true[1] == 4.0f);
  CHECK(out_true[2] == 5.0f);
  CHECK(out_false[0] == 1.0f);
  CHECK(out_false[1] == 2.0f);
}

TEST_CASE("SequentialPartitionProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

  const auto mid = thrust::partition(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{});
  CHECK(mid - vec.begin() == 3);
  // all elements in [begin, mid) satisfy pred; none in [mid, end) do
  for (auto it = vec.begin(); it != mid; ++it)
  {
    CHECK(*it > 2.0f);
  }
  for (auto it = mid; it != vec.end(); ++it)
  {
    CHECK(*it <= 2.0f);
  }
}

TEST_CASE("SequentialInclusiveScanWithInitProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  thrust::host_vector<float> output(4);

  thrust::inclusive_scan(thrust::seq, input.begin(), input.end(), output.begin(), 10.0f, double_plus{});
  CHECK(output[0] == 11.0f); // 10 + 1
  CHECK(output[1] == 13.0f); // 11 + 2
  CHECK(output[2] == 16.0f); // 13 + 3
  CHECK(output[3] == 20.0f); // 16 + 4
}

TEST_CASE("SequentialCountIfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  const auto n = thrust::count_if(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{});
  CHECK(n == 3);
}

TEST_CASE("SequentialAllOfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{3.0f, 4.0f, 5.0f};

  CHECK(thrust::all_of(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{}));
}

TEST_CASE("SequentialAnyOfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f, 3.0f};

  CHECK(thrust::any_of(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{}));
}

TEST_CASE("SequentialNoneOfProxyReference", "[sequential][proxy_reference]")
{
  thrust::host_vector<float> vec{1.0f, 2.0f};

  CHECK(thrust::none_of(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{}));
}

// Real proxy-reference test using device_vector
// This validates that wrapped_function correctly unwraps device_reference<T> → T
// before the implicit conversion to const double& occurs
TEST_CASE("SequentialFindIfRealProxyReference", "[sequential][proxy_reference]")
{
  thrust::device_vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // Use thrust::seq to force sequential backend
  // device_vector iterators return device_reference<float>, which is a proxy reference
  // This test validates that wrapped_function unwraps the proxy before conversion
  const auto result = thrust::find_if(thrust::seq, vec.begin(), vec.end(), double_greater_than_two{});
  CHECK(result - vec.begin() == 2);
}
