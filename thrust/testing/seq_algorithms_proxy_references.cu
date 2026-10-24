// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The sequential backend forwards iterators straight to libcu++ algorithms. Iterators whose reference type is a proxy
// (e.g. device_vector's device_reference) must therefore be unwrapped before a user functor sees them, which is what
// thrust::detail::wrapped_function does.
//
// The proxies are provided by a host_vector with an allocator whose pointer is a thrust::pointer tagged with the host
// system. Its iterators dereference to thrust::reference<T, ...>, but everything stays accessible on the host, so
// thrust::seq can run in a plain unit test without a kernel launch.

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cstddef>
#include <new>

#include <catch2/catch_test_macros.hpp>

template <typename T>
struct proxy_allocator
{
  // tagged_reference (like device_reference) provides the swap overload that libcu++'s iter_swap needs
  using value_type = T;
  using pointer    = thrust::pointer<T, thrust::host_system_tag, thrust::tagged_reference<T, thrust::host_system_tag>>;
  using const_pointer =
    thrust::pointer<const T, thrust::host_system_tag, thrust::tagged_reference<const T, thrust::host_system_tag>>;
  using reference       = typename pointer::reference;
  using const_reference = typename const_pointer::reference;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind
  {
    using other = proxy_allocator<U>;
  };

  proxy_allocator() = default;

  template <typename U>
  proxy_allocator(const proxy_allocator<U>&)
  {}

  pointer allocate(std::size_t n)
  {
    return pointer{static_cast<T*>(::operator new(n * sizeof(T)))};
  }

  void deallocate(pointer p, std::size_t)
  {
    ::operator delete(thrust::raw_pointer_cast(p));
  }

  friend bool operator==(const proxy_allocator&, const proxy_allocator&)
  {
    return true;
  }

  friend bool operator!=(const proxy_allocator&, const proxy_allocator&)
  {
    return false;
  }
};

using proxy_vector = thrust::host_vector<float, proxy_allocator<float>>;

static_assert(!::cuda::std::is_same_v<decltype(*::cuda::std::declval<proxy_vector&>().begin()), float&>,
              "proxy_vector must dereference to a proxy reference, otherwise these tests prove nothing");

// probe functors only compile when they receive exactly the raw type, never a proxy reference
struct probe_unary_pred
{
  template <typename T>
  _CCCL_HOST_DEVICE bool operator()(const T& x) const
  {
    static_assert(::cuda::std::is_same_v<T, float>, "proxy reference was not unwrapped");
    return x > 2.0f;
  }
};

template <typename Op>
struct probe_binary_pred
{
  template <typename A, typename B>
  _CCCL_HOST_DEVICE bool operator()(const A& a, const B& b) const
  {
    static_assert(::cuda::std::is_same_v<A, float> && ::cuda::std::is_same_v<B, float>,
                  "proxy reference was not unwrapped");
    return Op{}(a, b);
  }
};

using probe_less  = probe_binary_pred<::cuda::std::less<>>;
using probe_equal = probe_binary_pred<::cuda::std::equal_to<>>;

struct probe_binary_op
{
  template <typename A, typename B>
  _CCCL_HOST_DEVICE float operator()(const A& a, const B& b) const
  {
    static_assert(::cuda::std::is_same_v<A, float> && ::cuda::std::is_same_v<B, float>,
                  "proxy reference was not unwrapped");
    return a + b;
  }
};

// a non-const lvalue reference cannot bind to a proxy at all
struct probe_mutate
{
  _CCCL_HOST_DEVICE void operator()(float& x) const
  {
    x += 1.0f;
  }
};

TEST_CASE("SequentialFindIfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4, 5};
  const auto result = thrust::find_if(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{});
  CHECK(result - vec.begin() == 2);
}

TEST_CASE("SequentialForEachProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3};
  thrust::for_each(thrust::seq, vec.begin(), vec.end(), probe_mutate{});
  CHECK(vec == proxy_vector{2, 3, 4});
}

TEST_CASE("SequentialForEachNProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3};
  const auto result = thrust::for_each_n(thrust::seq, vec.begin(), 3, probe_mutate{});
  CHECK(result == vec.end());
  CHECK(vec == proxy_vector{2, 3, 4});
}

TEST_CASE("SequentialCountIfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4, 5};
  CHECK(thrust::count_if(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{}) == 3);
}

TEST_CASE("SequentialAllOfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{3, 4, 5};
  CHECK(thrust::all_of(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{}));
}

TEST_CASE("SequentialAnyOfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3};
  CHECK(thrust::any_of(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{}));
}

TEST_CASE("SequentialNoneOfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2};
  CHECK(thrust::none_of(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{}));
}

TEST_CASE("SequentialMinElementProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{3, 1, 2, 5, 4};
  const auto result = thrust::min_element(thrust::seq, vec.begin(), vec.end(), probe_less{});
  CHECK(result - vec.begin() == 1);
}

TEST_CASE("SequentialMaxElementProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{3, 1, 2, 5, 4};
  const auto result = thrust::max_element(thrust::seq, vec.begin(), vec.end(), probe_less{});
  CHECK(result - vec.begin() == 3);
}

TEST_CASE("SequentialMinmaxElementProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{3, 1, 2, 5, 4};
  const auto result = thrust::minmax_element(thrust::seq, vec.begin(), vec.end(), probe_less{});
  CHECK(result.first - vec.begin() == 1);
  CHECK(result.second - vec.begin() == 3);
}

TEST_CASE("SequentialLowerBoundProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4, 5};
  const auto result = thrust::lower_bound(thrust::seq, vec.begin(), vec.end(), 3.0f, probe_less{});
  CHECK(result - vec.begin() == 2);
}

TEST_CASE("SequentialUpperBoundProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4, 5};
  const auto result = thrust::upper_bound(thrust::seq, vec.begin(), vec.end(), 3.0f, probe_less{});
  CHECK(result - vec.begin() == 3);
}

TEST_CASE("SequentialBinarySearchProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4, 5};
  CHECK(thrust::binary_search(thrust::seq, vec.begin(), vec.end(), 3.0f, probe_less{}));
  CHECK(!thrust::binary_search(thrust::seq, vec.begin(), vec.end(), 3.5f, probe_less{}));
}

TEST_CASE("SequentialCopyIfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4, 5};
  proxy_vector output(5);
  const auto new_end = thrust::copy_if(thrust::seq, input.begin(), input.end(), output.begin(), probe_unary_pred{});
  CHECK(new_end - output.begin() == 3);
  CHECK(proxy_vector(output.begin(), output.begin() + 3) == proxy_vector{3, 4, 5});
}

TEST_CASE("SequentialCopyIfStencilProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{10, 20, 30, 40, 50};
  proxy_vector stencil{1, 2, 3, 4, 5};
  proxy_vector output(5);
  const auto new_end =
    thrust::copy_if(thrust::seq, input.begin(), input.end(), stencil.begin(), output.begin(), probe_unary_pred{});
  CHECK(new_end - output.begin() == 3);
  CHECK(proxy_vector(output.begin(), output.begin() + 3) == proxy_vector{30, 40, 50});
}

TEST_CASE("SequentialRemoveIfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4, 5};
  const auto new_end = thrust::remove_if(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{});
  CHECK(new_end - vec.begin() == 2);
  CHECK(proxy_vector(vec.begin(), vec.begin() + 2) == proxy_vector{1, 2});
}

TEST_CASE("SequentialRemoveIfStencilProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{10, 20, 30, 40, 50};
  proxy_vector stencil{1, 2, 3, 4, 5};
  const auto new_end = thrust::remove_if(thrust::seq, vec.begin(), vec.end(), stencil.begin(), probe_unary_pred{});
  CHECK(new_end - vec.begin() == 2);
  CHECK(proxy_vector(vec.begin(), vec.begin() + 2) == proxy_vector{10, 20});
}

TEST_CASE("SequentialRemoveCopyIfProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4, 5};
  proxy_vector output(5);
  const auto new_end =
    thrust::remove_copy_if(thrust::seq, input.begin(), input.end(), output.begin(), probe_unary_pred{});
  CHECK(new_end - output.begin() == 2);
  CHECK(proxy_vector(output.begin(), output.begin() + 2) == proxy_vector{1, 2});
}

TEST_CASE("SequentialRemoveCopyIfStencilProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{10, 20, 30, 40, 50};
  proxy_vector stencil{1, 2, 3, 4, 5};
  proxy_vector output(5);
  const auto new_end = thrust::remove_copy_if(
    thrust::seq, input.begin(), input.end(), stencil.begin(), output.begin(), probe_unary_pred{});
  CHECK(new_end - output.begin() == 2);
  CHECK(proxy_vector(output.begin(), output.begin() + 2) == proxy_vector{10, 20});
}

TEST_CASE("SequentialUniqueProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 1, 2, 2, 3};
  const auto new_end = thrust::unique(thrust::seq, vec.begin(), vec.end(), probe_equal{});
  CHECK(new_end - vec.begin() == 3);
  CHECK(proxy_vector(vec.begin(), vec.begin() + 3) == proxy_vector{1, 2, 3});
}

TEST_CASE("SequentialUniqueCopyProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 1, 2, 2, 3};
  proxy_vector output(5);
  const auto new_end = thrust::unique_copy(thrust::seq, input.begin(), input.end(), output.begin(), probe_equal{});
  CHECK(new_end - output.begin() == 3);
  CHECK(proxy_vector(output.begin(), output.begin() + 3) == proxy_vector{1, 2, 3});
}

TEST_CASE("SequentialMergeProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector a{1, 3, 5};
  proxy_vector b{2, 4, 6};
  proxy_vector out(6);
  thrust::merge(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), probe_less{});
  CHECK(out == proxy_vector{1, 2, 3, 4, 5, 6});
}

TEST_CASE("SequentialSetDifferenceProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector a{1, 2, 3, 4, 5};
  proxy_vector b{2, 4};
  proxy_vector out(5);
  const auto new_end =
    thrust::set_difference(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), probe_less{});
  CHECK(new_end - out.begin() == 3);
  CHECK(proxy_vector(out.begin(), out.begin() + 3) == proxy_vector{1, 3, 5});
}

TEST_CASE("SequentialSetIntersectionProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector a{1, 2, 3, 4, 5};
  proxy_vector b{2, 4, 6};
  proxy_vector out(5);
  const auto new_end =
    thrust::set_intersection(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), probe_less{});
  CHECK(new_end - out.begin() == 2);
  CHECK(proxy_vector(out.begin(), out.begin() + 2) == proxy_vector{2, 4});
}

TEST_CASE("SequentialSetUnionProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector a{1, 3, 5};
  proxy_vector b{2, 3, 4};
  proxy_vector out(6);
  const auto new_end =
    thrust::set_union(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), probe_less{});
  CHECK(new_end - out.begin() == 5);
  CHECK(proxy_vector(out.begin(), out.begin() + 5) == proxy_vector{1, 2, 3, 4, 5});
}

TEST_CASE("SequentialSetSymmetricDifferenceProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector a{1, 2, 3, 4};
  proxy_vector b{3, 4, 5};
  proxy_vector out(7);
  const auto new_end =
    thrust::set_symmetric_difference(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), out.begin(), probe_less{});
  CHECK(new_end - out.begin() == 3);
  CHECK(proxy_vector(out.begin(), out.begin() + 3) == proxy_vector{1, 2, 5});
}

TEST_CASE("SequentialAdjacentDifferenceProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 4, 9, 16};
  proxy_vector output(4);
  const auto result =
    thrust::adjacent_difference(thrust::seq, input.begin(), input.end(), output.begin(), probe_binary_op{});
  CHECK(result == output.end());
  // the first element is copied, the rest is a[i] + a[i-1] because the probe op adds
  CHECK(output == proxy_vector{1, 5, 13, 25});
}

TEST_CASE("SequentialReduceProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 2, 3, 4};
  CHECK(thrust::reduce(thrust::seq, vec.begin(), vec.end(), 0.0f, probe_binary_op{}) == 10.0f);
}

TEST_CASE("SequentialInclusiveScanProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4};
  proxy_vector output(4);
  thrust::inclusive_scan(thrust::seq, input.begin(), input.end(), output.begin(), probe_binary_op{});
  CHECK(output == proxy_vector{1, 3, 6, 10});
}

TEST_CASE("SequentialInclusiveScanWithInitProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4};
  proxy_vector output(4);
  thrust::inclusive_scan(thrust::seq, input.begin(), input.end(), output.begin(), 10.0f, probe_binary_op{});
  CHECK(output == proxy_vector{11, 13, 16, 20});
}

TEST_CASE("SequentialExclusiveScanProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4};
  proxy_vector output(4);
  thrust::exclusive_scan(thrust::seq, input.begin(), input.end(), output.begin(), 0.0f, probe_binary_op{});
  CHECK(output == proxy_vector{0, 1, 3, 6});
}

TEST_CASE("SequentialInclusiveScanByKeyProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector keys{1, 1, 2, 2};
  proxy_vector values{1, 2, 3, 4};
  proxy_vector output(4);
  thrust::inclusive_scan_by_key(
    thrust::seq, keys.begin(), keys.end(), values.begin(), output.begin(), probe_equal{}, probe_binary_op{});
  CHECK(output == proxy_vector{1, 3, 3, 7});
}

TEST_CASE("SequentialExclusiveScanByKeyProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector keys{1, 1, 2, 2};
  proxy_vector values{1, 2, 3, 4};
  proxy_vector output(4);
  thrust::exclusive_scan_by_key(
    thrust::seq, keys.begin(), keys.end(), values.begin(), output.begin(), 0.0f, probe_equal{}, probe_binary_op{});
  CHECK(output == proxy_vector{0, 1, 0, 3});
}

TEST_CASE("SequentialPartitionProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 3, 2, 5, 4};
  const auto mid    = thrust::partition(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{});
  const auto n_true = mid - vec.begin();
  CHECK(n_true == 3);
  for (auto it = vec.begin(); it != mid; ++it)
  {
    CHECK(static_cast<float>(*it) > 2.0f);
  }
  for (auto it = mid; it != vec.end(); ++it)
  {
    CHECK(static_cast<float>(*it) <= 2.0f);
  }
}

TEST_CASE("SequentialStablePartitionProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{1, 3, 2, 5, 4};
  const auto mid = thrust::stable_partition(thrust::seq, vec.begin(), vec.end(), probe_unary_pred{});
  CHECK(mid - vec.begin() == 3);
  CHECK(vec == proxy_vector{3, 5, 4, 1, 2});
}

TEST_CASE("SequentialStablePartitionStencilProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{10, 20, 30, 40, 50};
  proxy_vector stencil{1, 3, 2, 5, 4};
  const auto mid = thrust::stable_partition(thrust::seq, vec.begin(), vec.end(), stencil.begin(), probe_unary_pred{});
  CHECK(mid - vec.begin() == 3);
  CHECK(vec == proxy_vector{20, 40, 50, 10, 30});
}

TEST_CASE("SequentialPartitionCopyProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4, 5};
  proxy_vector out_true(5);
  proxy_vector out_false(5);
  const auto result = thrust::partition_copy(
    thrust::seq, input.begin(), input.end(), out_true.begin(), out_false.begin(), probe_unary_pred{});
  CHECK(result.first - out_true.begin() == 3);
  CHECK(result.second - out_false.begin() == 2);
  CHECK(proxy_vector(out_true.begin(), out_true.begin() + 3) == proxy_vector{3, 4, 5});
  CHECK(proxy_vector(out_false.begin(), out_false.begin() + 2) == proxy_vector{1, 2});
}

TEST_CASE("SequentialStablePartitionCopyProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector input{1, 2, 3, 4, 5};
  proxy_vector out_true(5);
  proxy_vector out_false(5);
  const auto result = thrust::stable_partition_copy(
    thrust::seq, input.begin(), input.end(), out_true.begin(), out_false.begin(), probe_unary_pred{});
  CHECK(result.first - out_true.begin() == 3);
  CHECK(result.second - out_false.begin() == 2);
  CHECK(proxy_vector(out_true.begin(), out_true.begin() + 3) == proxy_vector{3, 4, 5});
  CHECK(proxy_vector(out_false.begin(), out_false.begin() + 2) == proxy_vector{1, 2});
}

TEST_CASE("SequentialStableSortProxyReference", "[sequential][proxy_reference]")
{
  proxy_vector vec{3, 1, 4, 1, 5, 9, 2, 6};
  thrust::stable_sort(thrust::seq, vec.begin(), vec.end(), probe_less{});
  CHECK(vec == proxy_vector{1, 1, 2, 3, 4, 5, 6, 9});
}
