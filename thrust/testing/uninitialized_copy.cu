#include <thrust/count.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/uninitialized_copy.h>

#include "copy_construct_test.h"
#include <unittest/unittest.h>

template <typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_system& system, InputIterator, InputIterator, ForwardIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("TestUninitializedCopyDispatchExplicit", "[uninitialized_copy]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_copy(sys, vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_tag, InputIterator, InputIterator, ForwardIterator result)
{
  *result = 13;
  return result;
}

TEST_CASE("TestUninitializedCopyDispatchImplicit", "[uninitialized_copy]")
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_system& system, InputIterator, Size, ForwardIterator result)
{
  system.validate_dispatch();
  return result;
}

TEST_CASE("TestUninitializedCopyNDispatchExplicit", "[uninitialized_copy]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_copy_n(sys, vec.begin(), vec.size(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_tag, InputIterator, Size, ForwardIterator result)
{
  *result = 13;
  return result;
}

TEST_CASE("TestUninitializedCopyNDispatchImplicit", "[uninitialized_copy]")
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_copy_n(thrust::retag<my_tag>(vec.begin()), vec.size(), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <class Vector>
void test_uninitialized_copy_simple_pod()
{
  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);
  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());
  Vector ref{0, 1, 2, 3, 4};
  REQUIRE(v2 == ref);
}
DECLARE_VECTOR_UNITTEST(test_uninitialized_copy_simple_pod);

template <typename Vector>
void test_uninitialized_copy_n_simple_pod()
{
  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);
  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());
  Vector ref{0, 1, 2, 3, 4};
  REQUIRE(v2 == ref);
}
DECLARE_VECTOR_UNITTEST(test_uninitialized_copy_n_simple_pod);

TEST_CASE("TestUninitializedCopyNonPODDevice", "[uninitialized_copy]")
{
  using T = CopyConstructTest;

  thrust::device_vector<T> v1(5), v2(5);

  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

  const size_t n_device = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_device{});
  const size_t n_host   = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_host{});
  if constexpr (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA)
  {
    REQUIRE(n_device == v2.size());
    REQUIRE(n_host == 0u);
  }
  else
  {
    REQUIRE(n_device == 0u);
    REQUIRE(n_host == v2.size());
  }
}

TEST_CASE("TestUninitializedCopyNNonPODDevice", "[uninitialized_copy]")
{
  using T = CopyConstructTest;

  thrust::device_vector<T> v1(5), v2(5);

  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

  const size_t n_device = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_device{});
  const size_t n_host   = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_host{});
  if constexpr (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA)
  {
    REQUIRE(n_device == v2.size());
    REQUIRE(n_host == 0u);
  }
  else
  {
    REQUIRE(n_device == 0u);
    REQUIRE(n_host == v2.size());
  }
}

TEST_CASE("TestUninitializedCopyNonPODHost", "[uninitialized_copy]")
{
  using T = CopyConstructTest;

  thrust::host_vector<T> v1(5), v2(5);

  T x;
  REQUIRE_FALSE(x.copy_constructed_on_device);
  REQUIRE_FALSE(x.copy_constructed_on_host);

  x = v1[0];
  REQUIRE_FALSE(x.copy_constructed_on_device);
  REQUIRE_FALSE(x.copy_constructed_on_host);

  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

  const size_t n_device = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_device{});
  const size_t n_host   = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_host{});
  REQUIRE(n_device == 0u);
  REQUIRE(n_host == v2.size());
}

TEST_CASE("TestUninitializedCopyNNonPODHost", "[uninitialized_copy]")
{
  using T = CopyConstructTest;

  thrust::host_vector<T> v1(5), v2(5);

  T x;
  REQUIRE_FALSE(x.copy_constructed_on_device);
  REQUIRE_FALSE(x.copy_constructed_on_host);

  x = v1[0];
  REQUIRE_FALSE(x.copy_constructed_on_device);
  REQUIRE_FALSE(x.copy_constructed_on_host);

  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

  const size_t n_device = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_device{});
  const size_t n_host   = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_host{});
  REQUIRE(n_device == 0u);
  REQUIRE(n_host == v2.size());
}
