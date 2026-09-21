#include <thrust/count.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/uninitialized_fill.h>

#include "copy_construct_test.h"
#include <unittest/unittest.h>

template <typename ForwardIterator, typename T>
void uninitialized_fill(my_system& system, ForwardIterator, ForwardIterator, const T&)
{
  system.validate_dispatch();
}

void TestUninitializedFillDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_fill(sys, vec.begin(), vec.begin(), 0);

  REQUIRE(sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedFillDispatchExplicit);

template <typename ForwardIterator, typename T>
void uninitialized_fill(my_tag, ForwardIterator first, ForwardIterator, const T&)
{
  *first = 13;
}

void TestUninitializedFillDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_fill(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), 0);

  REQUIRE(13 == vec.front());
}
DECLARE_UNITTEST(TestUninitializedFillDispatchImplicit);

template <class Vector>
void TestUninitializedFillPOD()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  T exemplar(7);

  thrust::uninitialized_fill(v.begin() + 1, v.begin() + 4, exemplar);

  Vector ref{0, exemplar, exemplar, exemplar, 4};
  REQUIRE(v == ref);

  exemplar = 8;

  thrust::uninitialized_fill(v.begin() + 0, v.begin() + 3, exemplar);

  ref = {exemplar, exemplar, exemplar, 7, 4};
  REQUIRE(v == ref);

  exemplar = 9;

  thrust::uninitialized_fill(v.begin() + 2, v.end(), exemplar);

  ref = {8, 8, exemplar, exemplar, 9};
  REQUIRE(v == ref);

  exemplar = 1;

  thrust::uninitialized_fill(v.begin(), v.end(), exemplar);

  ref = {exemplar, exemplar, exemplar, exemplar, exemplar};
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedFillPOD);

TEST_CASE("TestUninitializedFillNonPOD", "[uninitialized_fill]")
{
  using T                       = CopyConstructTest;
  const thrust::device_ptr<T> v = thrust::device_malloc<T>(5);

  const T exemplar;
  REQUIRE_FALSE(exemplar.copy_constructed_on_device);
  REQUIRE_FALSE(exemplar.copy_constructed_on_host);

  const T host_copy_of_exemplar(exemplar); // NOLINT(performance-unnecessary-copy-initialization)
  REQUIRE_FALSE(host_copy_of_exemplar.copy_constructed_on_device);
  REQUIRE(host_copy_of_exemplar.copy_constructed_on_host);

  // copy construct v from the exemplar
  thrust::uninitialized_fill(v, v + 1, exemplar);

  const auto n_device = thrust::count_if(v, v + 1, is_copy_constructed_on_device{});
  const auto n_host   = thrust::count_if(v, v + 1, is_copy_constructed_on_host{});
  if constexpr (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA)
  {
    REQUIRE(n_device == 1);
    REQUIRE(n_host == 0);
  }
  else
  {
    REQUIRE(n_device == 0);
    REQUIRE(n_host == 1);
  }

  thrust::device_free(v);
}
