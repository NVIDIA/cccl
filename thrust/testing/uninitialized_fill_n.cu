#include <thrust/count.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/uninitialized_fill.h>

#include "copy_construct_test.h"
#include <unittest/unittest.h>

// This file mirrors uninitialized_fill.cu but covers thrust::uninitialized_fill_n. It is kept in
// a separate translation unit from uninitialized_fill.cu on purpose: nvcc 13.0 (GCC host only;
// fixed in 13.3+) hangs in cudafe++ --parse_templates when a NV_IF_TARGET-branched type's copy
// constructor (see CopyConstructTest in copy_construct_test.h) is reached from both
// thrust::uninitialized_fill and thrust::uninitialized_fill_n in the same TU. Keeping the two
// APIs' tests in separate files avoids the hang without any workaround needed in the test bodies
// themselves.

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(my_system& system, ForwardIterator first, Size, const T&)
{
  system.validate_dispatch();
  return first;
}

void TestUninitializedFillNDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_fill_n(sys, vec.begin(), vec.size(), 0);

  REQUIRE(sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedFillNDispatchExplicit);

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(my_tag, ForwardIterator first, Size, const T&)
{
  *first = 13;
  return first;
}

void TestUninitializedFillNDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_fill_n(sys, vec.begin(), vec.size(), 0);

  REQUIRE(sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedFillNDispatchImplicit);

template <class Vector>
void TestUninitializedFillNPOD()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  T exemplar(7);

  typename Vector::iterator iter = thrust::uninitialized_fill_n(v.begin() + 1, 3, exemplar);

  Vector ref{0, exemplar, exemplar, exemplar, 4};
  REQUIRE(v.begin() + 4 == iter);
  REQUIRE(v == ref);

  exemplar = 8;

  iter = thrust::uninitialized_fill_n(v.begin() + 0, 3, exemplar);

  ref = {exemplar, exemplar, exemplar, 7, 4};
  REQUIRE(v.begin() + 3 == iter);
  REQUIRE(v == ref);

  exemplar = 9;

  iter = thrust::uninitialized_fill_n(v.begin() + 2, 3, exemplar);

  ref = {8, 8, exemplar, exemplar, 9};
  REQUIRE(v.end() == iter);
  REQUIRE(v == ref);

  exemplar = 1;

  iter = thrust::uninitialized_fill_n(v.begin(), v.size(), exemplar);

  ref = {exemplar, exemplar, exemplar, exemplar, exemplar};
  REQUIRE(v.end() == iter);
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedFillNPOD);

struct TestUninitializedFillNNonPOD
{
  void operator()(const size_t)
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
    thrust::uninitialized_fill_n(v, 1, exemplar);

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
};
TEST_CASE("TestUninitializedFillNNonPOD", "[uninitialized_fill_n]")
{
  const size_t s = GENERATE_THRUST_TEST_SIZES();
  TestUninitializedFillNNonPOD{}(s);
}
