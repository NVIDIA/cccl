#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/uninitialized_fill.h>

#include <nv/target>

#include <unittest/unittest.h>

// This file mirrors uninitialized_fill.cu but covers thrust::uninitialized_fill_n. It is kept in
// a separate translation unit from uninitialized_fill.cu on purpose: nvcc 13.0 (GCC host only;
// fixed in 13.3+) hangs in cudafe++ --parse_templates when a NV_IF_TARGET-branched type's copy
// constructor is reached from both thrust::uninitialized_fill and thrust::uninitialized_fill_n in
// the same TU (see CopyConstructTest below). Keeping the two APIs' tests in separate files avoids
// the hang without any workaround needed in the test bodies themselves.

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

struct CopyConstructTest
{
  CopyConstructTest() = default;

  _CCCL_HOST_DEVICE CopyConstructTest(const CopyConstructTest&)
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (copy_constructed_on_device = true; copy_constructed_on_host = false;),
                 (copy_constructed_on_device = false; copy_constructed_on_host = true;));
  }

  CopyConstructTest& operator=(const CopyConstructTest&) = default;

  bool copy_constructed_on_host{false};
  bool copy_constructed_on_device{false};
};

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
    REQUIRE_FALSE(exemplar.copy_constructed_on_device);
    REQUIRE(exemplar.copy_constructed_on_host);

    // copy construct v from the exemplar
    thrust::uninitialized_fill_n(v, 1, exemplar);

    T x;
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);

    x = v[0];
    REQUIRE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);

    thrust::device_free(v);
  }
};
DECLARE_UNITTEST(TestUninitializedFillNNonPOD);
