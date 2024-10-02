#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/uninitialized_fill.h>

#include <nv/target>
#include <unittest/unittest.h>

template <typename ForwardIterator, typename T>
void uninitialized_fill(my_system& system, ForwardIterator, ForwardIterator, const T&)
{
  system.validate_dispatch();
}

void TestUninitializedFillDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::uninitialized_fill(sys, vec.begin(), vec.begin(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
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

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestUninitializedFillDispatchImplicit);

template <typename ForwardIterator, typename Size, typename T>
ForwardIterator uninitialized_fill_n(my_system& system, ForwardIterator first, Size, const T&)
{
  system.validate_dispatch();
  return first;
}

void TestUninitializedFillNDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::uninitialized_fill_n(sys, vec.begin(), vec.size(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
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

  my_system sys(0);
  thrust::uninitialized_fill_n(sys, vec.begin(), vec.size(), 0);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedFillNDispatchImplicit);

template <class Vector>
void TestUninitializedFillPOD()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  T exemplar(7);

  thrust::uninitialized_fill(v.begin() + 1, v.begin() + 4, exemplar);

  Vector ref{0, exemplar, exemplar, exemplar, 4};
  ASSERT_EQUAL(v, ref);

  exemplar = 8;

  thrust::uninitialized_fill(v.begin() + 0, v.begin() + 3, exemplar);

  ref = {exemplar, exemplar, exemplar, 7, 4};
  ASSERT_EQUAL(v, ref);

  exemplar = 9;

  thrust::uninitialized_fill(v.begin() + 2, v.end(), exemplar);

  ref = {8, 8, exemplar, exemplar, 9};
  ASSERT_EQUAL(v, ref);

  exemplar = 1;

  thrust::uninitialized_fill(v.begin(), v.end(), exemplar);

  ref = {exemplar, exemplar, exemplar, exemplar, exemplar};
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedFillPOD);

struct CopyConstructTest
{
  _CCCL_HOST_DEVICE CopyConstructTest()
      : copy_constructed_on_host(false)
      , copy_constructed_on_device(false)
  {}

  _CCCL_HOST_DEVICE CopyConstructTest(const CopyConstructTest&)
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (copy_constructed_on_device = true; copy_constructed_on_host = false;),
                 (copy_constructed_on_device = false; copy_constructed_on_host = true;));
  }

  _CCCL_HOST_DEVICE CopyConstructTest& operator=(const CopyConstructTest& x)
  {
    copy_constructed_on_host   = x.copy_constructed_on_host;
    copy_constructed_on_device = x.copy_constructed_on_device;
    return *this;
  }

  bool copy_constructed_on_host;
  bool copy_constructed_on_device;
};

struct TestUninitializedFillNonPOD
{
  void operator()(const size_t)
  {
    using T                 = CopyConstructTest;
    thrust::device_ptr<T> v = thrust::device_malloc<T>(5);

    T exemplar;
    ASSERT_EQUAL(false, exemplar.copy_constructed_on_device);
    ASSERT_EQUAL(false, exemplar.copy_constructed_on_host);

    T host_copy_of_exemplar(exemplar);
    ASSERT_EQUAL(false, exemplar.copy_constructed_on_device);
    ASSERT_EQUAL(true, exemplar.copy_constructed_on_host);

    // copy construct v from the exemplar
    thrust::uninitialized_fill(v, v + 1, exemplar);

    T x;
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    x = v[0];
    ASSERT_EQUAL(true, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    thrust::device_free(v);
  }
};
DECLARE_UNITTEST(TestUninitializedFillNonPOD);

template <class Vector>
void TestUninitializedFillNPOD()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  T exemplar(7);

  typename Vector::iterator iter = thrust::uninitialized_fill_n(v.begin() + 1, 3, exemplar);

  Vector ref{0, exemplar, exemplar, exemplar, 4};
  ASSERT_EQUAL_QUIET(v.begin() + 4, iter);

  exemplar = 8;

  iter = thrust::uninitialized_fill_n(v.begin() + 0, 3, exemplar);

  ref = {exemplar, exemplar, exemplar, 7, 4};
  ASSERT_EQUAL_QUIET(v.begin() + 3, iter);

  exemplar = 9;

  iter = thrust::uninitialized_fill_n(v.begin() + 2, 3, exemplar);

  ref = {8, 8, exemplar, exemplar, 9};
  ASSERT_EQUAL_QUIET(v.end(), iter);

  exemplar = 1;

  iter = thrust::uninitialized_fill_n(v.begin(), v.size(), exemplar);

  ref = {exemplar, exemplar, exemplar, exemplar, exemplar};
  ASSERT_EQUAL_QUIET(v.end(), iter);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedFillNPOD);

struct TestUninitializedFillNNonPOD
{
  void operator()(const size_t)
  {
    using T                 = CopyConstructTest;
    thrust::device_ptr<T> v = thrust::device_malloc<T>(5);

    T exemplar;
    ASSERT_EQUAL(false, exemplar.copy_constructed_on_device);
    ASSERT_EQUAL(false, exemplar.copy_constructed_on_host);

    T host_copy_of_exemplar(exemplar);
    ASSERT_EQUAL(false, exemplar.copy_constructed_on_device);
    ASSERT_EQUAL(true, exemplar.copy_constructed_on_host);

    // copy construct v from the exemplar
    thrust::uninitialized_fill_n(v, 1, exemplar);

    T x;
    ASSERT_EQUAL(false, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    x = v[0];
    ASSERT_EQUAL(true, x.copy_constructed_on_device);
    ASSERT_EQUAL(false, x.copy_constructed_on_host);

    thrust::device_free(v);
  }
};
DECLARE_UNITTEST(TestUninitializedFillNNonPOD);
