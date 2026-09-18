#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/retag.h>
#include <thrust/uninitialized_copy.h>

#include <nv/target>

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
void TestUninitializedCopySimplePOD()
{
  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);
  thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());
  Vector ref{0, 1, 2, 3, 4};
  REQUIRE(v2 == ref);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedCopySimplePOD);

template <typename Vector>
void TestUninitializedCopyNSimplePOD()
{
  Vector v1{0, 1, 2, 3, 4};

  // copy to Vector
  Vector v2(5);
  thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());
  Vector ref{0, 1, 2, 3, 4};
  REQUIRE(v2 == ref);
}
DECLARE_VECTOR_UNITTEST(TestUninitializedCopyNSimplePOD);

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

struct TestUninitializedCopyNonPODDevice
{
  void operator()(const size_t)
  {
    using T = CopyConstructTest;

    thrust::device_vector<T> v1(5), v2(5);

    T x;
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);

    x = v1[0];
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);

    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

    x = v2[0];
    REQUIRE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);
  }
};
TEST_CASE("TestUninitializedCopyNonPODDevice", "[uninitialized_copy]")
{
  TestUninitializedCopyNonPODDevice();
}

struct TestUninitializedCopyNNonPODDevice
{
  void operator()(const size_t)
  {
    using T = CopyConstructTest;

    thrust::device_vector<T> v1(5), v2(5);

    T x;
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);

    x = v1[0];
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);

    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

    x = v2[0];
    REQUIRE(x.copy_constructed_on_device);
    REQUIRE_FALSE(x.copy_constructed_on_host);
  }
};
TEST_CASE("TestUninitializedCopyNNonPODDevice", "[uninitialized_copy]")
{
  TestUninitializedCopyNNonPODDevice();
}

struct TestUninitializedCopyNonPODHost
{
  void operator()(const size_t)
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

    x = v2[0];
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE(x.copy_constructed_on_host);
  }
};
TEST_CASE("TestUninitializedCopyNonPODHost", "[uninitialized_copy]")
{
  TestUninitializedCopyNonPODHost();
}

struct TestUninitializedCopyNNonPODHost
{
  void operator()(const size_t)
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

    x = v2[0];
    REQUIRE_FALSE(x.copy_constructed_on_device);
    REQUIRE(x.copy_constructed_on_host);
  }
};
TEST_CASE("TestUninitializedCopyNNonPODHost", "[uninitialized_copy]")
{
  TestUninitializedCopyNNonPODHost();
}
