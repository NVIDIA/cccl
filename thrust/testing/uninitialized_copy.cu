#include <thrust/count.h>
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

void TestUninitializedCopyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_copy(sys, vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedCopyDispatchExplicit);

template <typename InputIterator, typename ForwardIterator>
ForwardIterator uninitialized_copy(my_tag, InputIterator, InputIterator, ForwardIterator result)
{
  *result = 13;
  return result;
}

void TestUninitializedCopyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_copy(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}
DECLARE_UNITTEST(TestUninitializedCopyDispatchImplicit);

template <typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_system& system, InputIterator, Size, ForwardIterator result)
{
  system.validate_dispatch();
  return result;
}

void TestUninitializedCopyNDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::uninitialized_copy_n(sys, vec.begin(), vec.size(), vec.begin());

  REQUIRE(sys.is_valid());
}
DECLARE_UNITTEST(TestUninitializedCopyNDispatchExplicit);

template <typename InputIterator, typename Size, typename ForwardIterator>
ForwardIterator uninitialized_copy_n(my_tag, InputIterator, Size, ForwardIterator result)
{
  *result = 13;
  return result;
}

void TestUninitializedCopyNDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::uninitialized_copy_n(thrust::retag<my_tag>(vec.begin()), vec.size(), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}
DECLARE_UNITTEST(TestUninitializedCopyNDispatchImplicit);

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

// Reading a CopyConstructTest back to the host (e.g. via `v[0]`) can itself invoke its copy
// constructor on the host, clobbering the very flags being observed. Avoid that by checking the
// flags in place with count_if: the predicate runs wherever the elements live (on the device for
// the CUDA backend), and only a plain size_t count crosses back to the host.
struct is_copy_constructed_on_device
{
  _CCCL_HOST_DEVICE bool operator()(const CopyConstructTest& t) const
  {
    return t.copy_constructed_on_device;
  }
};

struct is_copy_constructed_on_host
{
  _CCCL_HOST_DEVICE bool operator()(const CopyConstructTest& t) const
  {
    return t.copy_constructed_on_host;
  }
};

// Only the CUDA backend runs "device" work as actual device code; the OMP/TBB/CPP backends
// execute their device_system algorithms on the host, so CopyConstructTest's copy constructor
// always takes the host branch there.
inline constexpr bool device_system_is_cuda = THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA;

struct TestUninitializedCopyNonPODDevice
{
  void operator()(const size_t)
  {
    using T = CopyConstructTest;

    thrust::device_vector<T> v1(5), v2(5);

    thrust::uninitialized_copy(v1.begin(), v1.end(), v2.begin());

    const size_t n_device = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_device{});
    const size_t n_host   = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_host{});
    if constexpr (device_system_is_cuda)
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
};
TEST_CASE("TestUninitializedCopyNonPODDevice", "[uninitialized_copy]")
{
  const size_t s = GENERATE_THRUST_TEST_SIZES();
  TestUninitializedCopyNonPODDevice{}(s);
}

struct TestUninitializedCopyNNonPODDevice
{
  void operator()(const size_t)
  {
    using T = CopyConstructTest;

    thrust::device_vector<T> v1(5), v2(5);

    thrust::uninitialized_copy_n(v1.begin(), v1.size(), v2.begin());

    const size_t n_device = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_device{});
    const size_t n_host   = thrust::count_if(v2.begin(), v2.end(), is_copy_constructed_on_host{});
    if constexpr (device_system_is_cuda)
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
};
TEST_CASE("TestUninitializedCopyNNonPODDevice", "[uninitialized_copy]")
{
  const size_t s = GENERATE_THRUST_TEST_SIZES();
  TestUninitializedCopyNNonPODDevice{}(s);
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
  const size_t s = GENERATE_THRUST_TEST_SIZES();
  TestUninitializedCopyNonPODHost{}(s);
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
  const size_t s = GENERATE_THRUST_TEST_SIZES();
  TestUninitializedCopyNNonPODHost{}(s);
}
