#include <thrust/count.h>
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

struct TestUninitializedFillNonPOD
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
    thrust::uninitialized_fill(v, v + 1, exemplar);

    const auto n_device = thrust::count_if(v, v + 1, is_copy_constructed_on_device{});
    const auto n_host   = thrust::count_if(v, v + 1, is_copy_constructed_on_host{});
    if constexpr (device_system_is_cuda)
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
TEST_CASE("TestUninitializedFillNonPOD", "[uninitialized_fill]")
{
  const size_t s = GENERATE_THRUST_TEST_SIZES();
  TestUninitializedFillNonPOD{}(s);
}
