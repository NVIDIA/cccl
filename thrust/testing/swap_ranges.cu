#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/retag.h>
#include <thrust/swap.h>
#include <thrust/system/cpp/memory.h>

#include <unittest/unittest.h>

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator2 swap_ranges(my_system& system, ForwardIterator1, ForwardIterator1, ForwardIterator2 first2)
{
  system.validate_dispatch();
  return first2;
}

TEST_CASE("TestSwapRangesDispatchExplicit", "[swap_ranges]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::swap_ranges(sys, vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator2 swap_ranges(my_tag, ForwardIterator1, ForwardIterator1, ForwardIterator2 first2)
{
  *first2 = 13;
  return first2;
}

TEST_CASE("TestSwapRangesDispatchImplicit", "[swap_ranges]")
{
  thrust::device_vector<int> vec(1);

  thrust::swap_ranges(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <class Vector>
void TestSwapRangesSimple()
{
  Vector v1{0, 1, 2, 3, 4};
  Vector v2{5, 6, 7, 8, 9};

  thrust::swap_ranges(v1.begin(), v1.end(), v2.begin());

  Vector ref1{5, 6, 7, 8, 9};
  REQUIRE(v1 == ref1);

  Vector ref2{0, 1, 2, 3, 4};
  REQUIRE(v2 == ref2);
}
DECLARE_VECTOR_UNITTEST(TestSwapRangesSimple);

template <typename T>
void TestSwapRanges(const size_t n)
{
  const thrust::host_vector<T> a1 = unittest::random_integers<T>(n);
  const thrust::host_vector<T> a2 = unittest::random_integers<T>(n);

  thrust::host_vector<T> h1   = a1;
  thrust::host_vector<T> h2   = a2;
  thrust::device_vector<T> d1 = a1;
  thrust::device_vector<T> d2 = a2;

  thrust::swap_ranges(h1.begin(), h1.end(), h2.begin());
  thrust::swap_ranges(d1.begin(), d1.end(), d2.begin());

  REQUIRE(h1 == a2);
  REQUIRE(d1 == a2);
  REQUIRE(h2 == a1);
  REQUIRE(d2 == a1);
}
DECLARE_VARIABLE_UNITTEST(TestSwapRanges);

#if (THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP)
TEST_CASE("TestSwapRangesForcedIterator", "[swap_ranges]")
{
  thrust::device_vector<int> A(3, 0);
  thrust::device_vector<int> B(3, 1);

  thrust::swap_ranges(thrust::retag<thrust::cpp::tag>(A.begin()),
                      thrust::retag<thrust::cpp::tag>(A.end()),
                      thrust::retag<thrust::cpp::tag>(B.begin()));

  REQUIRE(A[0] == 1);
  REQUIRE(A[1] == 1);
  REQUIRE(A[2] == 1);
  REQUIRE(B[0] == 0);
  REQUIRE(B[1] == 0);
  REQUIRE(B[2] == 0);
}
#endif

struct type_with_swap
{
  inline _CCCL_HOST_DEVICE type_with_swap()
      : m_x()
      , m_swapped(false)
  {}

  inline _CCCL_HOST_DEVICE type_with_swap(int x)
      : m_x(x)
      , m_swapped(false)
  {}

  inline _CCCL_HOST_DEVICE type_with_swap(int x, bool s)
      : m_x(x)
      , m_swapped(s)
  {}

  inline type_with_swap(const type_with_swap& other) = default;

  inline _CCCL_HOST_DEVICE bool operator==(const type_with_swap& other) const
  {
    return m_x == other.m_x && m_swapped == other.m_swapped;
  }

  type_with_swap& operator=(const type_with_swap&) = default;

  int m_x;
  bool m_swapped;
};

inline _CCCL_HOST_DEVICE void swap(type_with_swap& a, type_with_swap& b) noexcept
{
  using ::cuda::std::swap;
  swap(a.m_x, b.m_x);
  a.m_swapped = true;
  b.m_swapped = true;
}

TEST_CASE("TestSwapRangesUserSwap", "[swap_ranges]")
{
  thrust::host_vector<type_with_swap> h_A(3, type_with_swap(0));
  thrust::host_vector<type_with_swap> h_B(3, type_with_swap(1));

  thrust::device_vector<type_with_swap> d_A = h_A;
  thrust::device_vector<type_with_swap> d_B = h_B;

  // check that nothing is yet swapped
  type_with_swap ref = type_with_swap(0, false);

  REQUIRE((ref == h_A[0]));
  REQUIRE((ref == h_A[1]));
  REQUIRE((ref == h_A[2]));

  REQUIRE((ref == d_A[0]));
  REQUIRE((ref == d_A[1]));
  REQUIRE((ref == d_A[2]));

  ref = type_with_swap(1, false);

  REQUIRE((ref == h_B[0]));
  REQUIRE((ref == h_B[1]));
  REQUIRE((ref == h_B[2]));

  REQUIRE((ref == d_B[0]));
  REQUIRE((ref == d_B[1]));
  REQUIRE((ref == d_B[2]));

  // swap the ranges

  thrust::swap_ranges(h_A.begin(), h_A.end(), h_B.begin());
  thrust::swap_ranges(d_A.begin(), d_A.end(), d_B.begin());

  // check that things were swapped
  ref = type_with_swap(1, true);

  REQUIRE((ref == h_A[0]));
  REQUIRE((ref == h_A[1]));
  REQUIRE((ref == h_A[2]));

  REQUIRE((ref == d_A[0]));
  REQUIRE((ref == d_A[1]));
  REQUIRE((ref == d_A[2]));

  ref = type_with_swap(0, true);

  REQUIRE((ref == h_B[0]));
  REQUIRE((ref == h_B[1]));
  REQUIRE((ref == h_B[2]));

  REQUIRE((ref == d_B[0]));
  REQUIRE((ref == d_B[1]));
  REQUIRE((ref == d_B[2]));
}
