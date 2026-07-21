#include <thrust/iterator/retag.h>
#include <thrust/mismatch.h>

#include <unittest/unittest.h>
template <class Vector>
void TestMismatchSimple()
{
  Vector a{1, 2, 3, 4};
  Vector b{1, 2, 4, 3};

  ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first - a.begin(), 2);
  ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 2);

  b[2] = 3;

  ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first - a.begin(), 3);
  ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 3);

  b[3] = 4;

  ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).first - a.begin(), 4);
  ASSERT_EQUAL(thrust::mismatch(a.begin(), a.end(), b.begin()).second - b.begin(), 4);
}
DECLARE_VECTOR_UNITTEST(TestMismatchSimple);

template <class Vector>
void TestMismatchBoundedSimple()
{
  using T = typename Vector::value_type;

  // Equal-length ranges with mismatch
  {
    Vector a{1, 2, 3, 4};
    Vector b{1, 2, 4, 3};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end());
    ASSERT_EQUAL(result.first - a.begin(), 2);
    ASSERT_EQUAL(result.second - b.begin(), 2);
  }

  // Equal-length ranges, no mismatch
  {
    Vector a{1, 2, 3};
    Vector b{1, 2, 3};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end());
    ASSERT_EQUAL(result.first - a.begin(), 3);
    ASSERT_EQUAL(result.second - b.begin(), 3);
  }

  // Range1 shorter: stops when range1 exhausted
  {
    Vector a{1, 2};
    Vector b{1, 2, 99};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end());
    ASSERT_EQUAL(result.first - a.begin(), 2); // exhausted range1
    ASSERT_EQUAL(result.second - b.begin(), 2);
  }

  // Range2 shorter: stops when range2 exhausted
  {
    Vector a{1, 2, 99};
    Vector b{1, 2};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end());
    ASSERT_EQUAL(result.first - a.begin(), 2);
    ASSERT_EQUAL(result.second - b.begin(), 2); // exhausted range2
  }

  // Mismatch before either range ends (range2 shorter)
  {
    Vector a{1, 9, 3};
    Vector b{1, 2};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end());
    ASSERT_EQUAL(result.first - a.begin(), 1);
    ASSERT_EQUAL(result.second - b.begin(), 1);
  }

  // With binary predicate
  {
    Vector a{1, 2, 3};
    Vector b{1, 2};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end(), ::cuda::std::equal_to<T>());
    ASSERT_EQUAL(result.first - a.begin(), 2);
    ASSERT_EQUAL(result.second - b.begin(), 2);
  }

  // Empty ranges
  {
    Vector a;
    Vector b{1, 2};
    auto result = thrust::mismatch(a.begin(), a.end(), b.begin(), b.end());
    ASSERT_EQUAL(result.first - a.begin(), 0);
    ASSERT_EQUAL(result.second - b.begin(), 0);
  }
}
DECLARE_VECTOR_UNITTEST(TestMismatchBoundedSimple);

void TestMismatchBoundedWithExec()
{
  thrust::device_vector<int> a{1, 2, 3, 4};
  thrust::device_vector<int> b{1, 2, 99};

  // range1 longer, range2 shorter — stops at range2 end
  auto result = thrust::mismatch(thrust::device, a.begin(), a.end(), b.begin(), b.end());
  ASSERT_EQUAL(result.first - a.begin(), 2);
  ASSERT_EQUAL(result.second - b.begin(), 2);

  // With predicate
  result = thrust::mismatch(thrust::device, a.begin(), a.end(), b.begin(), b.end(), ::cuda::std::equal_to<int>());
  ASSERT_EQUAL(result.first - a.begin(), 2);
  ASSERT_EQUAL(result.second - b.begin(), 2);
}
DECLARE_UNITTEST(TestMismatchBoundedWithExec);

template <typename InputIterator1, typename InputIterator2>
cuda::std::pair<InputIterator1, InputIterator2>
mismatch(my_system& system, InputIterator1 first, InputIterator1, InputIterator2)
{
  system.validate_dispatch();
  return cuda::std::make_pair(first, first);
}

void TestMismatchDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::mismatch(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestMismatchDispatchExplicit);

template <typename InputIterator1, typename InputIterator2>
cuda::std::pair<InputIterator1, InputIterator2> mismatch(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
  *first = 13;
  return cuda::std::make_pair(first, first);
}

void TestMismatchDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::mismatch(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestMismatchDispatchImplicit);
