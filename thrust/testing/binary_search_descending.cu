#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

//////////////////////
// Scalar Functions //
//////////////////////

template <class Vector>
void TestScalarLowerBoundDescendingSimple()
{
  using T = typename Vector::value_type;

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), T{0}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::lower_bound(vec.begin(), vec.end(), T{1}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{2}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{3}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::lower_bound(vec.begin(), vec.end(), T{4}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), T{5}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::lower_bound(vec.begin(), vec.end(), T{6}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::lower_bound(vec.begin(), vec.end(), T{7}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), T{8}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::lower_bound(vec.begin(), vec.end(), T{9}, ::cuda::std::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarLowerBoundDescendingSimple);

template <class Vector>
void TestScalarUpperBoundDescendingSimple()
{
  using T = typename Vector::value_type;

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQUAL_QUIET(vec.begin() + 5, thrust::upper_bound(vec.begin(), vec.end(), T{0}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), T{1}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 4, thrust::upper_bound(vec.begin(), vec.end(), T{2}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{3}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{4}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 3, thrust::upper_bound(vec.begin(), vec.end(), T{5}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), T{6}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 2, thrust::upper_bound(vec.begin(), vec.end(), T{7}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 1, thrust::upper_bound(vec.begin(), vec.end(), T{8}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL_QUIET(vec.begin() + 0, thrust::upper_bound(vec.begin(), vec.end(), T{9}, ::cuda::std::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarUpperBoundDescendingSimple);

template <class Vector>
void TestScalarBinarySearchDescendingSimple()
{
  using T = typename Vector::value_type;

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQUAL(true, thrust::binary_search(vec.begin(), vec.end(), T{0}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{1}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(true, thrust::binary_search(vec.begin(), vec.end(), T{2}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{3}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{4}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(true, thrust::binary_search(vec.begin(), vec.end(), T{5}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{6}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(true, thrust::binary_search(vec.begin(), vec.end(), T{7}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(true, thrust::binary_search(vec.begin(), vec.end(), T{8}, ::cuda::std::greater<T>()));
  ASSERT_EQUAL(false, thrust::binary_search(vec.begin(), vec.end(), T{9}, ::cuda::std::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestScalarBinarySearchDescendingSimple);

template <class Vector>
void TestScalarEqualRangeDescendingSimple()
{
  using T = typename Vector::value_type;

  Vector vec{8, 7, 5, 2, 0};

  ASSERT_EQUAL_QUIET(vec.begin() + 4,
                     thrust::equal_range(vec.begin(), vec.end(), T{0}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 4,
                     thrust::equal_range(vec.begin(), vec.end(), T{1}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 3,
                     thrust::equal_range(vec.begin(), vec.end(), T{2}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 3,
                     thrust::equal_range(vec.begin(), vec.end(), T{3}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 3,
                     thrust::equal_range(vec.begin(), vec.end(), T{4}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 2,
                     thrust::equal_range(vec.begin(), vec.end(), T{5}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 2,
                     thrust::equal_range(vec.begin(), vec.end(), T{6}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 1,
                     thrust::equal_range(vec.begin(), vec.end(), T{7}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 0,
                     thrust::equal_range(vec.begin(), vec.end(), T{8}, ::cuda::std::greater<T>()).first);
  ASSERT_EQUAL_QUIET(vec.begin() + 0,
                     thrust::equal_range(vec.begin(), vec.end(), T{9}, ::cuda::std::greater<T>()).first);

  ASSERT_EQUAL_QUIET(vec.begin() + 5,
                     thrust::equal_range(vec.begin(), vec.end(), T{0}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 4,
                     thrust::equal_range(vec.begin(), vec.end(), T{1}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 4,
                     thrust::equal_range(vec.begin(), vec.end(), T{2}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 3,
                     thrust::equal_range(vec.begin(), vec.end(), T{3}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 3,
                     thrust::equal_range(vec.begin(), vec.end(), T{4}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 3,
                     thrust::equal_range(vec.begin(), vec.end(), T{5}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 2,
                     thrust::equal_range(vec.begin(), vec.end(), T{6}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 2,
                     thrust::equal_range(vec.begin(), vec.end(), T{7}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 1,
                     thrust::equal_range(vec.begin(), vec.end(), T{8}, ::cuda::std::greater<T>()).second);
  ASSERT_EQUAL_QUIET(vec.begin() + 0,
                     thrust::equal_range(vec.begin(), vec.end(), T{9}, ::cuda::std::greater<T>()).second);
}
DECLARE_VECTOR_UNITTEST(TestScalarEqualRangeDescendingSimple);
