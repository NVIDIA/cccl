#include <thrust/device_malloc_allocator.h>
#include <thrust/erase.h>

#include <unittest/unittest.h>

struct IsFive
{
  template <class T>
  _CCCL_HOST_DEVICE bool operator()(const T& x) const
  {
    return x == 5;
  }
};

template <class Vector>
Vector make_vector(std::initializer_list<int> values)
{
  Vector v;
  for (int x : values)
  {
    v.push_back(static_cast<typename Vector::value_type>(x));
  }
  return v;
}

template <class Vector, class Predicate>
void verify_erase_if(Vector input, Predicate pred, const Vector& expected)
{
  const auto old_size = input.size();

  const auto erased = thrust::erase_if(input, pred);

  ASSERT_EQUAL(input, expected);
  ASSERT_EQUAL(erased, old_size - expected.size());
  ASSERT_EQUAL(input.size(), expected.size());
}

template <class Vector, class Predicate>
void verify_erase_if(std::initializer_list<int> input, Predicate pred, std::initializer_list<int> expected)
{
  verify_erase_if<Vector>(make_vector<Vector>(input), pred, make_vector<Vector>(expected));
}

template <class Vector>
struct TestVectorRangeEraseIfSingleElement
{
  void operator()(size_t)
  {
    verify_erase_if<Vector>({5}, IsFive{}, {});
  }
};
VectorUnitTest<TestVectorRangeEraseIfSingleElement, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfSingleElementDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfSingleElement, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfSingleElementHostInstance;

template <class Vector>
struct TestVectorRangeEraseIfMultipleElements
{
  void operator()(size_t)
  {
    verify_erase_if<Vector>({1, 2, 3, 5, 5, 4, 5}, IsFive{}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseIfMultipleElements, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfMultipleElementsDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfMultipleElements, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfMultipleElementsHostInstance;

template <class Vector>
struct TestVectorRangeEraseIfEmptyVector
{
  void operator()(size_t)
  {
    verify_erase_if<Vector>({}, IsFive{}, {});
  }
};
VectorUnitTest<TestVectorRangeEraseIfEmptyVector, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfEmptyVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfEmptyVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfEmptyVectorHostInstance;

template <class Vector>
struct TestVectorRangeEraseIfNoMatch
{
  void operator()(size_t)
  {
    verify_erase_if<Vector>({1, 2, 3, 4}, IsFive{}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseIfNoMatch, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfNoMatchDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfNoMatch, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfNoMatchHostInstance;

template <class Vector>
struct TestVectorRangeEraseIfAllMatch
{
  void operator()(size_t)
  {
    verify_erase_if<Vector>({5, 5, 5, 5}, IsFive{}, {});
  }
};
VectorUnitTest<TestVectorRangeEraseIfAllMatch, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfAllMatchDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfAllMatch, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfAllMatchHostInstance;

template <class Vector>
struct TestVectorRangeEraseIfAlternatingMatches
{
  void operator()(size_t)
  {
    verify_erase_if<Vector>({5, 1, 5, 2, 5, 3, 5, 4}, IsFive{}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseIfAlternatingMatches,
               NumericTypes,
               thrust::device_vector,
               thrust::device_malloc_allocator>
  TestVectorRangeEraseIfAlternatingMatchesDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfAlternatingMatches, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfAlternatingMatchesHostInstance;

template <class Vector>
struct TestVectorRangeEraseIfBigVector
{
  void operator()(size_t)
  {
    const typename Vector::size_type n{10000};

    verify_erase_if<Vector>(Vector(n, typename Vector::value_type{5}), IsFive{}, Vector{});
  }
};
VectorUnitTest<TestVectorRangeEraseIfBigVector, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfBigVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfBigVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfBigVectorHostInstance;
