#include <thrust/device_malloc_allocator.h>
#include <thrust/erase.h>

#include <unittest/unittest.h>

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

template <class Vector>
void verify_erase(Vector input, typename Vector::value_type value, Vector expected)
{
  const auto old_size = input.size();

  const auto erased = thrust::erase(input, value);

  ASSERT_EQUAL(erased, old_size - expected.size());
  ASSERT_EQUAL(input.size(), expected.size());
  ASSERT_EQUAL(input, expected);
}

template <class Vector, class Predicate>
void verify_erase(std::initializer_list<int> input, Predicate pred, std::initializer_list<int> expected)
{
  verify_erase<Vector>(make_vector<Vector>(input), pred, make_vector<Vector>(expected));
}

template <class Vector>
struct TestVectorRangeEraseSingleElement
{
  void operator()(size_t)
  {
    verify_erase<Vector>(Vector{5}, typename Vector::value_type{5}, Vector{});
  }
};
VectorUnitTest<TestVectorRangeEraseSingleElement, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseSingleElementDeviceInstance;
VectorUnitTest<TestVectorRangeEraseSingleElement, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseSingleElementHostInstance;

template <class Vector>
struct TestVectorRangeEraseMultipleElements
{
  void operator()(size_t)
  {
    verify_erase<Vector>({1, 2, 3, 5, 5, 4, 5}, typename Vector::value_type{5}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseMultipleElements, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseMultipleElementsDeviceInstance;
VectorUnitTest<TestVectorRangeEraseMultipleElements, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseMultipleElementsHostInstance;

template <class Vector>
struct TestVectorRangeEraseFirstElement
{
  void operator()(size_t)
  {
    verify_erase<Vector>({0, 1, 2, 3}, typename Vector::value_type{0}, {1, 2, 3});
  }
};
VectorUnitTest<TestVectorRangeEraseFirstElement, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseFirstElementDeviceInstance;
VectorUnitTest<TestVectorRangeEraseFirstElement, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseFirstElementHostInstance;

template <class Vector>
struct TestVectorRangeEraseEmptyVector
{
  void operator()(size_t)
  {
    verify_erase<Vector>({}, typename Vector::value_type{0}, {});
  }
};
VectorUnitTest<TestVectorRangeEraseEmptyVector, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseEmptyVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseEmptyVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseEmptyVectorHostInstance;

template <class Vector>
struct TestVectorRangeEraseElementMissing
{
  void operator()(size_t)
  {
    verify_erase<Vector>({1, 2, 3, 4}, typename Vector::value_type{0}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseElementMissing, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseElementMissingDeviceInstance;
VectorUnitTest<TestVectorRangeEraseElementMissing, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseElementMissingHostInstance;

template <class Vector>
struct TestVectorRangeEraseAllElements
{
  void operator()(size_t)
  {
    verify_erase<Vector>({5, 5, 5, 5}, typename Vector::value_type{5}, {});
  }
};
VectorUnitTest<TestVectorRangeEraseAllElements, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseAllElementsDeviceInstance;
VectorUnitTest<TestVectorRangeEraseAllElements, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseAllElementsHostInstance;

template <class Vector>
struct TestVectorRangeEraseLastElement
{
  void operator()(size_t)
  {
    verify_erase<Vector>({1, 2, 3, 5}, typename Vector::value_type{5}, {1, 2, 3});
  }
};
VectorUnitTest<TestVectorRangeEraseLastElement, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseLastElementDeviceInstance;
VectorUnitTest<TestVectorRangeEraseLastElement, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseLastElementHostInstance;

template <class Vector>
struct TestVectorRangeEraseConsecutiveElementsAtStart
{
  void operator()(size_t)
  {
    verify_erase<Vector>({5, 5, 5, 1, 2, 3}, typename Vector::value_type{5}, {1, 2, 3});
  }
};
VectorUnitTest<TestVectorRangeEraseConsecutiveElementsAtStart,
               NumericTypes,
               thrust::device_vector,
               thrust::device_malloc_allocator>
  TestVectorRangeEraseConsecutiveElementsAtStartDeviceInstance;
VectorUnitTest<TestVectorRangeEraseConsecutiveElementsAtStart, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseConsecutiveElementsAtStartHostInstance;

template <class Vector>
struct TestVectorRangeEraseConsecutiveElementsAtEnd
{
  void operator()(size_t)
  {
    verify_erase<Vector>({1, 2, 3, 5, 5, 5}, typename Vector::value_type{5}, {1, 2, 3});
  }
};
VectorUnitTest<TestVectorRangeEraseConsecutiveElementsAtEnd,
               NumericTypes,
               thrust::device_vector,
               thrust::device_malloc_allocator>
  TestVectorRangeEraseConsecutiveElementsAtEndDeviceInstance;
VectorUnitTest<TestVectorRangeEraseConsecutiveElementsAtEnd, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseConsecutiveElementsAtEndHostInstance;

template <class Vector>
struct TestVectorRangeEraseConsecutiveElementsAtMid
{
  void operator()(size_t)
  {
    verify_erase<Vector>({1, 2, 5, 5, 5, 3, 4}, typename Vector::value_type{5}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseConsecutiveElementsAtMid,
               NumericTypes,
               thrust::device_vector,
               thrust::device_malloc_allocator>
  TestVectorRangeEraseConsecutiveElementsAtMidDeviceInstance;
VectorUnitTest<TestVectorRangeEraseConsecutiveElementsAtMid, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseConsecutiveElementsAtMidHostInstance;

template <class Vector>
struct TestVectorRangeEraseAlternatingElements
{
  void operator()(size_t)
  {
    verify_erase<Vector>({5, 1, 5, 2, 5, 3, 5, 4}, typename Vector::value_type{5}, {1, 2, 3, 4});
  }
};
VectorUnitTest<TestVectorRangeEraseAlternatingElements,
               NumericTypes,
               thrust::device_vector,
               thrust::device_malloc_allocator>
  TestVectorRangeEraseAlternatingElementsDeviceInstance;
VectorUnitTest<TestVectorRangeEraseAlternatingElements, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseAlternatingElementsHostInstance;

template <class Vector>
struct TestVectorRangeEraseNoneFromSingleElementVector
{
  void operator()(size_t)
  {
    verify_erase<Vector>(Vector{1}, typename Vector::value_type{5}, Vector{1});
  }
};
VectorUnitTest<TestVectorRangeEraseNoneFromSingleElementVector,
               NumericTypes,
               thrust::device_vector,
               thrust::device_malloc_allocator>
  TestVectorRangeEraseNoneFromSingleElementVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseNoneFromSingleElementVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseNoneFromSingleElementVectorHostInstance;

template <class Vector>
struct TestVectorRangeEraseBigVector
{
  void operator()(size_t)
  {
    const typename Vector::size_type n{10000};

    verify_erase<Vector>(Vector(n, typename Vector::value_type{5}), typename Vector::value_type{5}, Vector{});
  }
};
VectorUnitTest<TestVectorRangeEraseBigVector, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseBigVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseBigVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseBigVectorHostInstance;
