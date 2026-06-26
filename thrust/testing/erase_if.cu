#include <thrust/device_malloc_allocator.h>
#include <thrust/erase.h>

#include <unittest/unittest.h>

struct IsFive
{
  template <class T>
  bool operator()(const T& x) const
  {
    return x == 5;
  }
};

template <class Vector>
struct TestVectorRangeEraseIfSingleElement
{
  void operator()(size_t)
  {
    Vector v{5};

    auto prev_size = v.size();
    auto new_size  = 0ul;

    auto del = thrust::erase_if(v, IsFive{});

    ASSERT_EQUAL(v, Vector{});
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1, 2, 3, 5, 5, 4, 5};

    auto prev_size = v.size();
    auto new_size  = prev_size - 3ul;

    auto del = thrust::erase_if(v, IsFive{});

    Vector expected{1, 2, 3, 4};

    ASSERT_EQUAL(v, expected);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{};

    auto del = thrust::erase_if(v, IsFive{});

    ASSERT_EQUAL(v, Vector{});
    ASSERT_EQUAL(del, 0ul);
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
    Vector v{1, 2, 3, 4};

    auto prev_size = v.size();

    auto del = thrust::erase_if(v, IsFive{});

    Vector expected{1, 2, 3, 4};

    ASSERT_EQUAL(v, expected);
    ASSERT_EQUAL(v.size(), prev_size);
    ASSERT_EQUAL(del, 0ul);
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
    Vector v{5, 5, 5, 5};

    auto prev_size = v.size();

    auto del = thrust::erase_if(v, IsFive{});

    ASSERT_EQUAL(v, Vector{});
    ASSERT_EQUAL(v.size(), 0ul);
    ASSERT_EQUAL(del, prev_size);
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
    Vector v{5, 1, 5, 2, 5, 3, 5, 4};

    auto prev_size = v.size();

    auto del = thrust::erase_if(v, IsFive{});

    Vector expected{1, 2, 3, 4};

    ASSERT_EQUAL(v, expected);
    ASSERT_EQUAL(v.size(), 4ul);
    ASSERT_EQUAL(del, prev_size - 4ul);
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
    Vector v(10000, 5);

    auto del = thrust::erase_if(v, IsFive{});

    ASSERT_EQUAL(v, Vector{});
    ASSERT_EQUAL(del, 10000ul);
  }
};
VectorUnitTest<TestVectorRangeEraseIfBigVector, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseIfBigVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseIfBigVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseIfBigVectorHostInstance;
