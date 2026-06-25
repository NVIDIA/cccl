#include <thrust/device_malloc_allocator.h>
#include <thrust/erase.h>

#include <unittest/unittest.h>

template <class Vector>
struct TestVectorRangeEraseSingleElement
{
  void operator()(size_t)
  {
    Vector v{5};

    auto prev_size = v.size();
    auto new_size  = v.size() - 1ul;

    auto del = thrust::erase(v, 5);

    ASSERT_EQUAL(v, Vector{});
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1, 2, 3, 5, 5, 4, 5};

    auto prev_size = v.size();
    auto new_size  = v.size() - 3ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1, 2, 3, 4};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{0, 1, 2, 3};

    auto prev_size = v.size();
    auto new_size  = v.size() - 1ul;

    auto del = thrust::erase(v, 0);

    Vector v_comp{1, 2, 3};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{};

    auto prev_size = v.size();
    auto new_size  = prev_size;

    auto del = thrust::erase(v, 0);

    Vector v_comp{};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1, 2, 3, 4};

    auto prev_size = v.size();
    auto new_size  = prev_size;

    auto del = thrust::erase(v, 0);

    Vector v_comp{1, 2, 3, 4};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{5, 5, 5, 5};

    auto prev_size = v.size();
    auto new_size  = 0ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1, 2, 3, 5};

    auto prev_size = v.size();
    auto new_size  = prev_size - 1ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1, 2, 3};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{5, 5, 5, 1, 2, 3};

    auto prev_size = v.size();
    auto new_size  = prev_size - 3ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1, 2, 3};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1, 2, 3, 5, 5, 5};

    auto prev_size = v.size();
    auto new_size  = prev_size - 3ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1, 2, 3};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1, 2, 5, 5, 5, 3, 4};

    auto prev_size = v.size();
    auto new_size  = prev_size - 3ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1, 2, 3, 4};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{5, 1, 5, 2, 5, 3, 5, 4};

    auto prev_size = v.size();
    auto new_size  = prev_size - 4ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1, 2, 3, 4};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v{1};

    auto prev_size = v.size();
    auto new_size  = prev_size;

    auto del = thrust::erase(v, 5);

    Vector v_comp{1};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
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
    Vector v(10000, 5);

    auto prev_size = v.size();
    auto new_size  = 0ul;

    auto del = thrust::erase(v, 5);

    Vector v_comp{};
    ASSERT_EQUAL(v, v_comp);
    ASSERT_EQUAL(v.size(), new_size);
    ASSERT_EQUAL(del, prev_size - new_size);
  }
};
VectorUnitTest<TestVectorRangeEraseBigVector, NumericTypes, thrust::device_vector, thrust::device_malloc_allocator>
  TestVectorRangeEraseBigVectorDeviceInstance;
VectorUnitTest<TestVectorRangeEraseBigVector, NumericTypes, thrust::host_vector, std::allocator>
  TestVectorRangeEraseBigVectorHostInstance;
