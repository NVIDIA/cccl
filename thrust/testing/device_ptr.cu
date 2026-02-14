#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cuda/std/iterator>

#include <iterator>

#include <unittest/unittest.h>

#ifdef __cpp_lib_concepts
static_assert(std::indirectly_writable<thrust::device_ptr<uint8_t>, uint8_t>);
#endif // __cpp_lib_concepts
static_assert(cuda::std::indirectly_writable<thrust::device_ptr<uint8_t>, uint8_t>);

void TestDevicePointerManipulation()
{
  thrust::device_vector<int> data(5);

  thrust::device_ptr<int> begin(&data[0]);
  thrust::device_ptr<int> end(&data[0] + 5);

  ASSERT_EQUAL(end - begin, 5);

  begin++;
  begin--;

  ASSERT_EQUAL(end - begin, 5);

  begin += 1;
  begin -= 1;

  ASSERT_EQUAL(end - begin, 5);

  begin = begin + (int) 1;
  begin = begin - (int) 1;

  ASSERT_EQUAL(end - begin, 5);

  begin = begin + (unsigned int) 1;
  begin = begin - (unsigned int) 1;

  ASSERT_EQUAL(end - begin, 5);

  begin = begin + (size_t) 1;
  begin = begin - (size_t) 1;

  ASSERT_EQUAL(end - begin, 5);

  begin = begin + (ptrdiff_t) 1;
  begin = begin - (ptrdiff_t) 1;

  ASSERT_EQUAL(end - begin, 5);

  begin = begin + (thrust::device_ptr<int>::difference_type) 1;
  begin = begin - (thrust::device_ptr<int>::difference_type) 1;

  ASSERT_EQUAL(end - begin, 5);
}
DECLARE_UNITTEST(TestDevicePointerManipulation);

void TestMakeDevicePointer()
{
  using T = int;

  T* raw_ptr = 0;

  thrust::device_ptr<T> p0 = thrust::device_pointer_cast(raw_ptr);

  ASSERT_EQUAL(thrust::raw_pointer_cast(p0), raw_ptr);

  thrust::device_ptr<T> p1 = thrust::device_pointer_cast(p0);

  ASSERT_EQUAL(p0, p1);
}
DECLARE_UNITTEST(TestMakeDevicePointer);

template <typename Vector>
void TestRawPointerCast()
{
  using T = typename Vector::value_type;

  Vector vec(3);

  T* first;
  T* last;

  first = thrust::raw_pointer_cast(&vec[0]);
  last  = thrust::raw_pointer_cast(&vec[3]);
  ASSERT_EQUAL(last - first, 3);

  first = thrust::raw_pointer_cast(&vec.front());
  last  = thrust::raw_pointer_cast(&vec.back());
  ASSERT_EQUAL(last - first, 2);

  // Do we want these to work?
  // first = thrust::raw_pointer_cast(vec.begin());
  // last  = thrust::raw_pointer_cast(vec.end());
  // ASSERT_EQUAL(last - first, 3);
}
DECLARE_VECTOR_UNITTEST(TestRawPointerCast);

template <typename T>
void TestDevicePointerNullptrCompatibility()
{
  thrust::device_ptr<T> p0(nullptr);

  ASSERT_EQUAL_QUIET(nullptr, p0);
  ASSERT_EQUAL_QUIET(p0, nullptr);

  p0 = nullptr;

  ASSERT_EQUAL_QUIET(nullptr, p0);
  ASSERT_EQUAL_QUIET(p0, nullptr);
}
DECLARE_GENERIC_UNITTEST(TestDevicePointerNullptrCompatibility);

template <typename T>
void TestDevicePointerBoolConversion()
{
  thrust::device_ptr<T> p0(nullptr);
  auto const b = bool(p0);

  ASSERT_EQUAL_QUIET(false, b);
}
DECLARE_GENERIC_UNITTEST(TestDevicePointerBoolConversion);

void TestDevicePointerCompare()
{
  using T1 = int;

  thrust::device_vector<T1> v1 = {42, 1337};

  { // test same element type
    using device_ptr = thrust::device_ptr<T1>;

    device_ptr ptr1 = v1.data();
    device_ptr ptr2 = ptr1 + 1;

    // Equality
    ASSERT_EQUAL(true, (ptr1 == ptr1));
    ASSERT_EQUAL(false, (ptr1 != ptr1));

    ASSERT_EQUAL(false, (ptr1 == ptr2));
    ASSERT_EQUAL(true, (ptr1 != ptr2));

    // Relations
    ASSERT_EQUAL(true, (ptr1 < ptr2));
    ASSERT_EQUAL(true, (ptr1 <= ptr2));
    ASSERT_EQUAL(true, (ptr2 > ptr1));
    ASSERT_EQUAL(true, (ptr2 >= ptr1));

    ASSERT_EQUAL(false, (ptr2 < ptr1));
    ASSERT_EQUAL(false, (ptr2 <= ptr1));
    ASSERT_EQUAL(false, (ptr1 > ptr2));
    ASSERT_EQUAL(false, (ptr1 >= ptr2));
  }

  using T2 = float;
  { // Ensure different element types are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr  = thrust::device_ptr<T2>;

    static_assert(thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }

  { // test different pointer types
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T1, thrust::device_system_tag, thrust::tagged_reference<T1, thrust::device_system_tag>>;
    static_assert(!::cuda::std::is_same_v<device_ptr, other_ptr>);

    device_ptr ptr1 = v1.data();
    other_ptr ptr2{other_ptr{thrust::raw_pointer_cast(ptr1 + 1)}};

    // Equality
    ASSERT_EQUAL(true, (ptr1 == ptr1));
    ASSERT_EQUAL(false, (ptr1 != ptr1));

    ASSERT_EQUAL(false, (ptr1 == ptr2));
    ASSERT_EQUAL(true, (ptr1 != ptr2));

    // Relations
    ASSERT_EQUAL(true, (ptr1 < ptr2));
    ASSERT_EQUAL(true, (ptr1 <= ptr2));
    ASSERT_EQUAL(true, (ptr2 > ptr1));
    ASSERT_EQUAL(true, (ptr2 >= ptr1));

    ASSERT_EQUAL(false, (ptr2 < ptr1));
    ASSERT_EQUAL(false, (ptr2 <= ptr1));
    ASSERT_EQUAL(false, (ptr1 > ptr2));
    ASSERT_EQUAL(false, (ptr1 >= ptr2));
  }

  { // ensure that different pointer types with different element types are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T2, thrust::device_system_tag, thrust::tagged_reference<T1, thrust::device_system_tag>>;

    static_assert(thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }

  // For the non-cuda backends host_system_tag and device_system_tag are comparable
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  { // ensure that pointers with different tags are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T1, thrust::host_system_tag, thrust::tagged_reference<T1, thrust::host_system_tag>>;

    static_assert(!thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

  { // ensure that different pointers with different tags and types are not comparable
    using device_ptr = thrust::device_ptr<T1>;
    using other_ptr =
      thrust::pointer<T2, thrust::host_system_tag, thrust::tagged_reference<T1, thrust::host_system_tag>>;

    // For the non-cuda backends host_system_tag and device_system_tag are comparable
    static_assert(thrust::detail::is_pointer_system_convertible_v<device_ptr, other_ptr>
                  == (THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA));
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ptr, other_ptr>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ptr, other_ptr>);
  }
}
DECLARE_UNITTEST(TestDevicePointerCompare);
