#include <thrust/device_reference.h>
#include <thrust/device_vector.h>

#include <unittest/unittest.h>

void TestDeviceReferenceConstructorFromDeviceReference()
{
  using T = int;

  thrust::device_vector<T> v(1, 0);
  const thrust::device_reference<T> ref = v[0];

  // ref equals the object at v[0]
  REQUIRE(v[0] == ref);

  // the address of ref equals the address of v[0]
  REQUIRE(&v[0] == &ref);

  // modifying v[0] modifies ref
  v[0] = 13;
  REQUIRE(13 == ref);
  REQUIRE(v[0] == ref);

  // modifying ref modifies v[0]
  ref = 7;
  REQUIRE(7 == v[0]);
  REQUIRE(v[0] == ref);
}
TEST_CASE("TestDeviceReferenceConstructorFromDeviceReference", "[device_reference]")
{
  TestDeviceReferenceConstructorFromDeviceReference();
}

void TestDeviceReferenceConstructorFromDevicePointer()
{
  using T = int;

  thrust::device_vector<T> v(1, 0);
  const thrust::device_ptr<T> ptr = &v[0];
  const thrust::device_reference<T> ref(ptr);

  // ref equals the object pointed to by ptr
  REQUIRE(*ptr == ref);

  // the address of ref equals ptr
  REQUIRE(ptr == &ref);

  // modifying *ptr modifies ref
  *ptr = 13;
  REQUIRE(13 == ref);
  REQUIRE(v[0] == ref);

  // modifying ref modifies *ptr
  ref = 7;
  REQUIRE(7 == *ptr);
  REQUIRE(v[0] == ref);
}
TEST_CASE("TestDeviceReferenceConstructorFromDevicePointer", "[device_reference]")
{
  TestDeviceReferenceConstructorFromDevicePointer();
}

void TestDeviceReferenceAssignmentFromDeviceReference()
{
  // test same types
  using T0 = int;
  thrust::device_vector<T0> v0{0, 0};
  const thrust::device_reference<T0> ref0 = v0[0];
  const thrust::device_reference<T0> ref1 = v0[1];

  ref0 = 13;
  ref1 = ref0;

  // ref1 equals 13
  REQUIRE(13 == ref1);
  REQUIRE(ref0 == ref1);

  // test const references
  const thrust::device_reference<T0> cref0 = v0[0];
  const thrust::device_reference<T0> cref1 = v0[1];

  cref0 = 13;
  cref1 = cref0;

  // cref1 equals 13
  REQUIRE(13 == cref1);
  REQUIRE(cref0 == cref1);

  // mix const and non-const references
  ref0  = 12;
  cref0 = ref0;
  REQUIRE(12 == cref0);

  cref0 = 11;
  ref0  = cref0;
  REQUIRE(11 == cref0);

  // test different types
  using T1 = float;
  thrust::device_vector<T1> v1{0.0f};
  const thrust::device_reference<T1> ref2 = v1[0];

  ref2 = ref0;

  // ref2 equals 11.0f
  REQUIRE(11.0f == ref2);
  REQUIRE(ref0 == ref2);
}
TEST_CASE("TestDeviceReferenceAssignmentFromDeviceReference", "[device_reference]")
{
  TestDeviceReferenceAssignmentFromDeviceReference();
}

void TestDeviceReferenceManipulation()
{
  using T1 = int;

  thrust::device_vector<T1> v(1, 0);
  const thrust::device_ptr<T1> ptr = &v[0];
  thrust::device_reference<T1> ref(ptr);

  // reset
  ref = 0;

  // test prefix increment
  ++ref;
  REQUIRE(1 == ref);
  REQUIRE(1 == *ptr);
  REQUIRE(1 == v[0]);

  // reset
  ref = 0;

  // test postfix increment
  const T1 x1 = ref++;
  REQUIRE(0 == x1);
  REQUIRE(1 == ref);
  REQUIRE(1 == *ptr);
  REQUIRE(1 == v[0]);

  // reset
  ref = 0;

  // test addition-assignment
  ref += 5;
  REQUIRE(5 == ref);
  REQUIRE(5 == *ptr);
  REQUIRE(5 == v[0]);

  // reset
  ref = 0;

  // test prefix decrement
  --ref;
  REQUIRE(-1 == ref);
  REQUIRE(-1 == *ptr);
  REQUIRE(-1 == v[0]);

  // reset
  ref = 0;

  // test subtraction-assignment
  ref -= 5;
  REQUIRE(-5 == ref);
  REQUIRE(-5 == *ptr);
  REQUIRE(-5 == v[0]);

  // reset
  ref = 1;

  // test multiply-assignment
  ref *= 5;
  REQUIRE(5 == ref);
  REQUIRE(5 == *ptr);
  REQUIRE(5 == v[0]);

  // reset
  ref = 5;

  // test divide-assignment
  ref /= 5;
  REQUIRE(1 == ref);
  REQUIRE(1 == *ptr);
  REQUIRE(1 == v[0]);

  // reset
  ref = 5;

  // test modulus-assignment
  ref %= 5;
  REQUIRE(0 == ref);
  REQUIRE(0 == *ptr);
  REQUIRE(0 == v[0]);

  // reset
  ref = 1;

  // test left shift-assignment
  ref <<= 1;
  REQUIRE(2 == ref);
  REQUIRE(2 == *ptr);
  REQUIRE(2 == v[0]);

  // reset
  ref = 2;

  // test right shift-assignment
  ref >>= 1;
  REQUIRE(1 == ref);
  REQUIRE(1 == *ptr);
  REQUIRE(1 == v[0]);

  // reset
  ref = 0;

  // test OR-assignment
  ref |= 1;
  REQUIRE(1 == ref);
  REQUIRE(1 == *ptr);
  REQUIRE(1 == v[0]);

  // reset
  ref = 1;

  // test XOR-assignment
  ref ^= 1;
  REQUIRE(0 == ref);
  REQUIRE(0 == *ptr);
  REQUIRE(0 == v[0]);

  // test equality of const references
  const thrust::device_reference<const T1> ref1 = v[0];
  REQUIRE(ref1 == ref);
}
TEST_CASE("TestDeviceReferenceManipulation", "[device_reference]")
{
  TestDeviceReferenceManipulation();
}

void TestDeviceReferenceSwap()
{
  using T = int;

  thrust::device_vector<T> v(2);
  thrust::device_reference<T> ref1       = v.front();
  const thrust::device_reference<T> ref2 = v.back();

  ref1 = 7;
  ref2 = 13;

  // test ADL two-step swap
  using ::cuda::std::swap;
  swap(ref1, ref2);
  REQUIRE(13 == ref1);
  REQUIRE(7 == ref2);

  // test .swap()
  ref1.swap(ref2);
  REQUIRE(7 == ref1);
  REQUIRE(13 == ref2);
}
TEST_CASE("TestDeviceReferenceSwap", "[device_reference]")
{
  TestDeviceReferenceSwap();
}

void TestDeviceReferenceCompare()
{
  using T1 = int;

  thrust::device_vector<T1> v1 = {42, 1337};

  { // test same element type
    using device_ref = thrust::device_reference<T1>;

    const device_ref ref1 = v1.front();
    const device_ref ref2 = v1.back();

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  using T2                     = float;
  thrust::device_vector<T2> v2 = {42.0f, 1337.0f};
  { // test different element type
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::device_reference<T2>;

    const device_ref ref1 = v1.front();
    const other_ref ref2  = v2.back();

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  { // test different reference types
    using device_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T1, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!::cuda::std::is_same_v<device_ref, other_ref>);

    const device_ref ref1 = v1.front();
    const other_ref ref2{other_pointer{thrust::raw_pointer_cast(v1.data() + 1)}};

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  { // test different reference types with different element types
    using device_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T2, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!::cuda::std::is_same_v<device_ref, other_ref>);

    const device_ref ref1 = v1.front();
    const other_ref ref2{other_pointer{thrust::raw_pointer_cast(v2.data() + 1)}};

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  // For the non-cuda backends host_system_tag and device_system_tag are comparable
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  { // ensure that references with different tags are not comparable
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T1, thrust::host_system_tag>;
    static_assert(!::cuda::std::is_same_v<device_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename device_ref::pointer, typename other_ref::pointer>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ref, other_ref>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ref, other_ref>);
  }

  { // ensure that references with different tags and types are not comparable
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T2, thrust::host_system_tag>;
    static_assert(!::cuda::std::is_same_v<device_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename device_ref::pointer, typename other_ref::pointer>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ref, other_ref>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ref, other_ref>);
  }
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

  { // ensure that references with incomparable element type are not comparable
    using device_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<cuda::std::pair<T1, T2>, thrust::device_system_tag>;
    static_assert(!::cuda::std::is_same_v<device_ref, other_ref>);

    static_assert(
      thrust::detail::is_pointer_system_convertible_v<typename device_ref::pointer, typename other_ref::pointer>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<T1, cuda::std::pair<T1, T2>>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<device_ref, other_ref>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<device_ref, other_ref>);
  }
}
TEST_CASE("TestDeviceReferenceCompare", "[device_reference]")
{
  TestDeviceReferenceCompare();
}

void TestTaggedReferenceCompare()
{
  using T1 = int;

  thrust::device_vector<T1> v1 = {42, 1337};

  { // test same element type
    using tagged_ref = thrust::tagged_reference<T1, thrust::device_system_tag>;
    using tagged_ptr = typename tagged_ref::pointer;

    const tagged_ref ref1{tagged_ptr{thrust::raw_pointer_cast(v1.data())}};
    const tagged_ref ref2{tagged_ptr{thrust::raw_pointer_cast(v1.data() + 1)}};

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  using T2                     = float;
  thrust::device_vector<T2> v2 = {42.0f, 1337.0f};
  { // test different element type
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::device_reference<T2>;

    const tagged_ref ref1 = v1.front();
    const other_ref ref2  = v2.back();

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  { // test different reference types
    using tagged_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T1, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!::cuda::std::is_same_v<tagged_ref, other_ref>);

    const tagged_ref ref1 = v1.front();
    const other_ref ref2{other_pointer{thrust::raw_pointer_cast(v1.data() + 1)}};

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

  { // test different reference types with different element types
    using tagged_ref    = thrust::device_reference<T1>;
    using other_ref     = thrust::tagged_reference<T2, thrust::device_system_tag>;
    using other_pointer = typename other_ref::pointer;
    static_assert(!::cuda::std::is_same_v<tagged_ref, other_ref>);

    const tagged_ref ref1 = v1.front();
    const other_ref ref2{other_pointer{thrust::raw_pointer_cast(v2.data() + 1)}};

    // Equality
    REQUIRE(ref1 == ref1);
    REQUIRE_FALSE(ref1 != ref1);

    REQUIRE_FALSE(ref1 == ref2);
    REQUIRE(ref1 != ref2);

    // Relations
    REQUIRE(ref1 < ref2);
    REQUIRE(ref1 <= ref2);
    REQUIRE(ref2 > ref1);
    REQUIRE(ref2 >= ref1);

    REQUIRE_FALSE(ref2 < ref1);
    REQUIRE_FALSE(ref2 <= ref1);
    REQUIRE_FALSE(ref1 > ref2);
    REQUIRE_FALSE(ref1 >= ref2);
  }

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  { // ensure that references with different tags are not comparable
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T1, thrust::host_system_tag>;
    static_assert(!::cuda::std::is_same_v<tagged_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename tagged_ref::pointer, typename other_ref::pointer>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<tagged_ref, other_ref>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<tagged_ref, other_ref>);
  }

  { // ensure that references with different tags and types are not comparable
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<T2, thrust::host_system_tag>;
    static_assert(!::cuda::std::is_same_v<tagged_ref, other_ref>);

    static_assert(
      !thrust::detail::is_pointer_system_convertible_v<typename tagged_ref::pointer, typename other_ref::pointer>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<tagged_ref, other_ref>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<tagged_ref, other_ref>);
  }
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

  { // ensure that references with incomparable element type are not comparable
    using tagged_ref = thrust::device_reference<T1>;
    using other_ref  = thrust::tagged_reference<cuda::std::pair<T1, T2>, thrust::device_system_tag>;
    static_assert(!::cuda::std::is_same_v<tagged_ref, other_ref>);

    static_assert(
      thrust::detail::is_pointer_system_convertible_v<typename tagged_ref::pointer, typename other_ref::pointer>);
    static_assert(!::cuda::std::__is_cpp17_equality_comparable_v<tagged_ref, other_ref>);
    static_assert(!::cuda::std::__is_cpp17_less_than_comparable_v<tagged_ref, other_ref>);
  }
}
TEST_CASE("TestTaggedReferenceCompare", "[device_reference]")
{
  TestTaggedReferenceCompare();
}
