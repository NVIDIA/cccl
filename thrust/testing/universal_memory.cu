#include <thrust/allocate_unique.h>
#include <thrust/sequence.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/universal_vector.h>

#include <numeric>
#include <vector>

#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <unittest/unittest.h>

namespace
{
// The managed_memory_pointer class should be identified as a
// contiguous_iterator
static_assert(thrust::is_contiguous_iterator_v<thrust::universal_allocator<int>::pointer>);

template <typename T>
struct some_object
{
  some_object(T data)
      : m_data(data)
  {}

  void setter(T data)
  {
    m_data = data;
  }
  T getter() const
  {
    return m_data;
  }

private:
  T m_data;
};
} // namespace

template <typename T>
void test_universal_allocate_unique()
{
  // Simple test to ensure that pointers created with universal_memory_resource
  // can be dereferenced and used with STL code. This is necessary as some
  // STL implementations break when using fancy references that overload
  // operator&, so universal_memory_resource uses a special pointer type that
  // returns regular C++ references that can be safely used host-side.

  // These operations fail to compile with fancy references:
  auto raw = thrust::allocate_unique<T>(thrust::universal_allocator<T>{}, 42);
  auto obj = thrust::allocate_unique<some_object<T>>(thrust::universal_allocator<some_object<T>>{}, 42);

  static_assert(std::is_same_v<decltype(raw.get()), thrust::universal_ptr<T>>,
                "Unexpected pointer type returned from std::unique_ptr::get.");
  static_assert(std::is_same_v<decltype(obj.get()), thrust::universal_ptr<some_object<T>>>,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  REQUIRE(*raw == T(42));
  REQUIRE(*raw.get() == T(42));
  REQUIRE(obj->getter() == T(42));
  REQUIRE((*obj).getter() == T(42));
  REQUIRE(obj.get()->getter() == T(42));
  REQUIRE((*obj.get()).getter() == T(42));
}
DECLARE_GENERIC_UNITTEST(test_universal_allocate_unique);

template <typename T>
void test_universal_iteration_raw()
{
  auto array = thrust::allocate_unique_n<T>(thrust::universal_allocator<T>{}, 6, 42);

  static_assert(std::is_same_v<decltype(array.get()), thrust::universal_ptr<T>>,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    REQUIRE(*iter == T(42));
    REQUIRE(*iter.get() == T(42));
  }
}
DECLARE_GENERIC_UNITTEST(test_universal_iteration_raw);

template <typename T>
void test_universal_iteration_obj()
{
  auto array = thrust::allocate_unique_n<some_object<T>>(thrust::universal_allocator<some_object<T>>{}, 6, 42);

  static_assert(std::is_same_v<decltype(array.get()), thrust::universal_ptr<some_object<T>>>,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  for (auto iter = array.get(), end = array.get() + 6; iter < end; ++iter)
  {
    REQUIRE(iter->getter() == T(42));
    REQUIRE((*iter).getter() == T(42));
    REQUIRE(iter.get()->getter() == T(42));
    REQUIRE((*iter.get()).getter() == T(42));
  }
}
DECLARE_GENERIC_UNITTEST(test_universal_iteration_obj);

template <typename T>
void test_universal_raw_pointer_cast()
{
  auto obj = thrust::allocate_unique<T>(thrust::universal_allocator<T>{}, 42);

  static_assert(std::is_same_v<decltype(obj.get()), thrust::universal_ptr<T>>,
                "Unexpected pointer type returned from std::unique_ptr::get.");

  static_assert(std::is_same_v<decltype(thrust::raw_pointer_cast(obj.get())), T*>,
                "Unexpected pointer type returned from thrust::raw_pointer_cast.");

  static_assert(std::is_same_v<decltype(cuda::std::to_address(obj.get())), T*>,
                "Unexpected pointer type returned from cuda::std::to_address.");

  *thrust::raw_pointer_cast(obj.get()) = T(17);
  REQUIRE(*obj == T(17));

  *cuda::std::to_address(obj.get()) = T(42);
  REQUIRE(*obj == T(42));
}
DECLARE_GENERIC_UNITTEST(test_universal_raw_pointer_cast);

template <typename T>
void test_universal_thrust_vector(std::size_t const n)
{
  thrust::host_vector<T> host(n);
  thrust::universal_vector<T> universal(n);

  static_assert(std::is_same_v<typename std::decay_t<decltype(universal)>::pointer, thrust::universal_ptr<T>>,
                "Unexpected thrust::universal_vector pointer type.");

  thrust::sequence(host.begin(), host.end(), 0);
  thrust::sequence(universal.begin(), universal.end(), 0);
  REQUIRE(host.size() == n);
  REQUIRE(universal.size() == n);
  REQUIRE(host == universal);
}
DECLARE_VARIABLE_UNITTEST(test_universal_thrust_vector);

// TODO(bgruber): merge test into previous when we have Catch2
template <typename T>
void test_universal_host_pinned_thrust_vector(std::size_t const n)
{
  thrust::host_vector<T> host(n);
  thrust::universal_host_pinned_vector<T> universal(n);

  static_assert(
    std::is_same_v<typename std::decay_t<decltype(universal)>::pointer, thrust::universal_host_pinned_ptr<T>>);

  thrust::sequence(host.begin(), host.end(), 0);
  thrust::sequence(universal.begin(), universal.end(), 0);

  REQUIRE(host.size() == n);
  REQUIRE(universal.size() == n);
  REQUIRE(host == universal);
}
DECLARE_VARIABLE_UNITTEST(test_universal_host_pinned_thrust_vector);

// Verify that a std::vector using the universal allocator will work with
// Standard Library algorithms.
template <typename T>
void test_universal_std_vector(std::size_t const n)
{
  std::vector<T> host(n);
  std::vector<T, thrust::universal_allocator<T>> universal(n);

  static_assert(std::is_same_v<typename std::decay<decltype(universal)>::type::pointer, thrust::universal_ptr<T>>,
                "Unexpected std::vector pointer type.");

  std::iota(host.begin(), host.end(), 0);
  std::iota(universal.begin(), universal.end(), 0);

  REQUIRE(host.size() == n);
  REQUIRE(universal.size() == n);
  // host and universal have different allocator types, so std::vector::operator== does not apply
  REQUIRE_THAT(universal, Catch::Matchers::RangeEquals(host));
}
DECLARE_VARIABLE_UNITTEST(test_universal_std_vector);
