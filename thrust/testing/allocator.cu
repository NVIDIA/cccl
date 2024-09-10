#include <thrust/detail/config.h>

#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cpp/vector.h>

#include <memory>

#include <nv/target>
#include <unittest/unittest.h>

// WAR NVIDIA/cccl#1731
// Some tests miscompile for non-CUDA backends on MSVC 2017 and 2019 (though 2022 is fine).
// This is due to a bug in the compiler that breaks __THRUST_DEFINE_HAS_MEMBER_FUNCTION.
#if defined(_MSC_VER) && _MSC_VER <= 1929 && THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#  define WAR_BUG_1731
#endif

// The needs_copy_construct_via_allocator trait depends on has_member_function:
#ifndef WAR_BUG_1731

template <typename T>
struct my_allocator_with_custom_construct1 : thrust::device_malloc_allocator<T>
{
  _CCCL_HOST_DEVICE my_allocator_with_custom_construct1() {}

  _CCCL_HOST_DEVICE void construct(T* p)
  {
    *p = 13;
  }
};

template <typename T>
void TestAllocatorCustomDefaultConstruct(size_t n)
{
  thrust::device_vector<T> ref(n, 13);
  thrust::device_vector<T, my_allocator_with_custom_construct1<T>> vec(n);

  ASSERT_EQUAL_QUIET(ref, vec);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorCustomDefaultConstruct);

template <typename T>
struct my_allocator_with_custom_construct2 : thrust::device_malloc_allocator<T>
{
  _CCCL_HOST_DEVICE my_allocator_with_custom_construct2() {}

  template <typename Arg>
  _CCCL_HOST_DEVICE void construct(T* p, const Arg&)
  {
    *p = 13;
  }
};

template <typename T>
void TestAllocatorCustomCopyConstruct(size_t n)
{
  thrust::device_vector<T> ref(n, 13);
  thrust::device_vector<T> copy_from(n, 7);
  thrust::device_vector<T, my_allocator_with_custom_construct2<T>> vec(copy_from.begin(), copy_from.end());

  ASSERT_EQUAL_QUIET(ref, vec);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorCustomCopyConstruct);

#endif // !WAR_BUG_1731

// The has_member_destroy trait depends on has_member_function:
#ifndef WAR_BUG_1731

template <typename T>
struct my_allocator_with_custom_destroy
{
  // This is only used with thrust::cpp::vector:
  using system_type = thrust::cpp::tag;

  using value_type      = T;
  using reference       = T&;
  using const_reference = const T&;

  static bool g_state;

  _CCCL_HOST my_allocator_with_custom_destroy() {}

  _CCCL_HOST my_allocator_with_custom_destroy(const my_allocator_with_custom_destroy& other)
      : use_me_to_alloc(other.use_me_to_alloc)
  {}

  _CCCL_HOST ~my_allocator_with_custom_destroy() {}

  _CCCL_HOST_DEVICE void destroy(T*) noexcept
  {
    NV_IF_TARGET(NV_IS_HOST, (g_state = true;));
  }

  value_type* allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type* ptr, std::ptrdiff_t n) noexcept
  {
    use_me_to_alloc.deallocate(ptr, n);
  }

  bool operator==(const my_allocator_with_custom_destroy&) const
  {
    return true;
  }

  bool operator!=(const my_allocator_with_custom_destroy& other) const
  {
    return !(*this == other);
  }

  using is_always_equal = thrust::detail::true_type;

  // use composition rather than inheritance
  // to avoid inheriting std::allocator's member
  // function destroy
  std::allocator<T> use_me_to_alloc;
};

template <typename T>
bool my_allocator_with_custom_destroy<T>::g_state = false;

template <typename T>
void TestAllocatorCustomDestroy(size_t n)
{
  my_allocator_with_custom_destroy<T>::g_state = false;

  {
    thrust::cpp::vector<T, my_allocator_with_custom_destroy<T>> vec(n);
  } // destroy everything

  // state should only be true when there are values to destroy:
  ASSERT_EQUAL(n > 0, my_allocator_with_custom_destroy<T>::g_state);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorCustomDestroy);

#endif // !WAR_BUG_1731

template <typename T>
struct my_minimal_allocator
{
  using value_type = T;

  // XXX ideally, we shouldn't require these two aliases
  using reference       = T&;
  using const_reference = const T&;

  _CCCL_HOST my_minimal_allocator() {}

  _CCCL_HOST my_minimal_allocator(const my_minimal_allocator& other)
      : use_me_to_alloc(other.use_me_to_alloc)
  {}

  _CCCL_HOST ~my_minimal_allocator() {}

  value_type* allocate(std::ptrdiff_t n)
  {
    return use_me_to_alloc.allocate(n);
  }

  void deallocate(value_type* ptr, std::ptrdiff_t n) noexcept
  {
    use_me_to_alloc.deallocate(ptr, n);
  }

  std::allocator<T> use_me_to_alloc;
};

template <typename T>
void TestAllocatorMinimal(size_t n)
{
  thrust::cpp::vector<int, my_minimal_allocator<int>> vec(n, 13);

  // XXX copy to h_vec because ASSERT_EQUAL doesn't know about cpp::vector
  thrust::host_vector<int> h_vec(vec.begin(), vec.end());
  thrust::host_vector<int> ref(n, 13);

  ASSERT_EQUAL(ref, h_vec);
}
DECLARE_VARIABLE_UNITTEST(TestAllocatorMinimal);

void TestAllocatorTraitsRebind()
{
  ASSERT_EQUAL(
    (::cuda::std::is_same<typename thrust::detail::allocator_traits<
                            thrust::device_malloc_allocator<int>>::template rebind_traits<float>::other,
                          typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<float>>>::value),
    true);

  ASSERT_EQUAL(
    (::cuda::std::is_same<
      typename thrust::detail::allocator_traits<my_minimal_allocator<int>>::template rebind_traits<float>::other,
      typename thrust::detail::allocator_traits<my_minimal_allocator<float>>>::value),
    true);
}
DECLARE_UNITTEST(TestAllocatorTraitsRebind);

void TestAllocatorTraitsRebindCpp11()
{
  ASSERT_EQUAL(
    (::cuda::std::is_same<
      typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<int>>::template rebind_alloc<float>,
      thrust::device_malloc_allocator<float>>::value),
    true);

  ASSERT_EQUAL((::cuda::std::is_same<
                 typename thrust::detail::allocator_traits<my_minimal_allocator<int>>::template rebind_alloc<float>,
                 my_minimal_allocator<float>>::value),
               true);

  ASSERT_EQUAL(
    (::cuda::std::is_same<
      typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<int>>::template rebind_traits<float>,
      typename thrust::detail::allocator_traits<thrust::device_malloc_allocator<float>>>::value),
    true);

  ASSERT_EQUAL((::cuda::std::is_same<
                 typename thrust::detail::allocator_traits<my_minimal_allocator<int>>::template rebind_traits<float>,
                 typename thrust::detail::allocator_traits<my_minimal_allocator<float>>>::value),
               true);
}
DECLARE_UNITTEST(TestAllocatorTraitsRebindCpp11);
