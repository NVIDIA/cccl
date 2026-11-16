#include <thrust/detail/seq.h>
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/omp/detail/execution_policy.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/cstddef>

#include <memory>

#include <unittest/unittest.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <thrust/system/cuda/detail/execution_policy.h>
#endif

template <typename T>
struct test_allocator_t
{};

test_allocator_t<int> test_allocator             = test_allocator_t<int>();
const test_allocator_t<int> const_test_allocator = test_allocator_t<int>();

struct test_memory_resource_t final : thrust::mr::memory_resource<>
{
  void* do_allocate(std::size_t size, std::size_t) override
  {
    return reinterpret_cast<void*>(size); // NOLINT(performance-no-int-to-ptr)
  }

  void do_deallocate(void* ptr, std::size_t size, std::size_t) override
  {
    REQUIRE(ptr == reinterpret_cast<void*>(size)); // NOLINT(performance-no-int-to-ptr)
  }
} test_memory_resource;

struct allocation_record
{
  std::size_t element_size = 0;
  std::size_t count        = 0;
};

template <typename T>
struct recording_allocator
{
  using value_type = T;

  explicit recording_allocator(allocation_record& r)
      : record(&r)
  {}

  template <typename U>
  recording_allocator(const recording_allocator<U>& other)
      : record(other.record)
  {}

  T* allocate(std::size_t n)
  {
    record->element_size = sizeof(T);
    record->count        = n;
    return std::allocator<T>{}.allocate(n);
  }

  void deallocate(T* ptr, std::size_t n)
  {
    REQUIRE(record->element_size == sizeof(T));
    REQUIRE(record->count == n);
    std::allocator<T>{}.deallocate(ptr, n);
  }

  allocation_record* record;
};

// cannot be rebound and must not be copied
struct byte_allocator
{
  using value_type = char;

  byte_allocator()                      = default;
  byte_allocator(const byte_allocator&) = delete;

  char* allocate(std::size_t n)
  {
    outstanding += n;
    return std::allocator<char>{}.allocate(n);
  }

  void deallocate(char* ptr, std::size_t n)
  {
    outstanding -= n;
    std::allocator<char>{}.deallocate(ptr, n);
  }

  std::size_t outstanding = 0;
};

struct byte_pool
{
  using pointer = char*;
};

// a template, but over a pool rather than the value type, so it must not be rebound
template <typename Pool>
struct pool_byte_allocator
{
  using value_type = char;
  using pointer    = typename Pool::pointer;

  pointer allocate(std::size_t n)
  {
    outstanding += n;
    return std::allocator<char>{}.allocate(n);
  }

  void deallocate(pointer ptr, std::size_t n)
  {
    outstanding -= n;
    std::allocator<char>{}.deallocate(ptr, n);
  }

  std::size_t outstanding = 0;
};

template <typename Policy, template <typename> class CRTPBase>
struct policy_info
{
  using policy = Policy;

  template <template <typename, template <typename> class> class Template, typename Argument>
  struct apply_base_second
  {
    using type _CCCL_NODEBUG = Template<Argument, CRTPBase>;
  };
};

template <typename PolicyInfo>
struct TestAllocatorAttachment
{
  template <typename Expected, typename T>
  static void assert_correct(T)
  {
    REQUIRE(::cuda::std::is_same_v<
            T,
            typename PolicyInfo::template apply_base_second<thrust::detail::execute_with_allocator, Expected>::type>);
  }

  template <typename ExpectedResource, typename T>
  static void assert_npa_correct(T)
  {
    REQUIRE(::cuda::std::is_same_v<T,
                                   typename PolicyInfo::template apply_base_second<
                                     thrust::detail::execute_with_allocator,
                                     thrust::mr::allocator<cuda::std::max_align_t, ExpectedResource>>::type>);
  }

  template <typename Policy>
  void test_temporary_allocation_valid(Policy policy)
  {
    using thrust::detail::get_temporary_buffer;

    return_temporary_buffer(policy, get_temporary_buffer<int>(policy, 123).first, 123);
  }

  void operator()()
  {
    typename PolicyInfo::policy policy; // NOLINT(misc-const-correctness)

    // test correctness of attachment
    assert_correct<test_allocator_t<int>>(policy(test_allocator_t<int>()));
    assert_correct<test_allocator_t<int>&>(policy(test_allocator));
    assert_correct<test_allocator_t<int>>(policy(const_test_allocator));

    assert_npa_correct<test_memory_resource_t>(policy(&test_memory_resource));

    // test whether the resulting policy is actually usable
    // a real allocator is necessary here, unlike above
    std::allocator<int> alloc;
    const std::allocator<int> const_alloc;

    test_temporary_allocation_valid(policy(std::allocator<int>()));
    test_temporary_allocation_valid(policy(alloc));
    test_temporary_allocation_valid(policy(const_alloc));
    test_temporary_allocation_valid(policy(&test_memory_resource));

    test_rebinding(policy);
  }

  template <typename Policy>
  void test_rebinding(Policy policy)
  {
    using thrust::detail::get_temporary_buffer;

    allocation_record record;
    {
      auto p       = policy(recording_allocator<char>(record));
      const auto b = get_temporary_buffer<double>(p, 123);
      REQUIRE(record.element_size == sizeof(double));
      REQUIRE(record.count == 123);
      return_temporary_buffer(p, b.first, b.second);
    }

    record = allocation_record{};
    {
      auto p       = policy(recording_allocator<void>(record));
      const auto b = get_temporary_buffer<int>(p, 42);
      REQUIRE(record.element_size == sizeof(int));
      REQUIRE(record.count == 42);
      return_temporary_buffer(p, b.first, b.second);
    }

    byte_allocator bytes;
    {
      auto p       = policy(bytes);
      const auto b = get_temporary_buffer<double>(p, 10);
      REQUIRE(bytes.outstanding == 10 * sizeof(double));
      return_temporary_buffer(p, b.first, b.second);
      REQUIRE(bytes.outstanding == 0);
    }

    pool_byte_allocator<byte_pool> pool_bytes;
    {
      auto p       = policy(pool_bytes);
      const auto b = get_temporary_buffer<double>(p, 10);
      REQUIRE(pool_bytes.outstanding == 10 * sizeof(double));
      return_temporary_buffer(p, b.first, b.second);
      REQUIRE(pool_bytes.outstanding == 0);
    }
  }
};

using sequential_info = policy_info<thrust::detail::seq_t, thrust::system::detail::sequential::execution_policy>;
using cpp_par_info    = policy_info<thrust::system::cpp::detail::par_t, thrust::system::cpp::execution_policy>;
using omp_par_info    = policy_info<thrust::system::omp::detail::par_t, thrust::system::omp::execution_policy>;
using tbb_par_info    = policy_info<thrust::system::tbb::detail::par_t, thrust::system::tbb::execution_policy>;

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using cuda_par_info = policy_info<thrust::system::cuda::detail::par_t, thrust::cuda_cub::execute_on_stream_base>;
#endif

using test_types =
  unittest::type_list<sequential_info,
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
                      cuda_par_info,
#endif
                      cpp_par_info,
                      omp_par_info,
                      tbb_par_info>;
DECLARE_GENERIC_UNITTEST_WITH_TYPES(TestAllocatorAttachment, test_types);
