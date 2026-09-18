#include <thrust/detail/config.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <unittest/unittest.h>

template <typename BaseAlloc, bool PropagateOnSwap>
class stateful_allocator : public BaseAlloc
{
  using base_traits = cuda::std::allocator_traits<BaseAlloc>;

public:
  stateful_allocator(int i)
      : state(i)
  {}

  _CCCL_HOST ~stateful_allocator() {} // NOLINT(modernize-use-equals-default)

  stateful_allocator(const stateful_allocator& other)
      : BaseAlloc(other)
      , state(other.state)
  {}

  stateful_allocator& operator=(const stateful_allocator& other)
  {
    state = other.state;
    return *this;
  }

  stateful_allocator(stateful_allocator&& other) noexcept
      : BaseAlloc(static_cast<BaseAlloc&&>(other))
      , state(other.state)
  {
    other.state = 0;
  }

  stateful_allocator& operator=(stateful_allocator&& other) noexcept
  {
    state       = other.state;
    other.state = 0;
    return *this;
  }

  static int last_allocated;
  static int last_deallocated;

  using pointer         = typename base_traits::pointer;
  using const_pointer   = typename base_traits::const_pointer;
  using reference       = typename cuda::std::iterator_traits<pointer>::reference;
  using const_reference = typename cuda::std::iterator_traits<const_pointer>::reference;

  pointer allocate(std::size_t size)
  {
    BaseAlloc alloc;
    last_allocated = state;
    return base_traits::allocate(alloc, size);
  }

  void deallocate(pointer ptr, std::size_t size) noexcept
  {
    BaseAlloc alloc;
    last_deallocated = state;
    return base_traits::deallocate(alloc, ptr, size);
  }

  static void construct(pointer ptr)
  {
    BaseAlloc alloc;
    return base_traits::construct(alloc, ptr);
  }

  static void destroy(pointer ptr) noexcept
  {
    BaseAlloc alloc;
    return base_traits::destroy(alloc, ptr);
  }

  bool operator==(const stateful_allocator& rhs) const
  {
    return state == rhs.state;
  }

  bool operator!=(const stateful_allocator& rhs) const
  {
    return state != rhs.state;
  }

  friend std::ostream& operator<<(std::ostream& os, const stateful_allocator& alloc)
  {
    os << "stateful_alloc(" << alloc.state << ")";
    return os;
  }

  using is_always_equal                        = thrust::detail::false_type;
  using propagate_on_container_copy_assignment = thrust::detail::true_type;
  using propagate_on_container_move_assignment = thrust::detail::true_type;
  using propagate_on_container_swap            = thrust::detail::integral_constant<bool, PropagateOnSwap>;

private:
  int state;
};

template <typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_allocated = 0;

template <typename BaseAlloc, bool PropagateOnSwap>
int stateful_allocator<BaseAlloc, PropagateOnSwap>::last_deallocated = 0;

using host_alloc   = stateful_allocator<std::allocator<int>, true>;
using device_alloc = stateful_allocator<thrust::device_allocator<int>, true>;

using host_vector   = thrust::host_vector<int, host_alloc>;
using device_vector = thrust::device_vector<int, device_alloc>;

using host_alloc_nsp   = stateful_allocator<std::allocator<int>, false>;
using device_alloc_nsp = stateful_allocator<thrust::device_allocator<int>, false>;

using host_vector_nsp   = thrust::host_vector<int, host_alloc_nsp>;
using device_vector_nsp = thrust::device_vector<int, device_alloc_nsp>;

template <typename Vector>
void TestVectorAllocatorConstructors()
{
  using Alloc = typename Vector::allocator_type;
  const Alloc alloc1(1);
  const Alloc alloc2(2);

  Vector v1(alloc1);
  REQUIRE(v1.get_allocator() == alloc1);

  Vector v2(10, alloc1);
  REQUIRE(v2.size() == 10u);
  REQUIRE(v2.get_allocator() == alloc1);
  REQUIRE(Alloc::last_allocated == 1);
  Alloc::last_allocated = 0;

  Vector v3(10, 17, alloc1);
  REQUIRE(v3 == std::vector<int>(10, 17));
  REQUIRE(v3.get_allocator() == alloc1);
  REQUIRE(Alloc::last_allocated == 1);
  Alloc::last_allocated = 0;

  Vector v4(v3, alloc2);
  REQUIRE(v3 == v4);
  REQUIRE(v4.get_allocator() == alloc2);
  REQUIRE(Alloc::last_allocated == 2);
  Alloc::last_allocated = 0;

  // FIXME: uncomment this after the vector_base(vector_base&&, const Alloc&)
  // is fixed and implemented
  // Vector v5(std::move(v3), alloc2);
  // ASSERT_EQUAL((v4 == v5), true);
  // ASSERT_EQUAL(v5.get_allocator(), alloc2);
  // ASSERT_EQUAL(Alloc::last_allocated, 1);
  // Alloc::last_allocated = 0;

  Vector v6(v4.begin(), v4.end(), alloc2);
  REQUIRE(v4 == v6);
  REQUIRE(v6.get_allocator() == alloc2);
  REQUIRE(Alloc::last_allocated == 2);
}

TEST_CASE("TestVectorAllocatorConstructorsHost", "[vector_allocators]")
{
  TestVectorAllocatorConstructors<host_vector>();
}

TEST_CASE("TestVectorAllocatorConstructorsDevice", "[vector_allocators]")
{
  TestVectorAllocatorConstructors<device_vector>();
}

template <typename Vector>
void TestVectorAllocatorPropagateOnCopyAssignment()
{
  REQUIRE(cuda::std::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value);

  using Alloc = typename Vector::allocator_type;
  const Alloc alloc1(1);
  const Alloc alloc2(2);

  Vector v1(10, alloc1);
  Vector v2(15, alloc2);

  v2 = v1;
  REQUIRE(v1 == v2);
  REQUIRE(v2.get_allocator() == alloc1);
  REQUIRE(Alloc::last_allocated == 1);
  REQUIRE(Alloc::last_deallocated == 2);
}

TEST_CASE("TestVectorAllocatorPropagateOnCopyAssignmentHost", "[vector_allocators]")
{
  TestVectorAllocatorPropagateOnCopyAssignment<host_vector>();
}

TEST_CASE("TestVectorAllocatorPropagateOnCopyAssignmentDevice", "[vector_allocators]")
{
  TestVectorAllocatorPropagateOnCopyAssignment<device_vector>();
}

template <typename Vector>
void TestVectorAllocatorPropagateOnMoveAssignment()
{
  using Alloc = typename Vector::allocator_type;
  REQUIRE(cuda::std::allocator_traits<typename Vector::allocator_type>::propagate_on_container_copy_assignment::value);

  using Alloc = typename Vector::allocator_type;
  const Alloc alloc1(1);
  const Alloc alloc2(2);

  {
    Vector v1(10, alloc1);
    Vector v2(15, alloc2);

    v2 = std::move(v1);
    REQUIRE(v2.get_allocator() == alloc1);
    REQUIRE(Alloc::last_allocated == 2);
    REQUIRE(Alloc::last_deallocated == 2);
  }

  REQUIRE(Alloc::last_deallocated == 1);
}

TEST_CASE("TestVectorAllocatorPropagateOnMoveAssignmentHost", "[vector_allocators]")
{
  TestVectorAllocatorPropagateOnMoveAssignment<host_vector>();
}

TEST_CASE("TestVectorAllocatorPropagateOnMoveAssignmentDevice", "[vector_allocators]")
{
  TestVectorAllocatorPropagateOnMoveAssignment<device_vector>();
}

template <typename Vector>
void TestVectorAllocatorPropagateOnSwap()
{
  using Alloc = typename Vector::allocator_type;
  const Alloc alloc1(1);
  const Alloc alloc2(2);

  Vector v1(10, alloc1);
  Vector v2(17, alloc1);
  using ::cuda::std::swap;
  swap(v1, v2);

  REQUIRE(v1.size() == 17u);
  REQUIRE(v2.size() == 10u);

  Vector v3(15, alloc1);
  Vector v4(31, alloc2);
  REQUIRE_THROWS_MATCHES(
    swap(v3, v4),
    thrust::detail::allocator_mismatch_on_swap,
    Catch::Matchers::Message("swap called on containers with allocators that propagate on swap, "
                             "but compare non-equal"));
}

TEST_CASE("TestVectorAllocatorPropagateOnSwapHost", "[vector_allocators]")
{
  TestVectorAllocatorPropagateOnSwap<host_vector_nsp>();
}

TEST_CASE("TestVectorAllocatorPropagateOnSwapDevice", "[vector_allocators]")
{
  TestVectorAllocatorPropagateOnSwap<device_vector_nsp>();
}
