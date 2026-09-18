#include <thrust/detail/config.h>

#include <thrust/detail/caching_allocator.h>

#include <unittest/unittest.h>

template <typename Allocator>
void test_implementation(Allocator alloc)
{
  using Traits = typename cuda::std::allocator_traits<Allocator>;
  using Ptr    = typename Allocator::pointer;

  const Ptr p = Traits::allocate(alloc, 123);
  Traits::deallocate(alloc, p, 123);

  const Ptr p2 = Traits::allocate(alloc, 123);
  REQUIRE(p == p2);
}

void TestSingleDeviceTLSCachingAllocator()
{
  test_implementation(thrust::detail::single_device_tls_caching_allocator());
};
TEST_CASE("TestSingleDeviceTLSCachingAllocator", "[caching_allocator]")
{
  TestSingleDeviceTLSCachingAllocator();
}
