#include <cuda/__functional/address_stability.h>

#include <unittest/unittest.h>

struct my_plus
{
  _CCCL_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }
};

void TestAddressStability()
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  static_assert(!proclaims_copyable_arguments<thrust::plus<int>>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(thrust::plus<int>{}))>::value, "");

  static_assert(!proclaims_copyable_arguments<my_plus>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus{}))>::value, "");
}
DECLARE_UNITTEST(TestAddressStability);
