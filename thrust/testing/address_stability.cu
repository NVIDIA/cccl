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
  using ::cuda::allow_copied_arguments;
  using ::cuda::allows_copied_arguments;

  static_assert(!allows_copied_arguments<thrust::plus<int>, int, int>::value, "");
  static_assert(allows_copied_arguments<decltype(allow_copied_arguments(thrust::plus<int>{})), int, int>::value, "");

  static_assert(!allows_copied_arguments<my_plus, int, int>::value, "");
  static_assert(allows_copied_arguments<decltype(allow_copied_arguments(my_plus{})), int, int>::value, "");
}
DECLARE_UNITTEST(TestAddressStability);
