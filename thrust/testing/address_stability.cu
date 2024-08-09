#include <thrust/address_stability.h>

#include <unittest/unittest.h>

struct MyPlus
{
  _CCCL_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }
};

void TestAddressStability()
{
  using thrust::can_copy_arguments;

  static_assert(!can_copy_arguments<thrust::plus<int>, int, int>::value, "");
  static_assert(can_copy_arguments<decltype(thrust::allow_copied_arguments(thrust::plus<int>{})), int, int>::value, "");

  static_assert(!can_copy_arguments<MyPlus, int, int>::value, "");
  static_assert(can_copy_arguments<decltype(thrust::allow_copied_arguments(MyPlus{})), int, int>::value, "");
}
DECLARE_UNITTEST(TestAddressStability);
