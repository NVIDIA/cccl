// Regression test for NVBug 2720132.
//
// Summary of 2720132:
//
// 1. The large allocation fails due to running out of memory.
// 2. A `thrust::system::system_error` exception is thrown.
// 3. Local objects are destroyed as the stack is unwound, leading to the destruction of `x`.
// 4. `x` runs a parallel algorithm in its destructor to call the destructors of all of its elements.
// 5. Launching that parallel algorithm fails because of the prior CUDA out of memory error.
// 6. A `thrust::system::system_error` exception is thrown.
// 7. Because we've already got an active exception, `terminate` is called.

#include <thrust/device_vector.h>

#include <cstdint>

#include <unittest/unittest.h>

struct non_trivial
{
  _CCCL_HOST_DEVICE non_trivial() {}
  _CCCL_HOST_DEVICE ~non_trivial() {}
};

void test_out_of_memory_recovery()
{
  try
  {
    thrust::device_vector<non_trivial> x(1);

    thrust::device_vector<std::uint32_t> y(0x00ffffffffffffff);
  }
  catch (...)
  {}
}
DECLARE_UNITTEST(test_out_of_memory_recovery);
