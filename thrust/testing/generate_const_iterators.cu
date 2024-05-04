
#include <unittest/runtime_static_assert.h>
#include <unittest/unittest.h>

// The runtime_static_assert header needs to come first as we are overwriting thrusts internal static assert
#include <thrust/generate.h>

struct generator
{
  _CCCL_HOST_DEVICE int operator()() const
  {
    return 1;
  }
};

void TestGenerateConstIteratorCompilationError()
{
  thrust::host_vector<int> test1(10);

  ASSERT_STATIC_ASSERT(thrust::generate(test1.cbegin(), test1.cend(), generator()));
  ASSERT_STATIC_ASSERT(thrust::generate_n(test1.cbegin(), 10, generator()));
}
DECLARE_UNITTEST(TestGenerateConstIteratorCompilationError);

void TestFillConstIteratorCompilationError()
{
  thrust::host_vector<int> test1(10);
  ASSERT_STATIC_ASSERT(thrust::fill(test1.cbegin(), test1.cend(), 1));
}
DECLARE_UNITTEST(TestFillConstIteratorCompilationError);
