#include <cuda_fp16.h>

#include <thrust/complex.h>
#include <thrust/detail/alignment.h>
#include <thrust/detail/preprocessor.h>

#include <unittest/unittest.h>

template <typename T, typename VectorT>
void TestComplexAlignment()
{
  static_assert(sizeof(thrust::complex<T>) == sizeof(VectorT));
  static_assert(alignof(thrust::complex<T>) == alignof(VectorT));

  static_assert(sizeof(thrust::complex<T const>) == sizeof(VectorT));
  static_assert(alignof(thrust::complex<T const>) == alignof(VectorT));
}
TEST_CASE("TestComplexCharAlignment", "[complex]")
{
  TestComplexAlignment<char, char2>();
}
TEST_CASE("TestComplexShortAlignment", "[complex]")
{
  TestComplexAlignment<short, short2>();
}
TEST_CASE("TestComplexIntAlignment", "[complex]")
{
  TestComplexAlignment<int, int2>();
}
TEST_CASE("TestComplexLongAlignment", "[complex]")
{
  TestComplexAlignment<long, long2>();
}
TEST_CASE("TestComplexHalfAlignment", "[complex]")
{
  TestComplexAlignment<__half, __half2>();
}
TEST_CASE("TestComplexFloatAlignment", "[complex]")
{
  TestComplexAlignment<float, float2>();
}
TEST_CASE("TestComplexDoubleAlignment", "[complex]")
{
  TestComplexAlignment<double, double2>();
}
