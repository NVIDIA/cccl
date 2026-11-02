#include <cuda_fp16.h>

#include <thrust/complex.h>
#include <thrust/detail/alignment.h>
#include <thrust/detail/preprocessor.h>

#include <unittest/unittest.h>

template <typename T, typename VectorT>
void test_complex_alignment()
{
  static_assert(sizeof(thrust::complex<T>) == sizeof(VectorT));
  static_assert(alignof(thrust::complex<T>) == alignof(VectorT));

  static_assert(sizeof(thrust::complex<T const>) == sizeof(VectorT));
  static_assert(alignof(thrust::complex<T const>) == alignof(VectorT));
}
TEST_CASE("TestComplexCharAlignment", "[complex]")
{
  test_complex_alignment<char, char2>();
}
TEST_CASE("TestComplexShortAlignment", "[complex]")
{
  test_complex_alignment<short, short2>();
}
TEST_CASE("TestComplexIntAlignment", "[complex]")
{
  test_complex_alignment<int, int2>();
}
TEST_CASE("TestComplexLongAlignment", "[complex]")
{
  test_complex_alignment<long, long2>();
}
TEST_CASE("TestComplexHalfAlignment", "[complex]")
{
  test_complex_alignment<__half, __half2>();
}
TEST_CASE("TestComplexFloatAlignment", "[complex]")
{
  test_complex_alignment<float, float2>();
}
TEST_CASE("TestComplexDoubleAlignment", "[complex]")
{
  test_complex_alignment<double, double2>();
}
