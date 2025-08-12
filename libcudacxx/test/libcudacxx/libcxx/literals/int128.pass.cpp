#include <cuda/std/cassert>

#include "literal.h"

// nvcc complains about the i128 literal constants being too large
_CCCL_DIAG_SUPPRESS_NVCC(23)

#if _CCCL_HAS_INT128()

using namespace test_integer_literals;

__host__ __device__ constexpr void test_binary()
{
  assert(0b0_i128 == 0b0);
  assert(0b1_i128 == 0b1);
  assert(0b1111'1111_i128 == 0b1111'1111);
  assert(0b1111'1111'1111'1111_i128 == 0b1111'1111'1111'1111);
  assert(0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111_i128
         == 0b0111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111'1111);

  assert(-0b0_i128 == 0b0);
  assert(-0b1_i128 == -0b1);
  assert(-0b1111'1111_i128 == -0b1111'1111);
  assert(-0b1111'1111'1111'1111_i128 == -0b1111'1111'1111'1111);

  assert(0B1 == 0b1);
}

__host__ __device__ constexpr void test_octal()
{
  assert(00_i128 == 00);
  assert(01_i128 == 01);
  assert(07_i128 == 07);
  assert(017_i128 == 017);
  assert(0377_i128 == 0377);
  assert(01'777'777'777'777'777'777'777'375_i128 == (__int128_t{~0ull} << 9) + 0'375);

  assert(-00_i128 == 00);
  assert(-01_i128 == -01);
  assert(-07_i128 == -07);
  assert(-017_i128 == -017);
  assert(-0377_i128 == -0377);
  assert(-01'777'777'777'777'777'777'777'375_i128 == -(__int128_t{~0ull} << 9) - 0'375);
}

__host__ __device__ constexpr void test_decimal()
{
  assert(0_i128 == 0);
  assert(1_i128 == 1);
  assert(123_i128 == 123);
  assert(123'456'789_i128 == 123'456'789);
  assert(9'223'372'036'854'775'807_i128 == 9'223'372'036'854'775'807);
  assert(12'345'678'901'234'567'890_i128 == __int128_t{12'345'678'901'234'567'890ull});

  assert(-0_i128 == 0);
  assert(-1_i128 == -1);
  assert(-123_i128 == -123);
  assert(-123'456'789_i128 == -123'456'789);
  assert(-9'223'372'036'854'775'807_i128 == -9'223'372'036'854'775'807);
  assert(-12'345'678'901'234'567'890_i128 == -__int128_t{12'345'678'901'234'567'890ull});
}

__host__ __device__ constexpr void test_hexadecimal()
{
  assert(0x0_i128 == 0x0);
  assert(0x1_i128 == 0x1);
  assert(0xF_i128 == 0xF);
  assert(0xFF_i128 == 0xFF);
  assert(0xFFFF_i128 == 0xFFFF);
  assert(0x7FFF'FFFF'FFFF'FFFF_i128 == 0x7FFF'FFFF'FFFF'FFFF);
  assert(0xFF'FFFF'FFFF'FFFF'FFFF_i128 == (__int128_t{0xFFFF'FFFF'FFFF'FFFF} << 8) + 0xff);
  assert(0x7FFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF_i128 == static_cast<__int128_t>(~__uint128_t{0} >> 1));

  assert(-0x0_i128 == 0x0);
  assert(-0x1_i128 == -0x1);
  assert(-0xF_i128 == -0xF);
  assert(-0xFF_i128 == -0xFF);
  assert(-0xFFFF_i128 == -0xFFFF);
  assert(-0x7FFF'FFFF'FFFF'FFFF_i128 == -0x7FFF'FFFF'FFFF'FFFF);
  assert(-0xFF'FFFF'FFFF'FFFF'FFFF_i128 == -(__int128_t{0xFFFF'FFFF'FFFF'FFFF} << 8) - 0xff);
  assert(-0x7FFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF_i128 == static_cast<__int128_t>(~(~__uint128_t{0} >> 1)) + 1);

  assert(0Xabcdef_i128 == 0xABCDEF);
}

__host__ __device__ constexpr bool test()
{
  test_binary();
  test_octal();
  test_decimal();
  test_hexadecimal();

  return true;
}

#endif // _CCCL_HAS_INT128()

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  test();
  static_assert(test());
#endif // _CCCL_HAS_INT128()
  return 0;
}
