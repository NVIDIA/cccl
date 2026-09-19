//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include "hash_test_helper.h"
#include "literal.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(23) // integer constant is too large

#if _CCCL_HAS_INT128()

TEST_FUNC void test()
{
  using namespace test_integer_literals;

  hash_test<cuda::hash_algorithm::murmurhash3_x64_128> murmurhash3_x64_128_test;

  murmurhash3_x64_128_test(cuda::std::int32_t(0), 0x5896'2316'1cf5'26f1'cfa0'f7dd'd84c'76bc_u128, 0);
  murmurhash3_x64_128_test(cuda::std::int32_t(9), 0xe22f'e429'0d7f'b7ae'18b1'50d0'55d8'e3d3_u128, 0);
  murmurhash3_x64_128_test(cuda::std::int32_t(42), 0xe2d2'3d6a'2bbc'b816'286f'48e6'1c6e'34cf_u128, 0);
  murmurhash3_x64_128_test(cuda::std::int32_t(42), 0xf9e3'fe3d'853f'a768'1f35'a00f'446c'3666_u128, 42);

  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int32_t, 2>{2, 2}, 0x5cdb'0863'7d1e'9d0d'a99b'1693'2285'6329_u128, 0);
  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int32_t, 3>{1, 4, 9}, 0x8947'1a29'c153'6d4a'0426'c243'f24c'7810_u128, 42);
  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int32_t, 4>{42, 64, 108, 1024}, 0x40a9'd2ba'9d2e'9c80'3c23'b972'bb19'ac29_u128, 63);
  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int32_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    0x621e'f312'a122'7e70'2dd4'86f4'6901'd718_u128,
    1024);

  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int64_t, 2>{2, 2}, 0xcf51'fb88'00a2'a7e1'76b9'44c9'28e1'089f_u128, 0);
  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int64_t, 3>{1, 4, 9}, 0x6200'4bbd'c71f'8c0d'ba8d'd0b7'20e3'ba43_u128, 42);
  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int64_t, 4>{42, 64, 108, 1024}, 0xcf87'ef07'2a17'79ea'79ef'8ffe'1aba'01bc_u128, 63);
  murmurhash3_x64_128_test(
    cuda::std::array<cuda::std::int64_t, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    0x925c'a669'da8e'70e4'd5db'0a3b'b56b'2371_u128,
    1024);
}

#endif // _CCCL_HAS_INT128()

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  test();
#endif // _CCCL_HAS_INT128()
  return 0;
}
