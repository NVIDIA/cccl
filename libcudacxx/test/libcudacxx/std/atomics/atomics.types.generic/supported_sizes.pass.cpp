//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename T>
TEST_FUNC void check_supported_type(T v)
{
  cuda::std::atomic<T> atom(v);
  cuda::std::atomic_ref<T> ref(v);
}

int main(int, char**)
{
  check_supported_type(static_cast<char>(0));
  check_supported_type(static_cast<signed char>(0));
  check_supported_type(static_cast<unsigned char>(0));
  check_supported_type(static_cast<short>(0));
  check_supported_type(static_cast<unsigned short>(0));
  check_supported_type(static_cast<int>(0));
  check_supported_type(static_cast<unsigned int>(0));
  check_supported_type(static_cast<long>(0));
  check_supported_type(static_cast<unsigned long>(0));
  check_supported_type(static_cast<long long>(0));
  check_supported_type(static_cast<unsigned long long>(0));
  check_supported_type(static_cast<wchar_t>(0));
  check_supported_type(static_cast<char16_t>(0));
  check_supported_type(static_cast<char32_t>(0));
  check_supported_type(static_cast<uintptr_t>(0));
  check_supported_type(static_cast<uint8_t>(0));
  check_supported_type(static_cast<int16_t>(0));
  check_supported_type(static_cast<uint16_t>(0));
  check_supported_type(static_cast<int32_t>(0));
  check_supported_type(static_cast<uint32_t>(0));
  check_supported_type(static_cast<int64_t>(0));
  check_supported_type(static_cast<uint64_t>(0));
#if _CCCL_HAS_INT128()
  NV_IF_TARGET(NV_IS_DEVICE,
               // Perform check only on device
               (check_supported_type(static_cast<__int128_t>(0)); check_supported_type(static_cast<__uint128_t>(0));))
#endif // _CCCL_HAS_INT128()

  return 0;
}
