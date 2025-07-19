//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/__binutils_>
#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <string>

void test()
{
  // 1. test signature of
  static_assert(cuda::std::is_same_v<decltype(cuda::demangle(cuda::std::string_view{})), std::string>);
  static_assert(!noexcept(cuda::demangle(cuda::std::string_view{})));

  // 2. test positive case
  {
    constexpr auto real_mangled_name = "_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
    const auto demangled             = cuda::demangle(real_mangled_name);
    assert(demangled == "clstmp01<cls01>::clstmp01_mf01()");
  }

  // 3. test error case
#if _CCCL_HAS_EXCEPTIONS()
  {
    constexpr auto fake_mangled_name = "B@d_iDentiFier";
    try
    {
      auto demangled = cuda::demangle(fake_mangled_name);
      assert(false);
    }
    catch (const cuda::cuda_error& e)
    {
      assert(e.status() == cudaErrorInvalidSymbol);
    }
    catch (...)
    {
      assert(false);
    }
  }
#endif // _CCCL_HAS_EXCEPTIONS()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
