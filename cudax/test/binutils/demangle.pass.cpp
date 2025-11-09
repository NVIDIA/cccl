//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This test uses assert(...) for checking results
#undef NDEBUG

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <cuda/experimental/binutils.cuh>

#include <stdexcept>
#include <string>

namespace cudax = cuda::experimental;

bool test()
{
  // 1. Test signature
  static_assert(cuda::std::is_same_v<std::string, decltype(cudax::demangle(cuda::std::string_view{}))>);
  static_assert(!noexcept(cudax::demangle(cuda::std::string_view{})));

  // 2. Test positive case
  {
    constexpr auto real_mangled_name = "_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
    const auto demangled             = cudax::demangle(real_mangled_name);
    assert(demangled == "clstmp01<cls01>::clstmp01_mf01()");
  }

  // 3. Test error case
  {
#if _CCCL_HAS_EXCEPTIONS()
    constexpr auto fake_mangled_name = "B@d_iDentiFier";
    try
    {
      auto demangled = cudax::demangle(fake_mangled_name);
      assert(false);
    }
    catch (const std::runtime_error&)
    {
      assert(true);
    }
    catch (...)
    {
      assert(false);
    }
#endif // _CCCL_HAS_EXCEPTIONS()
  }

  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
