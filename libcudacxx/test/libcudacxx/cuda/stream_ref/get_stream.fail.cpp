//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// cuda::get_stream must reject types that are only convertible to a stream through a non-const conversion operator,
// because it receives its argument by const reference.

#include <cuda/stream>

#include "test_macros.h"

struct lvalue_ref_qualified_stream
{
  ::cudaStream_t stream_{};

  operator ::cudaStream_t() & noexcept
  {
    return stream_;
  }
};

struct rvalue_ref_qualified_stream
{
  ::cudaStream_t stream_{};

  operator ::cudaStream_t() && noexcept
  {
    return stream_;
  }
};

TEST_FUNC void test()
{
  lvalue_ref_qualified_stream lvalue_stream{};
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}non-const conversion operator}}
  unused(::cuda::get_stream(lvalue_stream));

  rvalue_ref_qualified_stream rvalue_stream{};
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}non-const conversion operator}}
  unused(::cuda::get_stream(rvalue_stream));
}

int main(int, char**)
{
  return 0;
}
