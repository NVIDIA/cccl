//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#include <cuda/stream_ref>
#include <cuda/std/cassert>

int main(int argc, char** argv) {
    NV_IF_TARGET(NV_IS_HOST,(
      cuda::stream_ref left{reinterpret_cast<cudaStream_t>(42)};
      cuda::stream_ref right{reinterpret_cast<cudaStream_t>(1337)};
      static_assert(noexcept(left == right), "");
      static_assert(noexcept(left != right), "");

      assert(left == left);
      assert(left != right);
    ))

    return 0;
}
