//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  int nqpoints = 3;
  auto ltoken  = ctx.token();

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(NVHPC)
  ctx.parallel_for(exec_place::host(), box(5), ltoken.read())->*[nqpoints] __host__(size_t) {
    _CCCL_ASSERT(nqpoints == 3, "invalid value");
  };
#else
  // parallel_for on the host requires nvcc or nvc++, which alone can classify lambdas for
  // host/device execution; state that limitation instead of letting the kernel instantiation
  // explain it. The call below is declared but never defined, and its 'error' attribute fires
  // only at codegen: clang-tidy and -fsyntax-only sweeps analyze this file cleanly, while an
  // actual build fails at compile time with the message (or at link time, with the name as
  // the message, on compilers without the attribute). The noexcept is load-bearing: with a
  // nontrivial destructor in scope the call would otherwise be emitted as an invoke, whose
  // error-attribute diagnostic clang silently skips.
#  if _CCCL_HAS_ATTRIBUTE(error)
  __attribute__((error("parallel_for on exec_place::host() requires nvcc or nvc++")))
#  endif
  void parallel_for_on_the_host_requires_nvcc_or_nvcpp() noexcept;
  parallel_for_on_the_host_requires_nvcc_or_nvcpp();
#endif

  ctx.finalize();
}
