//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXCEPTION___
#define __CUDAX_EXCEPTION___

#include <stdexcept>

namespace cuda::experimental
{

struct launch_error : public ::std::runtime_error
{
  cudaError_t error;

  explicit launch_error(cudaError_t err)
      : ::std::runtime_error("")
      , error(err)
  {}
};
} // namespace cuda::experimental

#endif
