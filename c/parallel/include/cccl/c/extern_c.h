//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __cplusplus

#  define CCCL_C_EXTERN_C_BEGIN extern "C" {
#  define CCCL_C_EXTERN_C_END   }

#else

#  define CCCL_C_EXTERN_C_BEGIN
#  define CCCL_C_EXTERN_C_END

#endif
