//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DISABLE_NVFP_CONVERSIONS_AND_OPERATORS_H
#define SUPPORT_DISABLE_NVFP_CONVERSIONS_AND_OPERATORS_H

#define __CUDA_NO_FP4_CONVERSIONS__          1
#define __CUDA_NO_FP4_CONVERSION_OPERATORS__ 1
#define __CUDA_NO_FP6_CONVERSIONS__          1
#define __CUDA_NO_FP6_CONVERSION_OPERATORS__ 1
#define __CUDA_NO_FP8_CONVERSIONS__          1
#define __CUDA_NO_FP8_CONVERSION_OPERATORS__ 1
#define __CUDA_NO_HALF_CONVERSIONS__         1
#define __CUDA_NO_HALF_OPERATORS__           1
#define __CUDA_NO_HALF2_OPERATORS__          1
#define __CUDA_NO_BFLOAT16_CONVERSIONS__     1
#define __CUDA_NO_BFLOAT16_OPERATORS__       1
#define __CUDA_NO_BFLOAT162_OPERATORS__      1

#endif // SUPPORT_DISABLE_NVFP_CONVERSIONS_AND_OPERATORS_H
