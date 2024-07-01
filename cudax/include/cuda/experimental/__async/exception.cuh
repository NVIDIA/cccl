//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <exception> // IWYU pragma: keep export

#include "config.cuh"

#if defined(__CUDACC__)
#  include <nv/target>
#  define _CUDAX_CATCH(...)
#  define _CUDAX_TRY(TRY, CATCH) \
    NV_IF_TARGET(NV_IS_HOST, (try { _NV_EVAL TRY } catch (...){_NV_EVAL CATCH}), ({_NV_EVAL TRY}))
#else
#  define _CUDAX_CATCH(...)
#  define _CUDAX_TRY(TRY, CATCH) _NV_EVAL(try { _NV_EVAL TRY } catch (...){_NV_EVAL CATCH})
#endif

#if defined(__CUDA_ARCH__)
// Treat everything as no-throw in device code
#  define _CUDAX_NOEXCEPT_EXPR(...) true
#else
// This is the default behavior for host code, and for nvc++
#  define _CUDAX_NOEXCEPT_EXPR(...) noexcept(__VA_ARGS__)
#endif
