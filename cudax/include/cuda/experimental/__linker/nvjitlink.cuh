//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___LINKER_NVJITLINK_CUH
#define _CUDAX___LINKER_NVJITLINK_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#undef NVJITLINK_NO_INLINE

#include <cuda/std/__cccl/prologue.h>

extern "C" {

nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle*, uint32_t, const char**);
nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle*);
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle, nvJitLinkInputType, const void*, size_t, const char*);
nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle, nvJitLinkInputType, const char*);
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle);
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle, void*);
nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle, char*);
nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle, char*);
nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle, char*);
}

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___LINKER_NVJITLINK_CUH
