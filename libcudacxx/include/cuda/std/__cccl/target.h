//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_TARGET_H
#define __CCCL_TARGET_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() && _CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_TARGET_is_device           ::nv::target::is_device
#  define _CCCL_TARGET_is_host             ::nv::target::is_host
#  define _CCCL_TARGET_any_target          ::nv::target::any_target
#  define _CCCL_TARGET_no_target           ::nv::target::no_target
#  define _CCCL_TARGET_is_exactly(_SM_VER) ::nv::target::is_exactly(::nv::target::sm_##_SM_VER)
#  define _CCCL_TARGET_provides(_SM_VER)   ::nv::target::provides(::nv::target::sm_##_SM_VER)

#  define _CCCL_TARGET_CONTINUE(_OP, ...)             _CCCL_##_OP##_TARGET(__VA_ARGS__)
#  define _CCCL_TARGET_EXPAND_BLOCK(...)              {__VA_ARGS__}
#  define _CCCL_TARGET_EXPAND_BLOCK_AND_CONTINUE(...) {__VA_ARGS__} _CCCL_TARGET_CONTINUE

#  define _CCCL_IF_TARGET(_COND)     \
    if target (_CCCL_TARGET_##_COND) \
    _CCCL_TARGET_EXPAND_BLOCK_AND_CONTINUE
#  define _CCCL_elif_TARGET(_COND) else if target (_CCCL_TARGET_##_COND) _CCCL_TARGET_EXPAND_BLOCK_AND_CONTINUE
#  define _CCCL_else_TARGET()      else _CCCL_TARGET_EXPAND_BLOCK
#  define _CCCL_endif_TARGET()
#else
#  if defined(__CUDA_ARCH__)
#    define _CCCL_TARGET_is_device           1
#    define _CCCL_TARGET_is_host             0
#    define _CCCL_TARGET_is_exactly(_SM_VER) // todo
#    define _CCCL_TARGET_provides(_SM_VER)   // todo
#  else
#    define _CCCL_TARGET_is_device           0
#    define _CCCL_TARGET_is_host             1
#    define _CCCL_TARGET_is_exactly(_SM_VER) 0
#    define _CCCL_TARGET_provides(_SM_VER)   0
#  endif
#  define _CCCL_TARGET_any_target 1
#  define _CCCL_TARGET_no_target  0

#  define _CCCL_TARGET_CONTINUE(_CAN_EXPAND, _OP, ...) _OP(_CAN_EXPAND, __VA_ARGS__)

#  define _CCCL_TARGET_CONTINUE_0(_OP, ...) _CCCL_TARGET_CONTINUE(0, _CCCL_##_OP##_TARGET, __VA_ARGS__)
#  define _CCCL_TARGET_CONTINUE_1(_OP, ...) _CCCL_TARGET_CONTINUE(1, _CCCL_##_OP##_TARGET, __VA_ARGS__)

// bitfield: 1. can expand, 2. condition, 3. should continue expansion
#  define _CCCL_TARGET_EXPAND_IF_0_0_0(...)
#  define _CCCL_TARGET_EXPAND_IF_0_1_0(...)
#  define _CCCL_TARGET_EXPAND_IF_1_0_0(...)
#  define _CCCL_TARGET_EXPAND_IF_1_1_0(...) {__VA_ARGS__}
#  define _CCCL_TARGET_EXPAND_IF_0_0_1(...) _CCCL_TARGET_CONTINUE_0
#  define _CCCL_TARGET_EXPAND_IF_0_1_1(...) _CCCL_TARGET_CONTINUE_0
#  define _CCCL_TARGET_EXPAND_IF_1_0_1(...) _CCCL_TARGET_CONTINUE_1
#  define _CCCL_TARGET_EXPAND_IF_1_1_1(...) {__VA_ARGS__} _CCCL_TARGET_CONTINUE_0

#  define _CCCL_TARGET_EVAL_IMPL(_CAN_EXPAND, _COND, _CONTINUE) \
    _CCCL_TARGET_EXPAND_IF_##_CAN_EXPAND##_##_COND##_##_CONTINUE
#  define _CCCL_TARGET_EVAL(_CAN_EXPAND, _COND, _CONTINUE) _CCCL_TARGET_EVAL_IMPL(_CAN_EXPAND, _COND, _CONTINUE)

#  define _CCCL_IF_TARGET(_COND)                _CCCL_TARGET_EVAL(1, _CCCL_TARGET_##_COND, 1)
#  define _CCCL_elif_TARGET(_CAN_EXPAND, _COND) _CCCL_TARGET_EVAL(_CAN_EXPAND, _CCCL_TARGET_##_COND, 1)
#  define _CCCL_else_TARGET(_CAN_EXPAND, _COND) _CCCL_TARGET_EVAL(_CAN_EXPAND, 1, 0)
#  define _CCCL_endif_TARGET(_CAN_EXPAND, _COND)
#endif

#endif // __CCCL_TARGET_H
