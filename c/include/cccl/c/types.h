//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#if defined(_WIN32)
#  define CCCL_C_API __declspec(dllexport)
#else // ^^^ _WIN32 ^^^ / vvv !_WIN32 vvv
#  define CCCL_C_API __attribute__((visibility("default")))
#endif // !_WIN32

enum class cccl_type_enum
{
  INT8    = 0,
  INT16   = 1,
  INT32   = 2,
  INT64   = 3,
  UINT8   = 4,
  UINT16  = 5,
  UINT32  = 6,
  UINT64  = 7,
  FLOAT32 = 8,
  FLOAT64 = 9,
  STORAGE = 10
};

struct cccl_type_info
{
  int size;
  int alignment;
  cccl_type_enum type;
};

enum class cccl_op_kind_t
{
  stateless = 0,
  stateful  = 1
};

struct cccl_op_t
{
  cccl_op_kind_t type;
  const char* name;
  const char* ltoir;
  int ltoir_size;
  int size;
  int alignment;
  void* state;
};

enum class cccl_iterator_kind_t
{
  pointer  = 0,
  iterator = 1
};

struct cccl_value_t
{
  cccl_type_info type;
  void* state;
};

struct cccl_iterator_t
{
  int size;
  int alignment;
  cccl_iterator_kind_t type;
  cccl_op_t advance;
  cccl_op_t dereference;
  cccl_type_info value_type;
  void* state;
};
