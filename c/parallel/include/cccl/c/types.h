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

#include <cccl/c/extern_c.h>
#include <stddef.h>

CCCL_C_EXTERN_C_BEGIN

typedef enum cccl_type_enum
{
  CCCL_INT8    = 0,
  CCCL_INT16   = 1,
  CCCL_INT32   = 2,
  CCCL_INT64   = 3,
  CCCL_UINT8   = 4,
  CCCL_UINT16  = 5,
  CCCL_UINT32  = 6,
  CCCL_UINT64  = 7,
  CCCL_FLOAT32 = 8,
  CCCL_FLOAT64 = 9,
  CCCL_STORAGE = 10,
} cccl_type_enum;

typedef struct cccl_type_info
{
  size_t size;
  size_t alignment;
  cccl_type_enum type;
} cccl_type_info;

typedef enum cccl_op_kind_t
{
  CCCL_STATELESS = 0,
  CCCL_STATEFUL  = 1,
} cccl_op_kind_t;

typedef struct cccl_op_t
{
  cccl_op_kind_t type;
  const char* name;
  const char* ltoir;
  size_t ltoir_size;
  size_t size;
  size_t alignment;
  void* state;
} cccl_op_t;

typedef enum cccl_iterator_kind_t
{
  CCCL_POINTER  = 0,
  CCCL_ITERATOR = 1,
} cccl_iterator_kind_t;

typedef struct cccl_value_t
{
  cccl_type_info type;
  void* state;
} cccl_value_t;

typedef struct cccl_iterator_t
{
  size_t size;
  size_t alignment;
  cccl_iterator_kind_t type;
  cccl_op_t advance;
  cccl_op_t dereference;
  cccl_type_info value_type;
  void* state;
} cccl_iterator_t;

typedef enum cccl_sort_order_t
{
  CCCL_ASCENDING  = 0,
  CCCL_DESCENDING = 1,
} cccl_sort_order_t;

CCCL_C_EXTERN_C_END
