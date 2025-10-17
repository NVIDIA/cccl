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
#  define CCCL_C_API __attribute__((__visibility__("default")))
#endif // !_WIN32

#include <stddef.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>

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
  CCCL_FLOAT16 = 8, // This may be unsupported if _CCCL_HAS_NVFP16() is false but we can't include the header to check
                    // that here
  CCCL_FLOAT32 = 9,
  CCCL_FLOAT64 = 10,
  CCCL_STORAGE = 11,
  CCCL_BOOLEAN = 12,
} cccl_type_enum;

typedef struct cccl_type_info
{
  size_t size;
  size_t alignment;
  cccl_type_enum type;
} cccl_type_info;

typedef enum cccl_op_kind_t
{
  // Arbitrary semantics, without state.
  CCCL_STATELESS = 0,
  // Arbitrary semantics, with state.
  CCCL_STATEFUL = 1,
  // Well-known semantics, required to be stateless.
  // Equivalent to corresponding function objects in C++'s <functional>.
  // If the types involved are primitive, only the kind field is necessary.
  // Otherwise, the cccl_op_t object must also contain the rest of the fields,
  // as appropriate.
  CCCL_PLUS          = 2,
  CCCL_MINUS         = 3,
  CCCL_MULTIPLIES    = 4,
  CCCL_DIVIDES       = 5,
  CCCL_MODULUS       = 6,
  CCCL_EQUAL_TO      = 7,
  CCCL_NOT_EQUAL_TO  = 8,
  CCCL_GREATER       = 9,
  CCCL_LESS          = 10,
  CCCL_GREATER_EQUAL = 11,
  CCCL_LESS_EQUAL    = 12,
  CCCL_LOGICAL_AND   = 13,
  CCCL_LOGICAL_OR    = 14,
  CCCL_LOGICAL_NOT   = 15,
  CCCL_BIT_AND       = 16,
  CCCL_BIT_OR        = 17,
  CCCL_BIT_XOR       = 18,
  CCCL_BIT_NOT       = 19,
  CCCL_IDENTITY      = 20,
  CCCL_NEGATE        = 21,
  CCCL_MINIMUM       = 22,
  CCCL_MAXIMUM       = 23,
} cccl_op_kind_t;

typedef enum cccl_op_code_type
{
  CCCL_OP_LTOIR      = 0, // Pre-compiled LTO-IR (default for backward compatibility)
  CCCL_OP_CPP_SOURCE = 1 // C++ source code
} cccl_op_code_type;

typedef struct cccl_op_t
{
  cccl_op_kind_t type;
  const char* name;
  const char* code; // Renamed from 'ltoir' - can be either LTO-IR or C++ source
  size_t code_size; // Renamed from 'ltoir_size'
  cccl_op_code_type code_type; // New field to distinguish content type
  size_t size;
  size_t alignment;
  void* state;
} cccl_op_t;

typedef struct cccl_build_config
{
  const char** extra_compile_flags; // e.g., {"-DENABLE_FAST_MATH", "-O3"}
  size_t num_extra_compile_flags;
  const char** extra_include_dirs; // e.g., {"/path/to/my/headers"}
  size_t num_extra_include_dirs;
} cccl_build_config;

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

typedef union
{
  int64_t signed_offset;
  uint64_t unsigned_offset;
} cccl_increment_t;

typedef void (*cccl_host_op_fn_ptr_t)(void*, cccl_increment_t);

typedef struct cccl_iterator_t
{
  size_t size;
  size_t alignment;
  cccl_iterator_kind_t type;
  cccl_op_t advance;
  cccl_op_t dereference;
  cccl_type_info value_type;
  void* state;
  cccl_host_op_fn_ptr_t host_advance;
} cccl_iterator_t;

typedef enum cccl_sort_order_t
{
  CCCL_ASCENDING  = 0,
  CCCL_DESCENDING = 1,
} cccl_sort_order_t;

CCCL_C_EXTERN_C_END
