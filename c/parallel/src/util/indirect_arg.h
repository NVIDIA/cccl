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

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <cccl/c/types.h>

// GCC emits a dangling pointer warning in the `LargeSegmentsSelector` and
// `SmallSegmentsSelector` functions in segmented_sort.cu. The warning occurs
// when `return indirect_arg_t()` invokes the `indirect_arg_t` constructor that
// accepts a `cccl_op_t`. Even though this is a stateless op, we must initialize
// the `ptr` member to a valid address.
//
// We cannot use `nullptr` because the pointer is passed to a driver API that
// requires the size of empty arguments to be 1 (not 0), meaning it will attempt
// to copy a byte from the address.
//
// Initially, we initialized `ptr` to `this`, but this triggered the dangling
// pointer warning in GCC. To avoid this, we use a global variable instead.
static inline char _global_storage = 0;

struct indirect_arg_t
{
  void* ptr;

  indirect_arg_t(cccl_iterator_t& it)
      : ptr(it.type == cccl_iterator_kind_t::CCCL_POINTER ? &it.state : it.state)
  {}

  indirect_arg_t(cccl_op_t& op)
      : ptr(op.type == cccl_op_kind_t::CCCL_STATEFUL ? op.state : &_global_storage)
  {}

  indirect_arg_t(cccl_value_t& val)
      : ptr(val.state)
  {}

  void* operator&() const
  {
    return ptr;
  }
};

template <typename U>
concept Increment64 = std::is_integral_v<U> && sizeof(U) == sizeof(int64_t);

struct indirect_iterator_t
{
  void* ptr;
  size_t value_size;
  cccl_host_op_fn_ptr_t host_advance_fn_p;

  indirect_iterator_t(cccl_iterator_t& it)
      : ptr{nullptr}
      , value_size{0}
      , host_advance_fn_p{nullptr}
  {
    if (it.type == cccl_iterator_kind_t::CCCL_POINTER)
    {
      value_size = it.value_type.size;
      ptr        = &it.state;
    }
    else
    {
      ptr               = it.state;
      host_advance_fn_p = it.host_advance;
    }
  }

  void* operator&() const
  {
    return ptr;
  }

  template <Increment64 U>
  void operator+=(U offset)
  {
    if (value_size)
    {
      // CCCL_POINTER case
      // ptr is a pointer to pointer we need to increment
      // read the iterator pointer value
      char*& p = *static_cast<char**>(ptr);
      // increment the value
      p += (offset * value_size);
    }
    else
    {
      if (host_advance_fn_p)
      {
        if constexpr (std::is_signed_v<U>)
        {
          cccl_increment_t incr{.signed_offset = offset};
          (*host_advance_fn_p)(ptr, incr);
        }
        else
        {
          cccl_increment_t incr{.unsigned_offset = offset};
          (*host_advance_fn_p)(ptr, incr);
        }
      }
      else
      {
        throw std::runtime_error("Attempt to increment iterator from host, but host advance function is not defined");
      }
    }
  }
};
