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

#include <type_traits>

#include <cccl/c/types.h>
#include <stdio.h>

struct indirect_arg_t
{
  void* ptr;

  indirect_arg_t(cccl_iterator_t& it)
      : ptr(it.type == cccl_iterator_kind_t::CCCL_POINTER ? &it.state : it.state)
  {}

  indirect_arg_t(cccl_op_t& op)
      : ptr(op.type == cccl_op_kind_t::CCCL_STATELESS ? this : op.state)
  {}

  indirect_arg_t(cccl_value_t& val)
      : ptr(val.state)
  {}

  void* operator&() const
  {
    return ptr;
  }
};

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

  template <typename U,
            std::enable_if_t<std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == sizeof(int64_t), int> = 0>
  void operator+=(U signed_offset)
  {
    if (value_size)
    {
      fflush(stderr);
      printf("SIGNED  Pointer\n");
      fflush(stdout);
      // CCCL_POINTER case
      ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + (signed_offset * value_size));
    }
    else
    {
      fflush(stderr);
      printf("SIGNED  Iterator\n");
      fflush(stdout);
      if (host_advance_fn_p)
      {
        cccl_increment_t incr{.signed_offset = signed_offset};
        (*host_advance_fn_p)(ptr, incr);
      }
    }
  }

  template <typename U,
            std::enable_if_t<std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == sizeof(uint64_t), int> = 0>
  void operator+=(U unsigned_offset)
  {
    if (value_size)
    {
      fflush(stderr);
      printf("UNSIGNED Pointer\n");
      fflush(stdout);
      // CCCL_POINTER case
      ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + (unsigned_offset * value_size));
    }
    else
    {
      fflush(stderr);
      printf("UNSIGNED  Iterator\n");
      fflush(stdout);

      if (host_advance_fn_p)
      {
        cccl_increment_t incr{.unsigned_offset = unsigned_offset};
        (*host_advance_fn_p)(ptr, incr);
      }
    }
  }
};
