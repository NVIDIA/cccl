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

#include <memory>
#include <variant>

void* copy_or_allocate_into_aligned_storage(
  void* space, // Pointer to buffer in which to copy memory
  size_t space_size, // Space available in this buffer (number of bytes available after offset)
  size_t offset, // Offset into memory from which to begin alignment (space+offset == beginning of buffer)
  void* source, // Source buffer to copy from
  size_t src_alignment, // Source's required alignment,
  size_t src_size // Source's size
);

template <typename T, size_t Extra>
struct small_vector_stack
{
  T stored_obj;
  char space[Extra];
};

template <typename T, size_t Extra = sizeof(T)>
struct small_vector
{
  std::variant<std::unique_ptr<void*>, small_vector_stack<T, Extra>> storage;

  constexpr inline small_vector(T t)
      : storage(small_vector_stack<T, Extra>{T})
  {}

  const small_vector& operator=(const small_vector&) = delete;
  const small_vector& operator=(small_vector&&)      = delete;
  small_vector(const small_vector&)                  = delete;
  small_vector(small_vector&&)                       = delete;

  inline small_vector(T t, void* source, size_t alignment, size_t size)
  {
    auto allocated_space = copy_or_allocate_into_aligned_storage(
      &local, sizeof(local.space), sizeof(local.stored_obj), source, alignment, size);

    if (allocated_space)
    {
      is_allocated          = true;
      allocated             = small_vector_allocated<T>{(T*) allocated_space};
      *allocated.stored_obj = t;
    }
    else
    {
      local.stored_obj = t;
    }
  }

  inline ~small_vector()
  {
    if (is_allocated)
    {
      free(allocated.stored_obj);
    }
  }

  inline T* get()
  {
    if (is_allocated)
    {
      return allocated.stored_obj;
    }
    return &local.stored_obj;
  }

  inline void* get_space()
  {
    if (is_allocated)
    {
      return allocated + 1;
    }
    return &local.space;
  }
};
