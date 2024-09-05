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

void* copy_or_allocate_into_aligned_storage(
  void* space, // Pointer to buffer in which to copy memory
  size_t space_size, // Space available in this buffer (number of bytes available after offset)
  size_t offset, // Offset into memory from which to begin alignment (space+offset == beginning of buffer)
  void* source, // Source buffer to copy from
  size_t src_alignment, // Source's required alignment,
  size_t src_size // Source's size
);

template <typename T, size_t Extra>
struct small_aligned_storage_stack
{
  T stored_obj;
  char space[Extra];
};

template <typename T>
struct small_aligned_storage_allocated
{
  T* stored_obj;
};

// small_aligned_storage<T> manages objects that 'may' hold more data than initially sized for.
// it implements in C++ a managed variable length member located at the end of T.
// This allows packing together unrelated data as sometimes used for kernel parameters.
// This is maybe a dumb idea.
template <typename T, size_t Extra = 8>
struct small_aligned_storage
{
  union
  {
    small_aligned_storage_allocated<T> allocated;
    small_aligned_storage_stack<T, Extra> local;
  };

  bool is_allocated = false;

  constexpr inline small_aligned_storage(T t)
  {
    local.stored_obj = t;
  }

  const small_aligned_storage& operator=(const small_aligned_storage&) = delete;
  const small_aligned_storage& operator=(small_aligned_storage&&)      = delete;
  small_aligned_storage(const small_aligned_storage&)                  = delete;
  small_aligned_storage(small_aligned_storage&&)                       = delete;

  inline small_aligned_storage(T t, void* source, size_t alignment, size_t size)
  {
    auto allocated_space = copy_or_allocate_into_aligned_storage(
      &local, sizeof(local.space), sizeof(local.stored_obj), source, alignment, size);

    if (allocated_space)
    {
      is_allocated          = true;
      allocated             = small_aligned_storage_allocated<T>{(T*) allocated_space};
      *allocated.stored_obj = t;
    }
    else
    {
      local.stored_obj = t;
    }
  }

  inline ~small_aligned_storage()
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
