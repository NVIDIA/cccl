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

// small_aligned_storage<T> manages objects that 'may' hold more data than initially sized for.
// it implements in C++ a managed variable length member located at the end of T.
// This allows packing together unrelated data as sometimes used for kernel parameters.
// This is maybe a dumb idea.
void* copy_or_allocate_into_aligned_storage(
  void* space, // Pointer to buffer in which to copy memory
  size_t space_size, // Space available in this buffer (number of bytes available after offset)
  size_t offset, // Offset into memory from which to begin alignment (space+offset == beginning of buffer)
  void* source, // Source buffer to copy from
  size_t src_alignment, // Source's required alignment,
  size_t src_size // Source's size
);

template <typename T>
struct small_aligned_storage_stack
{
  T stored_obj;
  char space[8];
};

template <typename T>
struct small_aligned_storage_allocated
{
  T* stored_obj;
};

template <typename T>
struct small_aligned_storage
{
  union
  {
    small_aligned_storage_allocated<T> allocated;
    small_aligned_storage_stack<T> local;
  };

  bool is_allocated;

  inline small_aligned_storage(T t, void* source, size_t alignment, size_t size)
  {
    auto allocated_space = copy_or_allocate_into_aligned_storage(
      &local, sizeof(local.space), sizeof(local.stored_obj), source, alignment, size);

    if (allocated_space)
    {
      is_allocated = true;
      allocated    = (small_aligned_storage_allocated<T>*) allocated_space;
    }
  }

  ~small_aligned_storage()
  {
    if (is_allocated)
    {
      free(allocated);
    }
  }

  T* get()
  {
    if (is_allocated)
    {
      return allocated.stored_obj;
    }
    return &local.stored_obj;
  }

  void* get_space()
  {
    if (is_allocated)
    {
      return allocated + 1;
    }
    return &local.space;
  }
};
