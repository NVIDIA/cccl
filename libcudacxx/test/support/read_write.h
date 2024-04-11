//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCUDACXX_TEST_SUPPORT_READ_WRITE_H
#define LIBCUDACXX_TEST_SUPPORT_READ_WRITE_H

struct value_type_indirection
{
  using value_type = int;
  __host__ __device__ value_type& operator*() const;
};

struct element_type_indirection
{
  using element_type = long;
  __host__ __device__ element_type& operator*() const;
};

struct proxy_indirection
{
  using value_type = int;
  __host__ __device__ value_type operator*() const;
};

struct read_only_indirection
{
  using value_type = int const;
  __host__ __device__ value_type& operator*() const;
};

// doubles as missing_iter_reference_t
struct missing_dereference
{
  using value_type = int;
};

#endif // LIBCUDACXX_TEST_SUPPORT_READ_WRITE_H
