// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/tuple>

#include <ostream>

struct custom_key_t
{
  int key;

  __host__ __device__ friend bool operator==(const custom_key_t& a, const custom_key_t& b)
  {
    return a.key == b.key;
  }

  __host__ __device__ friend bool operator!=(const custom_key_t& a, const custom_key_t& b)
  {
    return a.key != b.key;
  }

  friend std::ostream& operator<<(std::ostream& os, const custom_key_t& ck)
  {
    return os << "{" << ck.key << "}";
  }
};

struct custom_pair_key_t
{
  int key;
  int payload;

  __host__ __device__ friend bool operator==(const custom_pair_key_t& a, const custom_pair_key_t& b)
  {
    return a.key == b.key && a.payload == b.payload;
  }

  __host__ __device__ friend bool operator!=(const custom_pair_key_t& a, const custom_pair_key_t& b)
  {
    return !(a == b);
  }

  friend std::ostream& operator<<(std::ostream& os, const custom_pair_key_t& cpk)
  {
    return os << "{" << cpk.key << ", " << cpk.payload << "}";
  }
};

struct keys_decomposer_t
{
  __host__ __device__ auto operator()(custom_key_t& k) const -> cuda::std::tuple<int&>
  {
    return {k.key};
  }
};

struct pairs_decomposer_t
{
  __host__ __device__ auto operator()(custom_pair_key_t& k) const -> cuda::std::tuple<int&>
  {
    return {k.key};
  }
};
