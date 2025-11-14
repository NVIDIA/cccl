// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/util_arch.cuh>

#include <c2h/catch2_test_helper.h>

template <auto V>
struct show;

template <int Nominal4ByteBlockThreads,
          int Nominal4ByteItemsPerThread,
          typename ComputeT,
          int ExpectedBlockThreads,
          int ExpectedItemsPerThread>
void check_mem_bound_scaling()
{
  using mbs = cub::detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, ComputeT>;

  if constexpr (mbs::ITEMS_PER_THREAD != ExpectedItemsPerThread)
  {
    show<mbs::ITEMS_PER_THREAD>::asdf();
  }
  STATIC_REQUIRE(mbs::ITEMS_PER_THREAD == ExpectedItemsPerThread);

  if constexpr (mbs::BLOCK_THREADS != ExpectedBlockThreads)
  {
    show<mbs::BLOCK_THREADS>::asdf();
  }
  STATIC_REQUIRE(mbs::BLOCK_THREADS == ExpectedBlockThreads);
}

C2H_TEST("MemBoundScaling", "[util][arch]")
{
  check_mem_bound_scaling<256, 1, char, 256, 2>();
  check_mem_bound_scaling<256, 16, char, 256, 32>();
  check_mem_bound_scaling<256, 20, char, 256, 40>();
  check_mem_bound_scaling<256, 100, char, 256, 200>();
  check_mem_bound_scaling<256, 500, char, 64, 1000>();
  check_mem_bound_scaling<256, 10000, char, 32, 20000>();

  check_mem_bound_scaling<256, 1, int, 256, 1>();
  check_mem_bound_scaling<256, 16, int, 256, 16>();
  check_mem_bound_scaling<256, 20, int, 256, 20>();
  check_mem_bound_scaling<256, 100, int, 128, 100>();
  check_mem_bound_scaling<256, 500, int, 32, 500>();
  check_mem_bound_scaling<256, 10000, int, 32, 10000>();

  check_mem_bound_scaling<256, 1, int4, 256, 1>();
  check_mem_bound_scaling<256, 16, int4, 256, 4>();
  check_mem_bound_scaling<256, 20, int4, 256, 5>();
  check_mem_bound_scaling<256, 100, int4, 128, 25>();
  check_mem_bound_scaling<256, 500, int4, 32, 125>();
  check_mem_bound_scaling<256, 10000, int4, 32, 2500>();

  using large_t = char[1024];
  check_mem_bound_scaling<256, 1, large_t, 64, 1>();
  check_mem_bound_scaling<256, 16, large_t, 64, 1>();
  check_mem_bound_scaling<256, 20, large_t, 64, 1>();
  check_mem_bound_scaling<256, 100, large_t, 64, 1>();
  check_mem_bound_scaling<256, 500, large_t, 64, 1>();
  check_mem_bound_scaling<256, 10000, large_t, 32, 39>();
}

template <int Nominal4ByteBlockThreads,
          int Nominal4ByteItemsPerThread,
          typename ComputeT,
          int ExpectedBlockThreads,
          int ExpectedItemsPerThread>
void check_reg_bound_scaling()
{
  using mbs = cub::detail::RegBoundScaling<Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, ComputeT>;

  if constexpr (mbs::ITEMS_PER_THREAD != ExpectedItemsPerThread)
  {
    show<mbs::ITEMS_PER_THREAD>::asdf();
  }
  STATIC_REQUIRE(mbs::ITEMS_PER_THREAD == ExpectedItemsPerThread);

  if constexpr (mbs::BLOCK_THREADS != ExpectedBlockThreads)
  {
    show<mbs::BLOCK_THREADS>::asdf();
  }
  STATIC_REQUIRE(mbs::BLOCK_THREADS == ExpectedBlockThreads);
}

C2H_TEST("RegBoundScaling", "[util][arch]")
{
  check_reg_bound_scaling<256, 1, char, 256, 1>();
  check_reg_bound_scaling<256, 16, char, 256, 16>();
  check_reg_bound_scaling<256, 20, char, 256, 20>();
  check_reg_bound_scaling<256, 100, char, 256, 100>();
  check_reg_bound_scaling<256, 500, char, 128, 500>();
  check_reg_bound_scaling<256, 10000, char, 32, 10000>();

  check_reg_bound_scaling<256, 1, int, 256, 1>();
  check_reg_bound_scaling<256, 16, int, 256, 16>();
  check_reg_bound_scaling<256, 20, int, 256, 20>();
  check_reg_bound_scaling<256, 100, int, 128, 100>();
  check_reg_bound_scaling<256, 500, int, 32, 500>();
  check_reg_bound_scaling<256, 10000, int, 32, 10000>();

  check_reg_bound_scaling<256, 1, int4, 256, 1>();
  check_reg_bound_scaling<256, 16, int4, 256, 4>();
  check_reg_bound_scaling<256, 20, int4, 256, 5>();
  check_reg_bound_scaling<256, 100, int4, 128, 25>();
  check_reg_bound_scaling<256, 500, int4, 32, 125>();
  check_reg_bound_scaling<256, 10000, int4, 32, 2500>();

  using large_t = char[1024];
  check_reg_bound_scaling<256, 1, large_t, 64, 1>();
  check_reg_bound_scaling<256, 16, large_t, 64, 1>();
  check_reg_bound_scaling<256, 20, large_t, 64, 1>();
  check_reg_bound_scaling<256, 100, large_t, 64, 1>();
  check_reg_bound_scaling<256, 500, large_t, 64, 1>();
  check_reg_bound_scaling<256, 10000, large_t, 32, 39>();
}
