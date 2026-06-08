//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

C2H_TEST("cuco make_valid_capacity rounding and validation", "[capacity]")
{
  using probing        = cudax::cuco::double_hashing<1, cudax::cuco::hash<int>>;
  constexpr int bucket = 1;

  static_assert(cudax::cuco::is_double_hashing_v<probing>, "scheme is double hashing");

  // make_valid_capacity rounds up and is idempotent; is_valid_capacity is derived from it
  constexpr auto valid = cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{1000});
  static_assert(valid >= 1000, "rounds up");
  static_assert(cudax::cuco::is_valid_capacity<probing, bucket>(valid), "result is valid");
  static_assert(cudax::cuco::make_valid_capacity<probing, bucket>(valid) == valid, "idempotent");

  // 1000 is not a valid double-hashing capacity; it rounds up to a prime-cycle capacity
  static_assert(!cudax::cuco::is_valid_capacity<probing, bucket>(::cuda::std::size_t{1000}), "1000 is not valid");

  // equal-rounding requests produce the same valid capacity
  static_assert(cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{1000})
                  == cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{1008}),
                "requests that round to the same capacity agree");

  // cuCollections extent_test parity: double hashing, cg_size 2, bucket_size 4.
  // 1234 rounds up to next_prime(ceil(1234 / 8) = 155) = 157, times the stride 8 -> 1256.
  using dh4             = cudax::cuco::double_hashing<2, cudax::cuco::hash<int>>;
  constexpr int bucket4 = 4;
  static_assert(cudax::cuco::make_valid_capacity<dh4, bucket4>(::cuda::std::size_t{1234}) == ::cuda::std::size_t{1256},
                "compile-time valid capacity matches the cuCollections extent test");
  REQUIRE(cudax::cuco::make_valid_capacity<dh4, bucket4>(::cuda::std::size_t{1234}) == ::cuda::std::size_t{1256});

  // a desired load factor outside (0, 1] is rejected
  using lp4   = cudax::cuco::linear_probing<2, cudax::cuco::hash<int>>;
  auto bad_lf = [](double __lf) {
    [[maybe_unused]] auto __r = cudax::cuco::make_valid_capacity<lp4, bucket4>(::cuda::std::size_t{1000}, __lf);
  };
  REQUIRE_THROWS(bad_lf(0.0));
  REQUIRE_THROWS(bad_lf(-0.5));
  REQUIRE_THROWS(bad_lf(1.5));
}
