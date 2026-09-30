//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/prime.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <stdexcept>

#include <testing.cuh>

namespace cudax = cuda::experimental;

C2H_TEST("cuco make_valid_capacity rounding and validation", "[capacity]")
{
  using probing                         = cudax::cuco::double_hashing<1, cuda::hash<int>>;
  [[maybe_unused]] constexpr int bucket = 1;

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
  using dh4                              = cudax::cuco::double_hashing<2, cuda::hash<int>>;
  [[maybe_unused]] constexpr int bucket4 = 4;
  static_assert(cudax::cuco::make_valid_capacity<dh4, bucket4>(::cuda::std::size_t{1234}) == ::cuda::std::size_t{1256},
                "compile-time valid capacity matches the cuCollections extent test");
  REQUIRE(cudax::cuco::make_valid_capacity<dh4, bucket4>(::cuda::std::size_t{1234}) == ::cuda::std::size_t{1256});

  // a desired load factor outside (0, 1] is rejected
  using lp4   = cudax::cuco::linear_probing<2, cuda::hash<int>>;
  auto bad_lf = [](double __lf) {
    [[maybe_unused]] auto __r = cudax::cuco::make_valid_capacity<lp4, bucket4>(::cuda::std::size_t{1000}, __lf);
  };
  REQUIRE_THROWS(bad_lf(0.0));
  REQUIRE_THROWS(bad_lf(-0.5));
  REQUIRE_THROWS(bad_lf(1.5));
}

C2H_TEST("cuco capacity rounding at integer boundaries", "[capacity]")
{
  using double_1 = cudax::cuco::double_hashing<1, cuda::hash<int>>;
  using double_2 = cudax::cuco::double_hashing<2, cuda::hash<int>>;
  using linear_1 = cudax::cuco::linear_probing<1, cuda::hash<int>>;
  using linear_2 = cudax::cuco::linear_probing<2, cuda::hash<int>>;
  using linear_4 = cudax::cuco::linear_probing<4, cuda::hash<int>>;

  constexpr auto i32_max = cuda::std::numeric_limits<cuda::std::int32_t>::max();
  constexpr auto i64_max = cuda::std::numeric_limits<cuda::std::int64_t>::max();
  constexpr auto u32_max = cuda::std::numeric_limits<cuda::std::uint32_t>::max();
  constexpr auto u64_max = cuda::std::numeric_limits<cuda::std::uint64_t>::max();

  static_assert(cudax::cuco::make_valid_capacity<double_1, 1>(i32_max) == i32_max);
  static_assert(cudax::cuco::make_valid_capacity<linear_1, 1>(i32_max) == i32_max);
  static_assert(cudax::cuco::make_valid_capacity<linear_1, 1>(i64_max) == i64_max);
  static_assert(cudax::cuco::make_valid_capacity<linear_1, 1>(u32_max) == u32_max);
  static_assert(cudax::cuco::make_valid_capacity<linear_1, 1>(u64_max) == u64_max);
  // Forming the stride must not multiply in int or narrow to the requested capacity's type.
  static_assert(
    cudax::cuco::make_valid_capacity<linear_4, 1 << 30>(cuda::std::uint64_t{1}) == cuda::std::uint64_t{1} << 32);

  REQUIRE(cudax::cuco::make_valid_capacity<double_1, 1>(i32_max) == i32_max);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_1, 1>(i64_max) == i64_max);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_1, 1>(u64_max) == u64_max);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<double_2, 1>(i32_max - 1)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_2, 1>(i32_max)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_2, 1>(i64_max)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_2, 1>(u64_max)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<double_1, 1>(u32_max)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<double_1, 1>(u64_max)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<double_1, 64>(cuda::std::int8_t{1})), std::logic_error);
}

C2H_TEST("cuco load factor conversion at integer boundaries", "[capacity]")
{
  using double_2 = cudax::cuco::double_hashing<2, cuda::hash<int>>;
  using linear_1 = cudax::cuco::linear_probing<1, cuda::hash<int>>;
  using linear_2 = cudax::cuco::linear_probing<2, cuda::hash<int>>;

  constexpr auto i64_min = cuda::std::numeric_limits<cuda::std::int64_t>::min();
  constexpr auto i64_max = cuda::std::numeric_limits<cuda::std::int64_t>::max();
  constexpr auto u64_max = cuda::std::numeric_limits<cuda::std::uint64_t>::max();

  static_assert(cudax::cuco::make_valid_capacity<linear_1, 1>(i64_max, 1.) == i64_max);
  static_assert(cudax::cuco::make_valid_capacity<linear_1, 1>(u64_max, 1.) == u64_max);

  REQUIRE(cudax::cuco::make_valid_capacity<double_2, 1>(0) == 4);
  REQUIRE(cudax::cuco::make_valid_capacity<double_2, 1>(i64_min) == 4);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_2, 1>(0) == 4);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_2, 1>(i64_min) == 2);
  REQUIRE(cudax::cuco::make_valid_capacity<double_2, 1>(i64_min, 0.5) == 4);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_2, 1>(i64_min, 0.5) == 2);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_2, 1>(0, 0.5) == 4);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_1, 1>(i64_max, 1.) == i64_max);
  REQUIRE(cudax::cuco::make_valid_capacity<linear_1, 1>(u64_max, 1.) == u64_max);
  // Fractional scaling retains double precision. Reject an out-of-range rounded estimate before
  // converting it, including when the exact integer result would be representable.
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_1, 1>(i64_max / 2, 0.5)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_1, 1>(u64_max / 2, 0.5)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_1, 1>(i64_max / 2 + 1, 0.5)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_1, 1>(u64_max / 2 + 1, 0.5)), std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_1, 1>(1, cuda::std::numeric_limits<double>::min())),
                    std::logic_error);
  REQUIRE_THROWS_AS((cudax::cuco::make_valid_capacity<linear_1, 1>(1, cuda::std::numeric_limits<double>::quiet_NaN())),
                    std::logic_error);
}

C2H_TEST("cuco next prime honors representable search bounds", "[capacity][prime]")
{
  constexpr auto u64_max       = cuda::std::numeric_limits<cuda::std::uint64_t>::max();
  constexpr auto largest_prime = cuda::std::uint64_t{18446744073709551557ull};

  static_assert(cudax::cuco::detail::__next_prime(0) == 2);
  static_assert(cudax::cuco::detail::__next_prime(0, 1) == 0);
  static_assert(cudax::cuco::detail::__next_prime(2, 2) == 2);
  static_assert(cudax::cuco::detail::__next_prime(14, 16) == 0);
  static_assert(cudax::cuco::detail::__next_prime(14, 17) == 17);
  static_assert(cudax::cuco::detail::__next_prime(18, 17) == 0);
  // A witness divisible by a candidate prime is inconclusive, rather than proof of compositeness.
  static_assert(cudax::cuco::detail::__next_prime(73, 73) == 73);
  static_assert(cudax::cuco::detail::__next_prime(193, 193) == 193);

  REQUIRE(cudax::cuco::detail::__next_prime(14, 16) == 0);
  REQUIRE(cudax::cuco::detail::__next_prime(14, 17) == 17);
  REQUIRE(cudax::cuco::detail::__next_prime(largest_prime) == largest_prime);
  REQUIRE(cudax::cuco::detail::__next_prime(largest_prime + 1) == 0);
  REQUIRE(cudax::cuco::detail::__next_prime(u64_max) == 0);
}
