/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/guarantees.cuh>

#include <limits>

#include "catch2_test_helper.h"

TEST_CASE("Determinism guarantees are ordered", "[guarantees]")
{
  auto strong = cub::require(cub::run_to_run_determinism);
  auto weak   = cub::require(cub::nondeterminism);

  REQUIRE(!(strong < strong));
  REQUIRE(!(weak < weak));
  REQUIRE(weak < strong);
  REQUIRE(!(strong < weak));
  STATIC_REQUIRE(cub::nondeterminism < cub::run_to_run_determinism);
  STATIC_REQUIRE(!(cub::run_to_run_determinism < cub::nondeterminism));
}

namespace detail
{
struct max_memory_footprint_t
{
  ::cuda::std::size_t bytes{::cuda::std::numeric_limits<::cuda::std::size_t>::max()};

  max_memory_footprint_t() = default;
  explicit max_memory_footprint_t(::cuda::std::size_t bytes)
      : bytes{bytes}
  {}

  friend inline bool operator<(max_memory_footprint_t lhs, max_memory_footprint_t rhs)
  {
    return lhs.bytes > rhs.bytes;
  }
};

struct max_memory_footprint_fn_t
{
  max_memory_footprint_t operator()(std::size_t bytes) const
  {
    return max_memory_footprint_t{bytes};
  }
};

} // namespace detail

enum class reduce_backend_detector_t
{
  run_to_run_deterministic__small_buffer,
  non_deterministic__small_buffer,
  run_to_run_deterministic__large_buffer,
  non_deterministic__large_buffer
};

class reduce
{
  // Example of exhaustive dispatch
  template <class IteratorT>
  static reduce_backend_detector_t
  sum(IteratorT,
      IteratorT,
      cub::detail::guarantees::run_to_run_deterministic_t,
      detail::max_memory_footprint_t max_footprint)
  {
    if (max_footprint.bytes == ::cuda::std::numeric_limits<std::size_t>::max())
    {
      return reduce_backend_detector_t::run_to_run_deterministic__large_buffer;
    }
    return reduce_backend_detector_t::run_to_run_deterministic__small_buffer;
  }

  template <class IteratorT>
  static reduce_backend_detector_t
  sum(IteratorT,
      IteratorT,
      cub::detail::guarantees::determinism_not_guaranteed_t,
      detail::max_memory_footprint_t max_footprint)
  {
    if (max_footprint.bytes == ::cuda::std::numeric_limits<std::size_t>::max())
    {
      return reduce_backend_detector_t::non_deterministic__large_buffer;
    }
    return reduce_backend_detector_t::non_deterministic__small_buffer;
  }

public:
  static constexpr detail::max_memory_footprint_fn_t max_memory_footprint{};

  // User-visible default guarantees
  using default_guarantees_t =
    cub::detail::guarantees::guarantees_t<cub::detail::guarantees::determinism_not_guaranteed_t,
                                          detail::max_memory_footprint_t>;

  template <class IteratorT, class RequirementsT = default_guarantees_t>
  static reduce_backend_detector_t
  sum(IteratorT begin, IteratorT end, RequirementsT requirements = default_guarantees_t())
  {
    auto guarantees    = cub::detail::requirements::mask(default_guarantees_t(), requirements);
    auto max_footprint = cub::detail::requirements::match<detail::max_memory_footprint_t>(guarantees);
    auto determinism = cub::detail::requirements::match<cub::detail::guarantees::run_to_run_deterministic_t>(guarantees);
    return reduce::sum(begin, end, determinism, max_footprint);
  }
};

TEST_CASE("Max memory footprint is ordered", "[guarantees]")
{
  detail::max_memory_footprint_t();
  auto small = reduce::max_memory_footprint(1);
  auto big   = reduce::max_memory_footprint(2);

  REQUIRE(!(small < small));
  REQUIRE(!(big < big));

  // smaller buffer size requirement is more strict
  REQUIRE(big < small);
  REQUIRE(!(small < big));
}

TEST_CASE("Tag dispatch works", "[guarantees]")
{
  int data[] = {1, 2, 3, 4, 5};

  {
    reduce_backend_detector_t backend = reduce::sum(cuda::std::begin(data), cuda::std::end(data));
    REQUIRE(backend == reduce_backend_detector_t::non_deterministic__large_buffer);
  }

  {
    reduce_backend_detector_t backend =
      reduce::sum(cuda::std::begin(data), cuda::std::end(data), cub::require(cub::run_to_run_determinism));
    REQUIRE(backend == reduce_backend_detector_t::run_to_run_deterministic__large_buffer);
  }

  {
    reduce_backend_detector_t backend =
      reduce::sum(cuda::std::begin(data), cuda::std::end(data), cub::require(reduce::max_memory_footprint(1)));
    REQUIRE(backend == reduce_backend_detector_t::non_deterministic__small_buffer);
  }

  {
    reduce_backend_detector_t backend = reduce::sum(
      cuda::std::begin(data),
      cuda::std::end(data),
      cub::require(cub::run_to_run_determinism, reduce::max_memory_footprint(1)));
    REQUIRE(backend == reduce_backend_detector_t::run_to_run_deterministic__small_buffer);
  }

  {
    reduce_backend_detector_t backend = reduce::sum(
      cuda::std::begin(data), cuda::std::end(data), cub::require(cub::nondeterminism, reduce::max_memory_footprint(1)));
    REQUIRE(backend == reduce_backend_detector_t::non_deterministic__small_buffer);
  }
}

namespace detail
{

template <class T>
struct preserve_partials_in_t
{
  T* d_ptr{};

  preserve_partials_in_t() = default;
  explicit preserve_partials_in_t(T* d_ptr)
      : d_ptr{d_ptr}
  {}
};

struct discard_partials_t
{};

constexpr bool operator<(discard_partials_t, discard_partials_t)
{
  return false;
}

template <class T>
constexpr bool operator<(discard_partials_t, preserve_partials_in_t<T>)
{
  return true;
}

template <class T>
constexpr bool operator<(preserve_partials_in_t<T>, discard_partials_t)
{
  return false;
}

template <class T, class U>
constexpr bool operator<(preserve_partials_in_t<T>, preserve_partials_in_t<U>)
{
  return false;
}

} // namespace detail

enum class scan_backend_detector_t
{
  weak,
  strong 
};

class scan
{
  // Two impplementations of scan:
  // 1. run-to-run deterministic, work efficient, preserving partials, and
  // 2. non-deterministic and memory efficient

  using weak_guarantees_t = cub::detail::guarantees::guarantees_t<cub::detail::guarantees::determinism_not_guaranteed_t,
                                                                  detail::discard_partials_t>;

  using strong_guarantees_t = cub::detail::guarantees::guarantees_t<cub::detail::guarantees::run_to_run_deterministic_t,
                                                                    detail::preserve_partials_in_t<void>>;

  using default_guarantees_t = weak_guarantees_t;

  template <class IteratorT, class RequirementsT>
  static typename ::cuda::std::enable_if<
    cub::detail::guarantees::statically_satisfy<weak_guarantees_t, RequirementsT>::value,
    scan_backend_detector_t>::type
  sum_impl(IteratorT, IteratorT, RequirementsT)
  {
    return scan_backend_detector_t::weak;
  }

  template <class IteratorT, class RequirementsT>
  static typename ::cuda::std::enable_if<
    !cub::detail::guarantees::statically_satisfy<weak_guarantees_t, RequirementsT>::value
      && cub::detail::guarantees::statically_satisfy<strong_guarantees_t, RequirementsT>::value,
    scan_backend_detector_t>::type
  sum_impl(IteratorT, IteratorT, RequirementsT)
  {
    return scan_backend_detector_t::strong;
  }

public:
  template <class T>
  static detail::preserve_partials_in_t<T> preserve_partials_in(T* d_ptr)
  {
    return detail::preserve_partials_in_t<T>{d_ptr};
  }

  template <class IteratorT, class RequirementsT = default_guarantees_t>
  scan_backend_detector_t sum(IteratorT begin, IteratorT end, RequirementsT requirements = default_guarantees_t())
  {
    return sum_impl(begin, end, cub::detail::requirements::mask(default_guarantees_t(), requirements));
  }
};

TEST_CASE("Subsumption works", "[guarantees]") 
{
  int data[] = {1, 2, 3, 4, 5};
  int partials[] = {1, 3, 6, 10, 15};

  {
    scan_backend_detector_t backend = scan().sum(cuda::std::begin(data), cuda::std::end(data));
    REQUIRE(backend == scan_backend_detector_t::weak);
  }

  {
    scan_backend_detector_t backend =
      scan().sum(cuda::std::begin(data), cuda::std::end(data), cub::require(cub::run_to_run_determinism));
    REQUIRE(backend == scan_backend_detector_t::strong);
  }

  {
    scan_backend_detector_t backend =
      scan().sum(cuda::std::begin(data), cuda::std::end(data), cub::require(scan::preserve_partials_in(partials)));
    REQUIRE(backend == scan_backend_detector_t::strong);
  }

  {
    scan_backend_detector_t backend = scan().sum(
      cuda::std::begin(data),
      cuda::std::end(data),
      cub::require(cub::run_to_run_determinism, scan::preserve_partials_in(partials)));
    REQUIRE(backend == scan_backend_detector_t::strong);
  }

  {
    scan_backend_detector_t backend = scan().sum(
      cuda::std::begin(data),
      cuda::std::end(data),
      cub::require(cub::nondeterminism, scan::preserve_partials_in(partials)));
    REQUIRE(backend == scan_backend_detector_t::strong);
  }
}
