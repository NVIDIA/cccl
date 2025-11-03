//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/source_location>

#include <atomic>
#include <cstdio>
#include <mutex>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief Generates a unique `std::atomic<unsigned long>` counter object for each literal usage.
 *
 * This templated operator overload allows for the creation of unique counter objects
 * based on template literal strings. Each unique instantiation of this template with a
 * different type results in a separate counter object. This can be used for counting
 * occurrences or events associated uniquely with compile-time known values.
 *
 * Example usage:
 * @code
 * counter<A>.increment();
 * counter<B>.increment();
 * std::cout << "A count: " << counter<A>.load() << std::endl; // Outputs: A count: 1
 * std::cout << "B count: " << counter<B>.load() << std::endl; // Outputs: B count: 2
 * @endcode
 *
 * @tparam T tag type
 */
template <typename T>
class counter
{
public:
  counter() = default;

  static auto load()
  {
    return count.load();
  }

  static auto increment()
  {
    count++;
  }

  static auto decrement()
  {
    count--;
  }

private:
  static inline ::std::atomic<unsigned long> count{0};
};

/**
 * @brief Generates an object for tracking the high water mark (maximum value) recorded.
 *
 * This generates a unique counter for each type `T`, which allows for
 * recording and querying the high water mark. The high water mark is the
 * highest value recorded using the `record` method. This utility can be used
 * to monitor peak usages or values in a system, such as maximum memory usage,
 * maximum concurrent connections, or other metrics where the peak value is of
 * interest.
 *
 * Each unique instantiation of this template with a different type results in
 * a separate tracking context, allowing for isolated tracking based on
 * compile-time known values.
 *
 * Example usage:
 * @code
 * high_water_mark<event>.record(1024); // Record a new value
 * std::cout << "Current high water mark: " << high_water_mark<event>.load() << std::endl;
 * @endcode
 *
 * @tparam T tag type for this counter
 *
 * @note The methods in this class are thread-safe, ensuring that
 *       concurrent updates from different threads are safely handled.
 */
template <typename T>
class high_water_mark
{
public:
  high_water_mark() = default;

  static void record(unsigned long v)
  {
    for (;;)
    {
      auto previous = tracker.load();
      if (previous >= v || tracker.compare_exchange_weak(previous, v))
      {
        break;
      }
    }
  }

  static unsigned long load()
  {
    return tracker.load();
  }

private:
  static inline ::std::atomic<unsigned long> tracker{0};
};
} // namespace cuda::experimental::stf::reserved
