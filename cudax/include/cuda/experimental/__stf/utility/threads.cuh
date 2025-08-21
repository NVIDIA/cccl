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
#include <mutex>

namespace cuda::experimental::stf::reserved
{

/**
 * @brief A simple RAII-style wrapper for ensuring single-threaded access to a code section. The interface is the same
 * as for `std::lock_guard`.
 *
 * The `single_threaded_section` class is designed to ensure, in debug (i.e., non-`NDEBUG`) mode, that only one thread
 * can execute a specific code section at a time. There is no guarantee that a passing program has no related bugs,
 * but there are no false positives.
 *
 * If `NDEBUG` is defined, all member functions are defined to do nothing. If `NDEBUG` is not defined, the constructor
 * `assert`s that the mutex can be acquired with no contention.
 *
 * `single_threaded_section` is reentrant even if `Mutex` is not, i.e. a function defining a `single_threaded_section`
 * object may call another function that in turn defines a `single_threaded_section` object on the same mutex.
 *
 * Logic: In a single-threaded section of code, the mutex is never contested. If another thread has the mutex locked
 * (hence the `assert` fails), then this section is not single-threaded.
 *
 * @tparam Mutex The type of mutex to use for synchronization.
 */
template <typename Mutex>
class single_threaded_section
{
public:
  using mutex_type = Mutex;

#ifndef NDEBUG

  explicit single_threaded_section(mutex_type& m,
                                   const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
      : mutex(m)
  {
    if (mutex.try_lock())
    {
      if constexpr (!::std::is_same_v<mutex_type, ::std::recursive_mutex>)
      {
        // Would not be able to reenter this mutex, so release immediately.
        mutex.unlock();
      }
      return;
    }
    fprintf(stderr, "%s(%u) Error: contested single-threaded section.\n", loc.file_name(), loc.line());
    abort();
  }

  single_threaded_section(mutex_type& m, ::std::adopt_lock_t) noexcept
      : mutex(m)
  {} // calling thread owns mutex

  single_threaded_section(const single_threaded_section&)            = delete;
  single_threaded_section& operator=(const single_threaded_section&) = delete;

  ~single_threaded_section()
  {
    if constexpr (::std::is_same_v<mutex_type, ::std::recursive_mutex>)
    {
      // Keep the recursive mutex up until destruction.
      mutex.unlock();
    }
  }

private:
  mutex_type& mutex;

#else

  explicit single_threaded_section(mutex_type&) {}
  single_threaded_section(mutex_type&, ::std::adopt_lock_t) noexcept {}
  single_threaded_section(const single_threaded_section&)            = delete;
  single_threaded_section& operator=(const single_threaded_section&) = delete;

#endif
};

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
