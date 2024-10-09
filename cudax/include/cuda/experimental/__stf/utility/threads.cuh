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

#include <atomic>
#include <experimental/source_location>
#include <mutex>

namespace cuda::experimental::stf {

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
class single_threaded_section {
public:
    using mutex_type = Mutex;

#ifndef NDEBUG

    explicit single_threaded_section(
            mutex_type& m, ::std::experimental::source_location loc = ::std::experimental::source_location::current())
            : mutex(m) {
        if (mutex.try_lock()) {
            if constexpr (!::std::is_same_v<mutex_type, ::std::recursive_mutex>) {
                // Would not be able to reenter this mutex, so release immediately.
                mutex.unlock();
            }
            return;
        }
        fprintf(stderr, "%s(%u) Error: contested single-threaded section.\n", loc.file_name(), loc.line());
        abort();
    }

    single_threaded_section(mutex_type& m, ::std::adopt_lock_t) noexcept : mutex(m) {}  // calling thread owns mutex

    single_threaded_section(const single_threaded_section&) = delete;
    single_threaded_section& operator=(const single_threaded_section&) = delete;

    ~single_threaded_section() {
        if constexpr (::std::is_same_v<mutex_type, ::std::recursive_mutex>) {
            // Keep the recursive mutex up until destruction.
            mutex.unlock();
        }
    }

private:
    mutex_type& mutex;

#else

    explicit single_threaded_section(mutex_type&) {}
    single_threaded_section(mutex_type&, ::std::adopt_lock_t) noexcept {}
    single_threaded_section(const single_threaded_section&) = delete;
    single_threaded_section& operator=(const single_threaded_section&) = delete;

#endif
};

/**
 * @brief Generates a unique `::std::atomic<unsigned long>` counter object for each literal usage.
 *
 * This templated operator overload allows for the creation of unique counter objects
 * based on template literal strings. Each unique instantiation of this template with a
 * different literal results in a separate counter object. This can be used for counting
 * occurrences or events associated uniquely with compile-time known values.
 *
 * The returned (reference to) `::std::atomic<unsigned long>` can be manipulated with `++`, `--`,
 * and `load` in the usual manner. Each instantiation has its own count, ensuring that counts are
 * isolated between different literal strings.
 *
 * Example usage:
 * @code
 * auto& counterA = "A"_counter;
 * counterA++;
 * auto& counterB = "B"_counter;
 * counterB++;
 * counterB++;
 * ::std::cout << "A count: " << counterA.load() << ::std::endl; // Outputs: A count: 1
 * ::std::cout << "B count: " << counterB.load() << ::std::endl; // Outputs: B count: 2
 * @endcode
 *
 * @tparam C The character type of the template literal.
 * @tparam ... The characters of the template literal.
 *
 * @return A reference to a `::std::atomic<unsigned long>` with static storage duration.
 */
template <typename C, C...>
::std::atomic<unsigned long>& operator""_counter() {
    static ::std::atomic<unsigned long> count { 0 };
    return count;
}

/**
 * @brief Generates an object for tracking the high water mark (maximum value) recorded.
 *
 * This templated user-defined literal operator generates a unique object for each
 * literal usage, which allows for recording and querying the high water mark. The high water mark
 * is the highest value recorded using the `record` method. This utility can be used to monitor
 * peak usages or values in a system, such as maximum memory usage, maximum concurrent connections,
 * or other metrics where the peak value is of interest.
 *
 * The returned type provides static methods to record a new value and query the
 * current high water mark. Each unique instantiation of this template with a different literal
 * results in a separate tracking context, allowing for isolated tracking based on compile-time
 * known values.
 *
 * Example usage:
 * @code
 * auto hwmTracker = "memory_usage"_high_water_mark;
 * hwmTracker.record(1024); // Record a new value
 * ::std::cout << "Current high water mark: " << hwmTracker.load() << ::std::endl;
 * @endcode
 *
 * @tparam C The character type of the template literal.
 * @tparam ... The characters of the template literal.
 *
 * @return An instance of the 'Result' class, which provides static methods for recording new values
 *         and querying the high water mark.
 *
 * @note The function and its returned 'Result' class are thread-safe, ensuring that
 *       concurrent updates from different threads are safely handled.
 */
template <typename C, C...>
auto operator""_high_water_mark() {
    static ::std::atomic<unsigned long> tracker { 0 };

    struct Result {
        static void record(unsigned long v) {
            for (;;) {
                auto previous = tracker.load();
                if (previous >= v || tracker.compare_exchange_weak(previous, v))
                    break;
            }
        }
        static unsigned long load() { return tracker.load(); }
    };

    return Result();
}

}  // namespace cuda::experimental::stf
