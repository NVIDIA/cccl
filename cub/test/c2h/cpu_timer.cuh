/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda/std/tuple>

#include <chrono>
#include <iostream>
#include <string>

// #define C2H_DEBUG_TIMING

#ifdef C2H_DEBUG_TIMING
#  define C2H_TIME_SECTION_INIT() \
    c2h::cpu_timer _c2h_timer_;   \
    (void) _c2h_timer_
#  define C2H_TIME_SECTION_RESET() _c2h_timer_.reset()
#  define C2H_TIME_SECTION(label)  _c2h_timer_.print_elapsed_seconds_and_reset(label)
#  define C2H_TIME_SCOPE(label)                          \
    c2h::scoped_cpu_timer _c2h_scoped_cpu_timer_(label); \
    (void) _c2h_scoped_cpu_timer_
#else
#  define C2H_TIME_SECTION_INIT()  /* no-op */ []() {}()
#  define C2H_TIME_SECTION_RESET() /* no-op */ []() {}()
#  define C2H_TIME_SECTION(label)  /* no-op */ []() {}()
#  define C2H_TIME_SCOPE(label)    /* no-op */ []() {}()
#endif

namespace c2h
{

class cpu_timer
{
  std::chrono::high_resolution_clock::time_point m_start;

public:
  cpu_timer()
      : m_start(std::chrono::high_resolution_clock::now())
  {}

  void reset()
  {
    m_start = std::chrono::high_resolution_clock::now();
  }

  int elapsed_ms() const
  {
    auto duration = std::chrono::high_resolution_clock::now() - m_start;
    auto ms       = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    return static_cast<int>(ms.count());
  }

  std::uint64_t elapsed_us() const
  {
    auto duration = std::chrono::high_resolution_clock::now() - m_start;
    auto us       = std::chrono::duration_cast<std::chrono::microseconds>(duration);
    return static_cast<std::uint64_t>(us.count());
  }

  void print_elapsed_seconds(const std::string& label)
  {
    printf("%0.6f s: %s\n", this->elapsed_us() / 1000000.f, label.c_str());
  }

  void print_elapsed_seconds_and_reset(const std::string& label)
  {
    this->print_elapsed_seconds(label);
    this->reset();
  }
};

class scoped_cpu_timer
{
  cpu_timer m_timer;
  std::string m_label;

public:
  explicit scoped_cpu_timer(std::string label)
      : m_label(std::move(label))
  {}

  ~scoped_cpu_timer()
  {
    m_timer.print_elapsed_seconds(m_label);
  }
};

} // namespace c2h
