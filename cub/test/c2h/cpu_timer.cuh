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

#include <chrono>
#include <iostream>
#include <string>

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
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    return static_cast<int>(ms.count());
  }

  void print_elapsed_seconds(const std::string& label)
  {
    std::cout << label << ": " << (this->elapsed_ms() / 1000.f) << "s\n";
  }

    void print_elapsed_seconds_and_reset(const std::string& label)
  {
    this->print_elapsed_seconds(label);
    this->reset();
  }
};

}
