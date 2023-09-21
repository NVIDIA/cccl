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

#pragma once

//! @file This file includes implementation of CUDA-specific utilities for custom Catch2 main 
//!       When CMake is configured to include all the tests into a single executable, this file
//!       is only included into catch2_runner_helper.cu. When CMake is configured to compile 
//!       each test as a separate binary, this file is included into each test.

#include <iostream>

int device_guard(int device_id)
{
  int device_count {};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
  {
    std::cerr << "Can't query devices number." << std::endl;
    std::exit(-1);
  }

  if (device_id >= device_count || device_id < 0)
  {
    std::cerr << "Invalid device ID: " << device_id << std::endl;
    std::exit(-1);
  }

  return device_id;
}

void set_device(int device_id)
{
  cudaSetDevice(device_guard(device_id));
}
