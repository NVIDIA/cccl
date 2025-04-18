/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <iostream>

int main()
{
  constexpr std::size_t N = 1000;
  thrust::device_vector<int> data(N, 1);

  const auto result = thrust::reduce(data.cbegin(), data.cend());

  std::cout << "Sum: " << result << std::endl;

  if (result != N)
  {
    std::cerr << "Error: Expected sum of " << N << ", but got " << result << std::endl;
    return 1;
  }

  std::cout << "Detected device system: ";
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  std::cout << "CUDA" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
  std::cout << "TBB" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
  std::cout << "OMP" << std::endl;
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
  std::cout << "CPP" << std::endl;
#endif

  return 0;
}
