# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(cccl.examples.ccclrt.kernel_launch_patterns.kernel kernel.cu)
set_target_properties(
  cccl.examples.ccclrt.kernel_launch_patterns.kernel
  PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(
  cccl.examples.ccclrt.kernel_launch_patterns.kernel
  PRIVATE libcudacxx::libcudacxx
)

add_executable(
  cccl.examples.ccclrt.kernel_launch_patterns.kernel_functor
  kernel_functor.cu
)
set_target_properties(
  cccl.examples.ccclrt.kernel_launch_patterns.kernel_functor
  PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(
  cccl.examples.ccclrt.kernel_launch_patterns.kernel_functor
  PRIVATE libcudacxx::libcudacxx
)

# Kernel lambdas require extended lambda support.
if ("${CMAKE_CUDA_COMPILER}" STREQUAL NVIDIA)
  add_executable(
    cccl.examples.ccclrt.kernel_launch_patterns.kernel_lambda
    kernel_lambda.cu
  )
  set_target_properties(
    cccl.examples.ccclrt.kernel_launch_patterns.kernel_lambda
    PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    cccl.examples.ccclrt.kernel_launch_patterns.kernel_lambda
    PRIVATE -extended-lambda
  )
  target_link_libraries(
    cccl.examples.ccclrt.kernel_launch_patterns.kernel_lambda
    PRIVATE libcudacxx::libcudacxx
  )
endif()
