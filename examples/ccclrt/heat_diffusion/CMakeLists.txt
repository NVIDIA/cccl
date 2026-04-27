# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CCCLRTDemo CUDA)

# Creates a cmake executable target for the main program
add_executable(heat_diffusion heat_diffusion.cu)
target_compile_features(heat_diffusion PUBLIC cuda_std_17)

# "Links" the CCCL Cmake target to the `heat_diffusion` executable. This configures everything needed to use
# CCCL headers, including setting up include paths, compiler flags, etc.
target_link_libraries(heat_diffusion PRIVATE CCCL::CCCL)

# This is only relevant for internal testing and not needed by end users.
if (BUILD_TESTING)
  add_test(NAME heat_diffusion COMMAND heat_diffusion)
endif()
