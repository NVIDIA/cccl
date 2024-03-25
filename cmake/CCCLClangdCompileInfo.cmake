# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Tell cmake to generate a json file of compile commands for clangd:
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink the compile command output to the source dir, where clangd will find it.
set(compile_commands_file "${CMAKE_BINARY_DIR}/compile_commands.json")
set(compile_commands_link "${CMAKE_SOURCE_DIR}/compile_commands.json")
message(STATUS "Creating symlink from ${compile_commands_link} to ${compile_commands_file}...")
cccl_execute_non_fatal_process(COMMAND
  "${CMAKE_COMMAND}" -E rm -f "${compile_commands_link}")
cccl_execute_non_fatal_process(COMMAND
  "${CMAKE_COMMAND}" -E touch "${compile_commands_file}")
cccl_execute_non_fatal_process(COMMAND
  "${CMAKE_COMMAND}" -E create_symlink "${compile_commands_file}" "${compile_commands_link}")
