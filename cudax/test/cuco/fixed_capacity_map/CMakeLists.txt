#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.insert_and_contains ${cudax_target}
  test_insert_and_contains.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.insert_if ${cudax_target}
  test_insert_if.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.find ${cudax_target}
  test_find.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.rebind ${cudax_target}
  test_rebind.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.retrieve_all ${cudax_target}
  test_retrieve_all.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.capacity ${cudax_target}
  test_capacity.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.shared_memory ${cudax_target}
  test_shared_memory.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.misaligned_storage ${cudax_target}
  test_misaligned_storage.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_map.key_sentinel ${cudax_target}
  test_key_sentinel.cu
)

# cudax_add_cuco_fail_test
function(cudax_add_cuco_fail_test target_name_var test_name source)
  set(test_target cudax.test.${test_name})

  cccl_add_executable(
    ${test_target}
    SOURCES ${source}
    NO_METATARGETS
    NO_CLANG_TIDY
  )
  target_link_libraries(${test_target} PRIVATE cudax.compiler_interface)
  cccl_add_xfail_compile_target_test(
    ${test_target}
    SOURCE_FILE "${source}"
    ERROR_REGEX_LABEL "expected-error"
  )

  set(${target_name_var} ${test_target} PARENT_SCOPE)
endfunction()

# Expected-failure compile tests (_fail.cu convention)
cudax_add_cuco_fail_test(test_target cuco.fixed_capacity_map.key_size_fail
  test_key_size_fail.cu
)

cudax_add_cuco_fail_test(test_target cuco.fixed_capacity_map.slot_size_fail
  test_slot_size_fail.cu
)
