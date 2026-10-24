#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

cudax_add_catch2_test(test_target cuco.bloom_filter.add_and_contains ${cudax_target}
  test_add_and_contains.cu
)

foreach (
  test_name
  IN
  ITEMS bulk_ref_equivalence variable_cg layout_equivalence arrow_compat policy
)
  cudax_add_catch2_test(test_target cuco.bloom_filter.${test_name} ${cudax_target}
    test_${test_name}.cu
  )
endforeach()
