# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cudax_add_catch2_test(test_target cuco.fixed_capacity_set.insert_and_contains ${cudax_target}
  test_insert_and_contains.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_set.capacity ${cudax_target}
  test_capacity.cu
)

cudax_add_catch2_test(test_target cuco.fixed_capacity_set.ref ${cudax_target}
  test_ref.cu
)
