/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once
#include <experimental/mdspan>
#ifdef STDBLAS_VERBOSE
#    include <iostream>
#endif
#include "experimental/__p1673_bits/blas1_add_nvhpc.hpp"
#include "experimental/__p1673_bits/blas1_copy_nvhpc.hpp"
#include "experimental/__p1673_bits/blas1_dot_nvhpc.hpp"
#include "experimental/__p1673_bits/blas1_scale_nvhpc.hpp"
#include "experimental/__p1673_bits/blas1_vector_norm2_nvhpc.hpp"
#include "experimental/__p1673_bits/blas2_matrix_vector_product_nvhpc.hpp"
#include "experimental/__p1673_bits/blas2_matrix_vector_solve_nvhpc.hpp"
#include "experimental/__p1673_bits/blas3_matrix_product_nvhpc.hpp"
#include "experimental/__p1673_bits/blas3_matrix_rank_2k_update_nvhpc.hpp"
#include "experimental/__p1673_bits/blas3_matrix_rank_k_update_nvhpc.hpp"
#include "experimental/__p1673_bits/blas3_triangular_matrix_matrix_solve_nvhpc.hpp"
#include "experimental/__p1673_bits/nvhpc_datatypes.hpp"
#include "experimental/__p1673_bits/nvhpc_error.hpp"
#include "experimental/__p1673_bits/nvhpc_helper.hpp"
#include "experimental/__p1673_bits/nvhpc_settings.hpp"
