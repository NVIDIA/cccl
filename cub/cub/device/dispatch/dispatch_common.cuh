// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

enum class AliasOption
{
  MayAlias,
  NoAlias
};

enum class SortOrder
{
  Ascending,
  Descending
};

enum class SelectionOption
{
  // KeepRejects: no, MayAlias: no
  Select,
  // KeepRejects: no, MayAlias: yes
  SelectPotentiallyInPlace,
  // KeepRejects: yes, MayAlias: no
  Partition
};

CUB_NAMESPACE_END
