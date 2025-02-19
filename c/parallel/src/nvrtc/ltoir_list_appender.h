//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility> // std::move

#include "command_list.h"
#include <cccl/c/types.h>

struct nvrtc_ltoir_list_appender
{
  nvrtc_ltoir_list& ltoir_list;

  void append(nvrtc_ltoir lto)
  {
    if (lto.ltsz)
    {
      ltoir_list.push_back(std::move(lto));
    }
  }

  void add_iterator_definition(cccl_iterator_t it)
  {
    if (cccl_iterator_kind_t::CCCL_ITERATOR == it.type)
    {
      append({it.advance.ltoir, it.advance.ltoir_size});
      append({it.dereference.ltoir, it.dereference.ltoir_size});
    }
  }
};
