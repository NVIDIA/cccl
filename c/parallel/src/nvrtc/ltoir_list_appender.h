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

struct nvrtc_linkable_list_appender
{
  nvrtc_linkable_list& linkable_list;

  void append(nvrtc_linkable linkable)
  {
    std::visit(
      [&](auto&& l) {
        if (l.size)
        {
          linkable_list.push_back(std::move(l));
        }
      },
      linkable);
  }

  // New method that handles both types
  void append_operation(cccl_op_t op)
  {
    if (op.code_type == CCCL_OP_LTOIR)
    {
      // LTO-IR goes directly to the link list
      append(nvrtc_linkable{nvrtc_ltoir{op.code, op.code_size}});
    }
    else
    {
      append(nvrtc_linkable{nvrtc_code{op.code, op.code_size}});
    }
  }

  void add_iterator_definition(cccl_iterator_t it)
  {
    if (cccl_iterator_kind_t::CCCL_ITERATOR == it.type)
    {
      append_operation(it.advance); // Use new method
      append_operation(it.dereference); // Use new method
    }
  }
};
