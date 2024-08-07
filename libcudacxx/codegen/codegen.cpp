//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <ostream>

#include "generators/compare_and_swap.h"
#include "generators/exchange.h"
#include "generators/fence.h"
#include "generators/fetch_ops.h"
#include "generators/header.h"
#include "generators/ld_st.h"

using namespace std::string_literals;

int main(int argc, char** argv)
{
  std::fstream filestream;

  if (argc == 2)
  {
    filestream.open(argv[1], filestream.out);
  }

  std::ostream& stream = filestream.is_open() ? filestream : std::cout;

  FormatHeader(stream);
  FormatFence(stream);
  FormatLoad(stream);
  FormatStore(stream);
  FormatCompareAndSwap(stream);
  FormatExchange(stream);
  FormatFetchOps(stream);
  FormatTail(stream);

  return 0;
}
