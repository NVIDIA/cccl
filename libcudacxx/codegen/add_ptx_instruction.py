#!/usr/bin/env python3
import argparse
from pathlib import Path

docs = Path("docs/libcudacxx/ptx/instructions")
test = Path("libcudacxx/test/libcudacxx/cuda/ptx")
src = Path("libcudacxx/include/cuda/__ptx/instructions")
ptx_header = Path("libcudacxx/include/cuda/ptx")


def add_docs(ptx_instr, url):
    cpp_instr = ptx_instr.replace(".", "_")
    underbar = "=" * len(ptx_instr)

    (docs / f"{cpp_instr}.rst").write_text(
        f""".. _libcudacxx-ptx-instructions-{ptx_instr.replace(".", "-")}:

{ptx_instr}
{underbar}

-  PTX ISA:
   `{ptx_instr} <{url}>`__

.. include:: generated/{cpp_instr}.rst
"""
    )


def add_test(ptx_instr):
    cpp_instr = ptx_instr.replace(".", "_")
    (test / f"ptx.{ptx_instr}.compile.pass.cpp").write_text(
        f"""//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

#include "generated/{cpp_instr}.h"

int main(int, char**)
{{
  return 0;
}}
"""
    )


def add_src(ptx_instr):
    cpp_instr = ptx_instr.replace(".", "_")
    (src / f"{cpp_instr}.h").write_text(
        f"""// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_{cpp_instr.upper()}_H_
#define _CUDA_PTX_{cpp_instr.upper()}_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/__ptx/ptx_helper_functions.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

#include <cuda/__ptx/instructions/generated/{cpp_instr}.h>

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_{cpp_instr.upper()}_H_
"""
    )


def add_include_reminder(ptx_instr):
    cpp_instr = ptx_instr.replace(".", "_")
    txt = ptx_header.read_text()
    reminder = f"""// TODO: #include <cuda/__ptx/instructions/{cpp_instr}.h>"""
    ptx_header.write_text(f"{txt}\n{reminder}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx_instruction", type=str)
    parser.add_argument("url", type=str)

    args = parser.parse_args()

    ptx_instr = args.ptx_instruction
    url = args.url

    # Enable using internal urls in the command-line, to be automatically converted to public URLs.
    if url.startswith("index.html"):
        url = url.replace(
            "index.html",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
        )

    add_test(ptx_instr)
    add_docs(ptx_instr, url)
    add_src(ptx_instr)
    add_include_reminder(ptx_instr)
