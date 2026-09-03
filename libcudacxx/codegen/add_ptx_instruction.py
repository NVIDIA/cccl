#!/usr/bin/env python3

##===----------------------------------------------------------------------===##
##
## Part of libcu++, the C++ Standard Library for your entire system,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

import argparse
from pathlib import Path

import cccl_paths

docs = (
    Path(cccl_paths.LIBCUDACXX_DIR).parent
    / "docs"
    / "libcudacxx"
    / "ptx"
    / "instructions"
)
test = Path(cccl_paths.LIBCUDACXX_TEST_DIR) / "libcudacxx" / "cuda" / "ptx"
src = Path(cccl_paths.LIBCUDACXX_INCLUDE_DIR) / "cuda" / "__ptx" / "instructions"
ptx_header = Path(cccl_paths.LIBCUDACXX_INCLUDE_DIR) / "cuda" / "ptx"
instr_docs = docs.parent / "instructions.rst"


def write_file(path, content, skip_existing):
    if path.exists():
        if skip_existing:
            print(f"Preserving existing file: {path}")
            return
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    path.write_text(content)


def validate_outputs(cpp_instr, test_instr, skip_existing):
    outputs = [
        docs / f"{cpp_instr}.rst",
        test / f"ptx.{test_instr}.compile.pass.cpp",
        src / f"{cpp_instr}.h",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not skip_existing:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files:\n{paths}")


def add_docs(ptx_instr, cpp_instr, docs_anchor, url, skip_existing):
    underbar = "=" * len(ptx_instr)

    write_file(
        docs / f"{cpp_instr}.rst",
        f""".. _libcudacxx-ptx-instructions-{docs_anchor}:

{ptx_instr}
{underbar}

-  PTX ISA:
   `{ptx_instr} <{url}>`__

.. include:: generated/{cpp_instr}.rst
""",
        skip_existing,
    )


def add_test(cpp_instr, test_instr, skip_existing):
    dst = test / f"ptx.{test_instr}.compile.pass.cpp"
    write_file(
        dst,
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
""",
        skip_existing,
    )


def add_src(cpp_instr, skip_existing):
    write_file(
        src / f"{cpp_instr}.h",
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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

#include <cuda/__ptx/instructions/generated/{cpp_instr}.h>

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_{cpp_instr.upper()}_H_
""",
        skip_existing,
    )


def add_ptx_header_include(cpp_instr):
    txt = ptx_header.read_text()
    include = f"#include <cuda/__ptx/instructions/{cpp_instr}.h>\n"
    if include in txt:
        return

    # just add as first new include. clang-format will sort it in
    idx = txt.index("#include <cuda/__ptx/instructions")
    txt = txt[:idx] + include + txt[idx:]
    ptx_header.write_text(txt)


def add_docs_include(cpp_instr):
    txt = instr_docs.read_text()
    include = f"   instructions/{cpp_instr}\n"
    if include in txt:
        return

    # Append to the existing toctree. Run the script in the desired order.
    first_include = txt.index("   instructions/")
    idx = txt.find("\n\n", first_include)
    txt = txt[:idx] + "\n" + include.rstrip() + txt[idx:]
    instr_docs.write_text(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx_instruction", type=str)
    parser.add_argument("url", type=str)
    parser.add_argument(
        "--cpp-instruction",
        help="Explicit header/test/docs stem (defaults to the PTX instruction with dots replaced by underscores)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Preserve existing scaffold files instead of refusing to overwrite them",
    )

    args = parser.parse_args()

    ptx_instr = args.ptx_instruction
    cpp_instr = args.cpp_instruction or ptx_instr.replace(".", "_")
    test_instr = cpp_instr.replace("_", ".") if args.cpp_instruction else ptx_instr
    docs_anchor = (
        cpp_instr.replace("_", "-")
        if args.cpp_instruction
        else ptx_instr.replace(".", "-")
    )
    url = args.url
    validate_outputs(cpp_instr, test_instr, args.skip_existing)

    # Enable using internal urls in the command-line, to be automatically converted to public URLs.
    if url.startswith("index.html"):
        url = url.replace(
            "index.html",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
        )

    add_test(cpp_instr, test_instr, args.skip_existing)
    add_docs(ptx_instr, cpp_instr, docs_anchor, url, args.skip_existing)
    add_src(cpp_instr, args.skip_existing)
    add_ptx_header_include(cpp_instr)
    add_docs_include(cpp_instr)
