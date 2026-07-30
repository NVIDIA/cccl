##===----------------------------------------------------------------------===##
##
## Part of libcu++, the C++ Standard Library for your entire system,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

import os

LIBCUDACXX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIBCUDACXX_CMAKE_DIR = os.path.join(LIBCUDACXX_DIR, "cmake")
LIBCUDACXX_CODEGEN_DIR = os.path.join(LIBCUDACXX_DIR, "codegen")
LIBCUDACXX_INCLUDE_DIR = os.path.join(LIBCUDACXX_DIR, "include")
LIBCUDACXX_TEST_DIR = os.path.join(LIBCUDACXX_DIR, "test")

DOCS_DIR = os.path.dirname(LIBCUDACXX_DIR)
DOCS_LIBCUDACXX_DIR = os.path.join(DOCS_DIR, "libcudacxx")
