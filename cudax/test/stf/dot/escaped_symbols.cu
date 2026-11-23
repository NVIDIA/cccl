//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Symbols containing `"` or `\` must be escaped in the DOT output
 *
 * Task, logical data and section symbols are written inside `label="..."`;
 * an unescaped quote in a symbol ends the string early and Graphviz rejects
 * the file. Symbols themselves stay verbatim (NVTX, diagnostics): the escaping
 * belongs to the DOT writer alone, and this test checks every path a symbol
 * takes into a label.
 */

#include <cuda/experimental/stf.cuh>

#include <fstream>
#include <random>
#include <sstream>

#if !_CCCL_COMPILER(MSVC)
#  include <unistd.h> // access(), unlink()
#endif // !_CCCL_COMPILER(MSVC)

using namespace cuda::experimental::stf;

int main()
{
#if !_CCCL_COMPILER(MSVC)
  const auto r = ::std::random_device{}();

  char filename[64];
  snprintf(filename, 64, "output_%u.dot", r);
  setenv("CUDASTF_DOT_FILE", filename, 1);

  context ctx;

  // one quote in a data symbol, one in a task symbol, a backslash and a quote in section symbols
  auto lA = ctx.logical_data(shape_of<slice<char>>(64)).set_symbol("A \"quoted\"");
  ctx.task(lA.write()).set_symbol("init \"A\"")->*[](cudaStream_t, auto) {};
  {
    auto guard = ctx.dot_section("sec back\\slash");
    ctx.task(lA.rw()).set_symbol("f1")->*[](cudaStream_t, auto) {};
    {
      auto guard_inner = ctx.dot_section("inner \"sec\"");
      ctx.task(lA.rw()).set_symbol("f2")->*[](cudaStream_t, auto) {};
    }
  }
  ctx.finalize();

  // Call this explicitly for the purpose of the test
  reserved::dot::instance().finish();

  EXPECT(access(filename, F_OK) != -1);

  ::std::ifstream in(filename);
  ::std::stringstream ss;
  ss << in.rdbuf();
  const ::std::string text = ss.str();

  // escaped forms are present ...
  EXPECT(text.find("init \\\"A\\\"") != ::std::string::npos);
  EXPECT(text.find("A \\\"quoted\\\"") != ::std::string::npos);
  EXPECT(text.find("sec back\\\\slash") != ::std::string::npos);
  EXPECT(text.find("inner \\\"sec\\\"") != ::std::string::npos);
  // ... and no raw quote survives inside a label: every `"` in the file is either
  // the delimiter of a quoted string or preceded by a backslash
  bool in_string = false;
  for (size_t i = 0; i < text.size(); i++)
  {
    if (text[i] == '\\' && in_string)
    {
      i++; // skip the escaped character
      continue;
    }
    if (text[i] == '"')
    {
      in_string = !in_string;
    }
  }
  EXPECT(!in_string);

  EXPECT(unlink(filename) == 0);
#endif // !_CCCL_COMPILER(MSVC)
}
