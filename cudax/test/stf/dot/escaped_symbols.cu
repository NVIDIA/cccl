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
 * Symbols are written inside `label="..."`; an unescaped quote in a symbol ends
 * the string early and Graphviz rejects the file. Symbols themselves stay
 * verbatim (NVTX, diagnostics): the escaping belongs to the DOT writer alone.
 *
 * A symbol enters a label in five places: the task vertex, the logical data
 * listed in it, the prerequisite vertex, the cluster of a section and the vertex
 * a collapsed section becomes. Prerequisite vertices and collapsed sections only
 * exist when CUDASTF_DOT_IGNORE_PREREQS=0 and CUDASTF_DOT_MAX_DEPTH are set, so
 * this test sets both and checks each of the five labels in full.
 */

#include <cuda/experimental/stf.cuh>

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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
  // prerequisite vertices carry the symbol of their task
  setenv("CUDASTF_DOT_IGNORE_PREREQS", "0", 1);
  // sections nested deeper than this become one vertex labeled with the section symbol
  setenv("CUDASTF_DOT_MAX_DEPTH", "1", 1);

  context ctx;

  // Quotes, a backslash next to a quote, a backslash followed by `n`, and a trailing
  // backslash (which would otherwise escape the closing quote of the label).
  auto lA = ctx.logical_data(shape_of<slice<char>>(64)).set_symbol("A \"quoted\"");
  ctx.task(lA.write()).set_symbol("init \"A\"")->*[](cudaStream_t, auto) {};
  // a backslash followed by `n` in a symbol is two characters, not the line break the writer puts between fields
  ctx.task(lA.rw()).set_symbol("two\\nchars")->*[](cudaStream_t, auto) {};
  {
    auto guard = ctx.dot_section("sec back\\slash");
    ctx.task(lA.rw()).set_symbol("mix\\\"\"\\")->*[](cudaStream_t, auto) {};
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

  // Collect the value of every label: from `label="` to the first quote that is
  // not escaped. A well-formed label is followed by `]` (vertex) or a newline
  // (cluster); a raw quote inside a symbol ends the value early and fails that.
  ::std::vector<::std::string> labels;
  const ::std::string key = "label=\"";
  for (size_t pos = text.find(key); pos != ::std::string::npos; pos = text.find(key, pos))
  {
    pos += key.size();
    size_t end = pos;
    while (end < text.size() && text[end] != '"')
    {
      end += (text[end] == '\\') ? 2 : 1;
    }
    EXPECT(end + 1 < text.size());
    const bool closed = text[end + 1] == ']' || text[end + 1] == '\n';
    EXPECT(closed);
    labels.push_back(text.substr(pos, end - pos));
    pos = end + 1;
  }

  const auto has_label = [&labels](const ::std::string& expected) {
    return ::std::find(labels.begin(), labels.end(), expected) != labels.end();
  };

  // task vertex, with the logical data it accesses
  EXPECT(has_label("init \\\"A\\\"\\nA \\\"quoted\\\"(write)(64) "));
  EXPECT(has_label("mix\\\\\\\"\\\"\\\\\\nA \\\"quoted\\\"(rw)(64) "));
  EXPECT(has_label("two\\\\nchars\\nA \\\"quoted\\\"(rw)(64) "));
  // prerequisite vertices
  EXPECT(has_label("Submittedinit \\\"A\\\""));
  EXPECT(has_label("Submittedmix\\\\\\\"\\\"\\\\"));
  EXPECT(has_label("Submittedtwo\\\\nchars"));
  // cluster of a section that is displayed
  EXPECT(has_label("sec back\\\\slash"));
  // vertex of a collapsed section
  EXPECT(text.find("fillcolor=\"white\" label=\"inner \\\"sec\\\"\"]") != ::std::string::npos);

  EXPECT(unlink(filename) == 0);
#endif // !_CCCL_COMPILER(MSVC)
}
