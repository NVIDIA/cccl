//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Counting words in a text using a launch kernel
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// determines whether the character is alphabetical
__host__ __device__ bool is_alpha(const char c)
{
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

int main()
{
  // Paragraph from 'The Raven' by Edgar Allan Poe
  // http://en.wikipedia.org/wiki/The_Raven
  const char raw_input[] =
    "  But the raven, sitting lonely on the placid bust, spoke only,\n"
    "  That one word, as if his soul in that one word he did outpour.\n"
    "  Nothing further then he uttered - not a feather then he fluttered -\n"
    "  Till I scarcely more than muttered `Other friends have flown before -\n"
    "  On the morrow he will leave me, as my hopes have flown before.'\n"
    "  Then the bird said, `Nevermore.'\n";

  context ctx;

  size_t text_len = sizeof(raw_input);
  auto ltext      = ctx.logical_data(const_cast<char*>(&raw_input[0]), {text_len});
  auto lcnt       = ctx.logical_data(shape_of<scalar_view<int>>());

  ctx.parallel_for(box(text_len - 1), ltext.read(), lcnt.reduce(reducer::sum<int>{}))
      ->*[] _CCCL_DEVICE(size_t i, auto text, int& s) {
            /* When we have the beginning of a new word, increment the counter */
            if (!is_alpha(text(i)) && is_alpha(text(i + 1)))
            {
              s++;
            }
          };

  int cnt = ctx.wait(lcnt);
  printf("Got %d words.\n", cnt);

  ctx.finalize();

  int ref_cnt = 0;
  for (size_t i = 0; i < sizeof(raw_input) - 1; i++)
  {
    if (!is_alpha(raw_input[i]) && is_alpha(raw_input[i + 1]))
    {
      ref_cnt++;
    }
  }

  _CCCL_ASSERT(cnt == ref_cnt, "Count mismatch");
}
