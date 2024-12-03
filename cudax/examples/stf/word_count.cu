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
  return (c >= 'A' && c <= 'z');
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

  // std::cout << "Text sample:" << std::endl;
  // std::cout << raw_input << std::endl;

  // fprintf(stderr, "TEXT size : %ld\n", sizeof(raw_input));

  context ctx;

  auto ltext = ctx.logical_data(const_cast<char*>(&raw_input[0]), {sizeof(raw_input)});

  int cnt   = 0;
  auto lcnt = ctx.logical_data(&cnt, {1});

  auto number_devices = 2;
  auto all_devs       = exec_place::repeat(exec_place::device(0), number_devices);

  auto spec = par(con(128));

  ctx.launch(spec, all_devs, ltext.read(), lcnt.rw())->*[] _CCCL_DEVICE(auto th, auto text, auto cnt) {
    int local_cnt = 0;
    for (size_t i = th.rank(); i < text.size() - 1; i += th.size())
    {
      /* If the thread encounters the beginning of a new word, increment
       * its local counter */
      if (!is_alpha(text(i)) && is_alpha(text(i + 1)))
      {
        local_cnt++;
      }
    }

    // Get a piece of shared memory, and zero it
    __shared__ int block_cnt;
    block_cnt = 0;
    th.inner().sync();

    // In every block, partial sums are gathered, and added to the result
    // by the first thread of the block.
    atomicAdd(&block_cnt, local_cnt);
    th.inner().sync();

    if (th.inner().rank() == 0)
    {
      atomicAdd(&cnt(0), block_cnt);
    }
  };

  ctx.finalize();

  int ref_cnt = 0;
  for (size_t i = 0; i < sizeof(raw_input) - 1; i++)
  {
    if (!is_alpha(raw_input[i]) && is_alpha(raw_input[i + 1]))
    {
      ref_cnt++;
    }
  }

  // fprintf(stderr, "Result : found %d words (expected %d)\n", cnt, ref_cnt);
  EXPECT(cnt == ref_cnt);
}
