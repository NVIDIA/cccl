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
 * @brief This test makes sure we can generate a dot file
 */

#include <cuda/experimental/stf.cuh>

#include <filesystem>

using namespace cuda::experimental::stf;

int main()
{
#if !defined(_CCCL_COMPILER_MSVC)
  // Generate a random dirname
  srand(static_cast<unsigned>(time(nullptr)));
  int r = rand();

  std::string dirname = "dump_" + std::to_string(r);
  setenv("CUDASTF_AUTO_DUMP", "1", 1);
  setenv("CUDASTF_AUTO_DUMP_DIR", dirname.c_str(), 1);

  context ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(64));

  // Ignored logical data
  auto lB = ctx.logical_data(shape_of<slice<int>>(64));
  lB.set_auto_dump(false);

  ctx.parallel_for(lA.shape(), lA.write(), lB.write())->*[] __device__(size_t i, auto a, auto b) {
    a(i) = 42 + i;
    b(i) = 42;
  };
  ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 3;
  };
  ctx.finalize();

  EXPECT(std::filesystem::exists(dirname));
  EXPECT(std::filesystem::is_directory(dirname));

  // Files 0 and 1 should exist
  for (int i = 0; i < 2; i++)
  {
    std::string f = dirname + "/" + std::to_string(i);
    EXPECT(std::filesystem::exists(f));
  }

  // File 2 should not exist
  EXPECT(!std::filesystem::exists(dirname + "/" + std::to_string(2)));

  std::filesystem::remove_all(dirname);
#endif // !_CCCL_COMPILER_MSVC
}
