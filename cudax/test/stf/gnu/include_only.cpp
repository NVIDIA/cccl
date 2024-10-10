//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "cudastf/__stf/graph/graph_ctx.h"
#include "cudastf/__stf/stream/stream_ctx.h"

using namespace cuda::experimental::stf;

int main() {
    graph_ctx ctx;
    ctx.finalize();

    stream_ctx ctx2;
    ctx2.finalize();
}
