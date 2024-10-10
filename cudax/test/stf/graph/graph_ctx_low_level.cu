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
 * @brief Test low level API of the graph context
 */

#include "cudastf/__stf/graph/graph_ctx.h"

using namespace cuda::experimental::stf;

int main(int argc, char** argv) {
    graph_ctx ctx;

    double X[1024], Y[1024];
    auto handle_X = ctx.logical_data(X);
    auto handle_Y = ctx.logical_data(Y);

    for (int k = 0; k < 10; k++) {
        graph_task<> t = ctx.task();
        t.add_deps(handle_X.rw());
        t.start();
        cudaGraphNode_t n;
        cuda_safe_call(cudaGraphAddEmptyNode(&n, t.get_graph(), nullptr, 0));
        t.end();
    }

    graph_task<> t2 = ctx.task();
    t2.add_deps(handle_X.read(), handle_Y.rw());
    t2.start();
    cudaGraphNode_t n2;
    cuda_safe_call(cudaGraphAddEmptyNode(&n2, t2.get_graph(), nullptr, 0));
    t2.end();

    ctx.submit();

    if (argc > 1) {
        std::cout << "Generating DOT output in " << argv[1] << std::endl;
        ctx.print_to_dot(argv[1]);
    }

    ctx.finalize();
}
