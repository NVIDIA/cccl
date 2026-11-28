// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The main() of NVBench (nvbench::main), compiled into each benchmark of the
// HPX system with the MSVC ABI: there, HPX::auto_wrap_main force-includes
// hpx/hpx_main.hpp into the benchmark's own sources (instead of wrapping main()
// at link time), so main() has to be one of them to run as an HPX thread.

#include <nvbench/main.cuh>

NVBENCH_MAIN
