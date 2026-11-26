// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deliberately compiled for a different (lower) architecture than main.cu and linked into the same -rdc=true
// (relocatable device code) executable. Linking translation units compiled for different architectures clamps the
// *virtual* architecture metadata of every kernel in the resulting module down to the lowest virtual architecture
// among all linked translation units, even though the actual (real) SASS remains correct for each kernel. See
// NVIDIA/cccl#11403.
__global__ void other_kernel() {}
