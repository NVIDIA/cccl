// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Fill: zero-input broadcast. Calls cub::DeviceTransform::Fill, which goes
// through the unified __transform_internal path -- our trait dispatch hook
// sees the zero-input case but currently has no trait spec for it, so this
// lands on CUB's standard Fill kernel. Wire a tile substitute later if Fill
// becomes a bottleneck.

#include <nvbench/nvbench.cuh>

#include <cub/device/device_transform.cuh>

#include <cuda_runtime.h>

template <typename T>
void fill(nvbench::state& state, nvbench::type_list<T>) {
    const auto n = state.get_int64("Elements{io}");
    T* out; cudaMalloc(&out, n * sizeof(T));
    state.add_element_count(n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Fill(out, n, T(42), launch.get_stream());
    });
    cudaFree(out);
}

// CUB sweeps integral types: int8/16/32/64
using fill_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

NVBENCH_BENCH_TYPES(fill, NVBENCH_TYPE_AXES(fill_types)).set_name("tile_fill")
    .add_int64_power_of_two_axis("Elements{io}", std::vector<nvbench::int64_t>{16, 20, 24, 28, 31});

NVBENCH_MAIN
