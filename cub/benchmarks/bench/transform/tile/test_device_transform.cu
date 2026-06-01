// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Standalone correctness tests for cub_tile::DeviceTransform.
// Sits next to the benches so it builds against the same tileiras
// toolchain and does not pretend to be part of CCCL's catch2 suite.

#include "device_transform.cuh"

#include <cuda_runtime.h>
#include <cuda/std/tuple>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace {

int g_failures = 0;

#define CUDA_CHECK(expr)                                                                  \
    do {                                                                                  \
        cudaError_t _e = (expr);                                                          \
        if (_e != cudaSuccess) {                                                          \
            std::fprintf(stderr, "%s:%d  CUDA error: %s\n", __FILE__, __LINE__,           \
                         cudaGetErrorString(_e));                                         \
            std::exit(2);                                                                 \
        }                                                                                 \
    } while (0)

template <typename T>
bool eq(T a, T b) { return a == b; }
inline bool eq(float a, float b) {
    float diff = std::fabs(a - b);
    float tol  = 1e-5f * std::fmax(std::fabs(a), std::fabs(b));
    return diff <= std::fmax(tol, 1e-6f);
}

template <typename T>
void expect_array(const char* name, const std::vector<T>& got, const std::vector<T>& want) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "[FAIL] %s: size %zu != %zu\n", name, got.size(), want.size());
        ++g_failures;
        return;
    }
    int mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!eq(got[i], want[i])) {
            if (mismatches < 4) {
                std::fprintf(stderr, "[FAIL] %s: idx=%zu got=%g want=%g\n",
                             name, i, double(got[i]), double(want[i]));
            }
            ++mismatches;
        }
    }
    if (mismatches) { ++g_failures; std::fprintf(stderr, "[FAIL] %s: %d mismatches\n", name, mismatches); }
    else            { std::printf("[ OK ] %s (n=%zu)\n", name, got.size()); }
}

struct identity_op { template <class A> __tile__ auto operator()(A a) const { return a; } };
struct square_op   { template <class A> __tile__ auto operator()(A a) const { return a * a; } };
struct add_op      { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a + b; } };
struct mul_op      { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a * b; } };

template <typename T>
std::vector<T> ramp(int64_t n, T start = T{0}, T step = T{1}) {
    std::vector<T> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = T(start + step * T(i));
    return v;
}

template <typename T>
struct GpuVec {
    T* d{};
    int64_t n{};
    explicit GpuVec(int64_t n) : n(n) { CUDA_CHECK(cudaMalloc(&d, n * sizeof(T))); }
    explicit GpuVec(const std::vector<T>& h) : GpuVec(int64_t(h.size())) {
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~GpuVec() { if (d) cudaFree(d); }
    std::vector<T> to_host() const {
        std::vector<T> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
};

template <typename T>
void test_identity(int64_t n) {
    auto h_in = ramp<T>(n, T{1}, T{1});
    GpuVec<T> dx(h_in), dy(n);
    CUDA_CHECK(cub_tile::DeviceTransform::Transform(
        ::cuda::std::make_tuple(dx.d), dy.d, n, identity_op{}));
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_array("identity", dy.to_host(), h_in);
}

template <typename T>
void test_square(int64_t n) {
    auto h_in = ramp<T>(n, T{1}, T{1});
    std::vector<T> want(n);
    for (int64_t i = 0; i < n; ++i) want[i] = h_in[i] * h_in[i];
    GpuVec<T> dx(h_in), dy(n);
    CUDA_CHECK(cub_tile::DeviceTransform::Transform(
        ::cuda::std::make_tuple(dx.d), dy.d, n, square_op{}));
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_array("square", dy.to_host(), want);
}

template <typename T>
void test_add(int64_t n) {
    auto ha = ramp<T>(n, T{1},   T{1});
    auto hb = ramp<T>(n, T{100}, T{2});
    std::vector<T> want(n);
    for (int64_t i = 0; i < n; ++i) want[i] = ha[i] + hb[i];
    GpuVec<T> da(ha), db(hb), dc(n);
    CUDA_CHECK(cub_tile::DeviceTransform::Transform(
        ::cuda::std::make_tuple(da.d, db.d), dc.d, n, add_op{}));
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_array("add", dc.to_host(), want);
}

template <typename T>
void test_mul(int64_t n) {
    auto ha = ramp<T>(n, T{1}, T{1});
    auto hb = ramp<T>(n, T{3}, T{1});
    std::vector<T> want(n);
    for (int64_t i = 0; i < n; ++i) want[i] = ha[i] * hb[i];
    GpuVec<T> da(ha), db(hb), dc(n);
    CUDA_CHECK(cub_tile::DeviceTransform::Transform(
        ::cuda::std::make_tuple(da.d, db.d), dc.d, n, mul_op{}));
    CUDA_CHECK(cudaDeviceSynchronize());
    expect_array("mul", dc.to_host(), want);
}

template <typename T>
void test_fill(int64_t n, T value) {
    GpuVec<T> dy(n);
    CUDA_CHECK(cub_tile::DeviceTransform::Fill(dy.d, n, value));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<T> want(n, value);
    expect_array("fill", dy.to_host(), want);
}

} // namespace

int main() {
    // pow-2, multiple tiles
    test_identity<std::int32_t>(4096);
    test_square<std::int32_t>(2048);
    test_add<float>(4096);
    test_mul<float>(2048);
    test_fill<std::int32_t>(1024, 42);

    // non-pow-2 num_items (still multiple of 16 to satisfy assume_divisible<16>)
    test_add<float>(4112);     // 16 * 257
    test_fill<std::int32_t>(1008, -7);   // 16 * 63

    // single full tile and below-one-tile (still >=16, div by 16)
    test_square<std::int32_t>(16);
    test_add<float>(64);

    if (g_failures) {
        std::fprintf(stderr, "\n%d test group(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nall tests passed\n");
    return 0;
}
