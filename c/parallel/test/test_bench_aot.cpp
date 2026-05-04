//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Benchmark: JIT vs link_ltoir vs Full AoT first-execution latency
//
// For each algorithm, measures wall-clock time for the complete first execution:
//   JIT        = compile(full op) + load() + execute() + sync
//   link_ltoir = link_ltoir(pre-compiled kernel LTOIR + op LTOIR) + execute() + sync
//   Full AoT   = load(pre-compiled cubin) + execute() + sync
//
// Pre-compile steps (untimed, simulating build-server work):
//   - kernel-only compile  → kernel LTOIR stored in build struct (for link_ltoir)
//   - full-op compile      → cubin stored in build struct (for Full AoT)
//
// Expected ordering: Full AoT < link_ltoir < JIT

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "test_util.h"
#include <c2h/catch2_test_helper.h>
#include <cccl/c/merge_sort.h>
#include <cccl/c/reduce.h>
#include <cccl/c/scan.h>

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0)
{
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static void print_header(int sm_major, int sm_minor, unsigned long long n)
{
  std::printf("\n--- AoT latency benchmark  (SM %d%d, N=%llu) ---\n", sm_major, sm_minor, n);
  std::printf("  %-30s  %12s   %20s   %20s\n", "algorithm", "JIT", "link_ltoir", "Full AoT");
  std::printf("  %s\n", std::string(92, '-').c_str());
  std::fflush(stdout);
}

static void print_row(const char* algo, double jit_ms, double link_ms, double full_ms)
{
  std::printf(
    "  %-30s  %7.1f ms      %7.3f ms (%4.0fx)   %7.3f ms (%4.0fx)\n",
    algo,
    jit_ms,
    link_ms,
    jit_ms / link_ms,
    full_ms,
    jit_ms / full_ms);
  std::fflush(stdout);
}

// Runs execute(nullptr, &size) to query temp storage, allocates, then runs execute(ptr, &size).
// Includes cudaDeviceSynchronize(). Suitable for use inside a timed block.
#define WITH_TEMP(execute)                                     \
  do                                                           \
  {                                                            \
    size_t _tmp_bytes = 0;                                     \
    REQUIRE(CUDA_SUCCESS == (execute) (nullptr, &_tmp_bytes)); \
    void* _d_tmp = nullptr;                                    \
    if (_tmp_bytes > 0)                                        \
    {                                                          \
      REQUIRE(cudaSuccess == cudaMalloc(&_d_tmp, _tmp_bytes)); \
    }                                                          \
    REQUIRE(CUDA_SUCCESS == (execute) (_d_tmp, &_tmp_bytes));  \
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());           \
    if (_d_tmp)                                                \
    {                                                          \
      REQUIRE(cudaSuccess == cudaFree(_d_tmp));                \
    }                                                          \
  } while (false)

C2H_TEST("AoT vs JIT first-execution latency", "[bench_aot]")
{
  cudaDeviceProp prop{};
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&prop, 0));
  const int cc_major = prop.major;
  const int cc_minor = prop.minor;

  constexpr uint64_t N = 1 << 20; // 1 M elements

  print_header(cc_major, cc_minor, (unsigned long long) N);

  // ── reduce ─────────────────────────────────────────────────────────────────
  // Input: N x 1s. Op: plus. Init: 0. Expected output: N.
  {
    const std::vector<int32_t> h_in(N, 1);
    pointer_t<int32_t> d_in(h_in), d_out(1);
    value_t<int32_t> init_val{0};
    cccl_iterator_t in_ = d_in, out_ = d_out;
    cccl_value_t init_ = init_val;

    operation_t op_full = make_operation("op", get_reduce_op(CCCL_INT32));
    cccl_op_t op_c      = op_full;

    cccl_op_t op_ko{};
    op_ko.type      = CCCL_STATELESS;
    op_ko.name      = "op";
    op_ko.code      = nullptr;
    op_ko.code_size = 0;
    op_ko.code_type = CCCL_OP_LTOIR;

    // ── Pre-compile: kernel-only (for link_ltoir timed path) ─────────────────
    cccl_device_reduce_build_result_t link_build{};
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_reduce_compile(
        &link_build,
        in_,
        out_,
        op_ko,
        init_,
        CCCL_RUN_TO_RUN,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(link_build.kernel_ltoir != nullptr);

    // ── Pre-compile: full op into cubin (for Full AoT timed path) ────────────
    cccl_device_reduce_build_result_t full_build{};
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_reduce_compile(
        &full_build,
        in_,
        out_,
        op_c,
        init_,
        CCCL_RUN_TO_RUN,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(full_build.cubin != nullptr);
    REQUIRE(full_build.library == nullptr);

    // ── JIT ──────────────────────────────────────────────────────────────────
    cccl_device_reduce_build_result_t jit_build{};
    auto t0 = clk::now();
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_reduce_compile(
        &jit_build,
        in_,
        out_,
        op_c,
        init_,
        CCCL_RUN_TO_RUN,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(CUDA_SUCCESS == cccl_device_reduce_load(&jit_build));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_reduce(jit_build, tmp, sz, in_, out_, N, op_c, init_, nullptr);
    });
    const double jit_ms = ms_since(t0);
    REQUIRE(d_out[0] == int32_t(N));
    REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&jit_build));

    // ── link_ltoir ───────────────────────────────────────────────────────────
    const void* op_blob = op_full.code.data();
    size_t op_size      = op_full.code.size();
    t0                  = clk::now();
    REQUIRE(CUDA_SUCCESS == cccl_device_reduce_link_ltoir(&link_build, &op_blob, &op_size, 1));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_reduce(link_build, tmp, sz, in_, out_, N, op_c, init_, nullptr);
    });
    const double link_ms = ms_since(t0);
    REQUIRE(d_out[0] == int32_t(N));
    REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&link_build));

    // ── Full AoT ─────────────────────────────────────────────────────────────
    t0 = clk::now();
    REQUIRE(CUDA_SUCCESS == cccl_device_reduce_load(&full_build));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_reduce(full_build, tmp, sz, in_, out_, N, op_c, init_, nullptr);
    });
    const double full_ms = ms_since(t0);
    REQUIRE(d_out[0] == int32_t(N));
    REQUIRE(CUDA_SUCCESS == cccl_device_reduce_cleanup(&full_build));

    print_row("reduce (int32)", jit_ms, link_ms, full_ms);
  }

  // ── scan ───────────────────────────────────────────────────────────────────
  // Input: N x 1s. Op: plus. Exclusive, init=0.
  // Expected: output[0]=0, output[N-1]=N-1.
  {
    const std::vector<int32_t> h_in(N, 1);
    pointer_t<int32_t> d_in(h_in), d_out(N);
    value_t<int32_t> init_val{0};
    cccl_iterator_t in_ = d_in, out_ = d_out;
    cccl_value_t init_      = init_val;
    cccl_type_info accum_ti = get_type_info<int32_t>();

    operation_t op_full = make_operation("op", get_reduce_op(CCCL_INT32));
    cccl_op_t op_c      = op_full;

    cccl_op_t op_ko{};
    op_ko.type      = CCCL_STATELESS;
    op_ko.name      = "op";
    op_ko.code      = nullptr;
    op_ko.code_size = 0;
    op_ko.code_type = CCCL_OP_LTOIR;

    // ── Pre-compile: kernel-only ──────────────────────────────────────────────
    cccl_device_scan_build_result_t link_build{};
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_scan_compile(
        &link_build,
        in_,
        out_,
        op_ko,
        accum_ti,
        /*force_inclusive=*/false,
        CCCL_VALUE_INIT,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(link_build.kernel_ltoir != nullptr);

    // ── Pre-compile: full op into cubin ───────────────────────────────────────
    cccl_device_scan_build_result_t full_build{};
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_scan_compile(
        &full_build,
        in_,
        out_,
        op_c,
        accum_ti,
        /*force_inclusive=*/false,
        CCCL_VALUE_INIT,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(full_build.cubin != nullptr);
    REQUIRE(full_build.library == nullptr);

    // ── JIT ──────────────────────────────────────────────────────────────────
    cccl_device_scan_build_result_t jit_build{};
    auto t0 = clk::now();
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_scan_compile(
        &jit_build,
        in_,
        out_,
        op_c,
        accum_ti,
        /*force_inclusive=*/false,
        CCCL_VALUE_INIT,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(CUDA_SUCCESS == cccl_device_scan_load(&jit_build));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_exclusive_scan(jit_build, tmp, sz, in_, out_, N, op_c, init_, nullptr);
    });
    const double jit_ms = ms_since(t0);
    REQUIRE(d_out[0] == int32_t(0));
    REQUIRE(d_out[int(N) - 1] == int32_t(N - 1));
    REQUIRE(CUDA_SUCCESS == cccl_device_scan_cleanup(&jit_build));

    // ── link_ltoir ───────────────────────────────────────────────────────────
    const void* op_blob = op_full.code.data();
    size_t op_size      = op_full.code.size();
    t0                  = clk::now();
    REQUIRE(CUDA_SUCCESS == cccl_device_scan_link_ltoir(&link_build, &op_blob, &op_size, 1));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_exclusive_scan(link_build, tmp, sz, in_, out_, N, op_c, init_, nullptr);
    });
    const double link_ms = ms_since(t0);
    REQUIRE(d_out[0] == int32_t(0));
    REQUIRE(d_out[int(N) - 1] == int32_t(N - 1));
    REQUIRE(CUDA_SUCCESS == cccl_device_scan_cleanup(&link_build));

    // ── Full AoT ─────────────────────────────────────────────────────────────
    t0 = clk::now();
    REQUIRE(CUDA_SUCCESS == cccl_device_scan_load(&full_build));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_exclusive_scan(full_build, tmp, sz, in_, out_, N, op_c, init_, nullptr);
    });
    const double full_ms = ms_since(t0);
    REQUIRE(d_out[0] == int32_t(0));
    REQUIRE(d_out[int(N) - 1] == int32_t(N - 1));
    REQUIRE(CUDA_SUCCESS == cccl_device_scan_cleanup(&full_build));

    print_row("scan (int32, excl.)", jit_ms, link_ms, full_ms);
  }

  // ── merge_sort ─────────────────────────────────────────────────────────────
  // Input keys: [N, N-1, ..., 2, 1] (descending). Keys-only.
  // Expected: keys_out[0]=1, keys_out[N-1]=N.
  {
    std::vector<int32_t> h_keys(N);
    std::iota(h_keys.rbegin(), h_keys.rend(), 1);
    pointer_t<int32_t> d_keys_in(h_keys), d_keys_out(N);

    cccl_iterator_t null_items{};
    null_items.type       = cccl_iterator_kind_t::CCCL_POINTER;
    null_items.state      = nullptr;
    null_items.value_type = get_type_info<int32_t>();

    operation_t op_full = make_operation("op", get_merge_sort_op(CCCL_INT32));
    cccl_op_t op_c      = op_full;
    cccl_iterator_t ki = d_keys_in, ko = d_keys_out;

    cccl_op_t op_ko{};
    op_ko.type      = CCCL_STATELESS;
    op_ko.name      = "op";
    op_ko.code      = nullptr;
    op_ko.code_size = 0;
    op_ko.code_type = CCCL_OP_LTOIR;

    // ── Pre-compile: kernel-only ──────────────────────────────────────────────
    cccl_device_merge_sort_build_result_t link_build{};
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_merge_sort_compile(
        &link_build,
        ki,
        null_items,
        ko,
        null_items,
        op_ko,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(link_build.kernel_ltoir != nullptr);

    // ── Pre-compile: full op into cubin ───────────────────────────────────────
    cccl_device_merge_sort_build_result_t full_build{};
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_merge_sort_compile(
        &full_build,
        ki,
        null_items,
        ko,
        null_items,
        op_c,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(full_build.cubin != nullptr);
    REQUIRE(full_build.library == nullptr);

    // ── JIT ──────────────────────────────────────────────────────────────────
    cccl_device_merge_sort_build_result_t jit_build{};
    auto t0 = clk::now();
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_merge_sort_compile(
        &jit_build,
        ki,
        null_items,
        ko,
        null_items,
        op_c,
        cc_major,
        cc_minor,
        TEST_CUB_PATH,
        TEST_THRUST_PATH,
        TEST_LIBCUDACXX_PATH,
        TEST_CTK_PATH,
        nullptr));
    REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_load(&jit_build));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_merge_sort(jit_build, tmp, sz, ki, null_items, ko, null_items, N, op_c, nullptr);
    });
    const double jit_ms = ms_since(t0);
    REQUIRE(d_keys_out[0] == int32_t(1));
    REQUIRE(d_keys_out[int(N) - 1] == int32_t(N));
    REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_cleanup(&jit_build));

    // ── link_ltoir ───────────────────────────────────────────────────────────
    REQUIRE(cudaSuccess == cudaMemcpy(d_keys_in.ptr, h_keys.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    const void* op_blob = op_full.code.data();
    size_t op_size      = op_full.code.size();
    t0                  = clk::now();
    REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_link_ltoir(&link_build, &op_blob, &op_size, 1));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_merge_sort(link_build, tmp, sz, ki, null_items, ko, null_items, N, op_c, nullptr);
    });
    const double link_ms = ms_since(t0);
    REQUIRE(d_keys_out[0] == int32_t(1));
    REQUIRE(d_keys_out[int(N) - 1] == int32_t(N));
    REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_cleanup(&link_build));

    // ── Full AoT ─────────────────────────────────────────────────────────────
    REQUIRE(cudaSuccess == cudaMemcpy(d_keys_in.ptr, h_keys.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    t0 = clk::now();
    REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_load(&full_build));
    WITH_TEMP([&](void* tmp, size_t* sz) {
      return cccl_device_merge_sort(full_build, tmp, sz, ki, null_items, ko, null_items, N, op_c, nullptr);
    });
    const double full_ms = ms_since(t0);
    REQUIRE(d_keys_out[0] == int32_t(1));
    REQUIRE(d_keys_out[int(N) - 1] == int32_t(N));
    REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_cleanup(&full_build));

    print_row("merge_sort (int32, keys)", jit_ms, link_ms, full_ms);
  }

  std::printf("  %s\n\n", std::string(92, '-').c_str());
}
