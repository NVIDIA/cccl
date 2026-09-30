//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef DETAIL_H
#define DETAIL_H

/// @file
/// Supporting details for the image pipeline example: synthetic image
/// generation and printing/output helpers.  These are not part of the
/// pipeline itself — in a real application, image generation would be
/// replaced by actual data loading, and the printing would be replaced
/// by your application's logging.

#include <image_pipeline.h>

// ── Image generation ─────────────────────────────────────────────────

/// Generate a synthetic space observation on the GPU, tile by tile,
/// and produce a downscaled input preview.
/// In a real application this would be replaced by loading actual data.
void generate_image(cuda::stream_ref stream, tile_buffers& bufs, int num_tiles, cuda::std::span<pixel_t> host_preview);

// ── Printing / output helpers ────────────────────────────────────────

void print_device_info(cuda::device_ref dev, cuda::arch_traits_t traits, size_t total_mem);
void print_tile_plan(int tile_rows, int tile_alignment, int num_tiles, size_t budget, size_t total_mem);
void print_allocation_info(size_t device_total, size_t gpu_budget, size_t tile_pixels, int tile_rows);
void print_pool_stats(tile_buffers& bufs);

struct iqr_result
{
  int p25, p75;
  [[nodiscard]] int width() const noexcept
  {
    return p75 - p25;
  }
};

[[nodiscard]] iqr_result compute_iqr(cuda::std::span<const histogram_count_t> hist, size_t total);
void print_pass_stats(double ms, long long total_selected, double mean_selected, float global_min, float global_max);
void print_sanity_check(iqr_result orig, iqr_result eq);
void print_summary(int num_tiles, int tile_rows, double pass1_ms, double pass2_ms, bool ok);
[[nodiscard]] bool write_bmp(const char* filename, cuda::std::span<const pixel_t> data, int width, int height);

/// Downscale a device tile using CUB BlockReduce and copy the result
/// to the host preview buffer.
void downscale_tile(
  cuda::stream_ref stream,
  tile_buffers& bufs,
  int slot,
  cuda::std::span<const pixel_t> dev_src,
  int row_offset,
  int tile_rows,
  cuda::std::span<pixel_t> host_preview);

#endif // DETAIL_H
