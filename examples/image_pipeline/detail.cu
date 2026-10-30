//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/// @file
/// Implementation of supporting details: synthetic image generation and
/// printing/output helpers.  See detail.h for the interface.

#include <cuda/algorithm>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/algorithm>
#include <cuda/std/span>
#include <cuda/stream>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <detail.h>

// ── Image generation ─────────────────────────────────────────────────

__device__ unsigned pixel_hash(int x, int y, unsigned seed)
{
  unsigned h = static_cast<unsigned>(x) * 1103515245u + static_cast<unsigned>(y) * 12345u + seed;
  h          = (h ^ (h >> 16)) * 0x45d9f3bu;
  h          = (h ^ (h >> 13)) * 0x85ebca6bu;
  return h ^ (h >> 16);
}

__device__ float hash01(int x, int y, unsigned seed)
{
  return static_cast<float>(pixel_hash(x, y, seed) & 0xFFFF) / 65535.0f;
}

__device__ float value_noise(float fx, float fy, float freq, unsigned seed)
{
  const float sx = fx * freq, sy = fy * freq;
  const int ix = static_cast<int>(floorf(sx)), iy = static_cast<int>(floorf(sy));
  float tx = sx - ix, ty = sy - iy;
  tx            = tx * tx * (3.0f - 2.0f * tx);
  ty            = ty * ty * (3.0f - 2.0f * ty);
  const float a = hash01(ix, iy, seed) + (hash01(ix + 1, iy, seed) - hash01(ix, iy, seed)) * tx;
  const float b = hash01(ix, iy + 1, seed) + (hash01(ix + 1, iy + 1, seed) - hash01(ix, iy + 1, seed)) * tx;
  return a + (b - a) * ty;
}

__device__ float fbm(float fx, float fy, int octaves, float freq, unsigned seed)
{
  float val = 0, amp = 1.0f, total = 0;
  for (int i = 0; i < octaves; ++i)
  {
    val += amp * value_noise(fx, fy, freq, seed + static_cast<unsigned>(i) * 7919u);
    total += amp;
    amp *= 0.5f;
    freq *= 2.0f;
  }
  return val / total;
}

struct generate_kernel
{
  template <typename Config>
  __device__ void operator()(Config config, cuda::std::span<pixel_t> out, int width, int row_offset)
  {
    const auto tid = cuda::gpu_thread.rank(cuda::grid, config);
    if (tid >= out.size())
    {
      return;
    }

    const int gx   = static_cast<int>(tid) % width;
    const int gy   = static_cast<int>(tid) / width + row_offset;
    const float fx = static_cast<float>(gx) / image_width;
    const float fy = static_cast<float>(gy) / image_height;

    float val         = 4.0f + 3.0f * fy;
    const float noise = (hash01(gx, gy, 0u) - 0.5f) * 6.0f;
    val += noise;

    const float neb1_d = ((fx - 0.6f) * (fx - 0.6f) + (fy - 0.35f) * (fy - 0.35f)) / 0.06f;
    float neb1         = 40.0f * expf(-neb1_d);
    const float neb2_d = ((fx - 0.35f) * (fx - 0.35f) + (fy - 0.6f) * (fy - 0.6f)) / 0.03f;
    float neb2         = 22.0f * expf(-neb2_d);

    const float tex = fbm(fx, fy, 5, 8.0f, 42u);
    neb1 *= (0.5f + tex);
    neb2 *= (0.3f + 0.7f * tex);

    const float dust      = fbm(fx + 0.1f, fy, 4, 6.0f, 137u);
    const float dust_mask = fmaxf(0.0f, 1.0f - 2.0f * fabsf(dust - 0.5f));
    neb1 *= (1.0f - 0.6f * dust_mask * expf(-neb1_d * 2.0f));
    val += neb1 + neb2;

    constexpr int star_grid = 64;
    const int cx            = (gx / star_grid) * star_grid + star_grid / 2;
    const int cy            = (gy / star_grid) * star_grid + star_grid / 2;
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        const int scx     = cx + dx * star_grid;
        const int scy     = cy + dy * star_grid;
        const unsigned sh = pixel_hash(scx, scy, 9999u);
        if (static_cast<float>(sh & 0xFFFF) / 65535.0f < 0.08f)
        {
          const float jx     = static_cast<float>((sh >> 4) & 0xFF) / 255.0f - 0.5f;
          const float jy     = static_cast<float>((sh >> 12) & 0xFF) / 255.0f - 0.5f;
          const float sx     = scx + jx * star_grid;
          const float sy     = scy + jy * star_grid;
          const float d2     = (gx - sx) * (gx - sx) + (gy - sy) * (gy - sy);
          const float radius = 2.0f + static_cast<float>((sh >> 20) & 0xF);
          const float bright = 80.0f + static_cast<float>((sh >> 24) & 0x7F);
          val += bright * expf(-d2 / (2.0f * radius * radius));
        }
      }
    }

    out[tid] = static_cast<pixel_t>(fminf(255.0f, fmaxf(0.0f, val)));
  }
};

// ── Image generation ─────────────────────────────────────────────────

void generate_image(cuda::stream_ref stream, tile_buffers& bufs, int num_tiles, cuda::std::span<pixel_t> host_preview)
{
  std::cout << "=== Image generation (GPU) ===\n";
  cuda::timed_event gen_start{stream};

  for (int t = 0; t < num_tiles; ++t)
  {
    const size_t offset  = static_cast<size_t>(t) * bufs.tile_pixels;
    const size_t count   = cuda::std::min(bufs.tile_pixels, image_pixels - offset);
    const int tile_rows  = static_cast<int>(count / image_width);
    const int row_offset = t * static_cast<int>(bufs.tile_pixels / image_width);

    constexpr int block_size = 256;
    const auto config        = cuda::distribute<block_size>(static_cast<int>(count));
    cuda::launch(stream, config, generate_kernel{}, bufs.dev_tile[0].first(count), image_width, row_offset);

    // Downscale the generated tile for the input preview while it's still on device.
    downscale_tile(stream, bufs, 0, bufs.dev_tile[0].first(count), row_offset, tile_rows, host_preview);

    // Copy generated tile to host for the histogram pass.
    cuda::copy_bytes(stream, bufs.dev_tile[0].first(count), bufs.host_image.subspan(offset, count));
  }

  cuda::timed_event gen_end{stream};
  stream.sync();
  const double gen_ms = (gen_end - gen_start).count() / 1e6;
  std::cout
    << "  Generated " << image_width << 'x' << image_height << " space observation (" << std::fixed
    << std::setprecision(0) << image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0) << " MB) in ~" << std::setprecision(1)
    << gen_ms << " ms\n\n"
    << std::defaultfloat << std::setprecision(6);
}

// ── Printing / output helpers ────────────────────────────────────────

void print_device_info(cuda::device_ref dev, cuda::arch_traits_t traits, size_t total_mem)
{
  const auto name = dev.name();
  const auto cc   = dev.attribute(cuda::device_attributes::compute_capability);
  std::cout << "\nSelected device " << dev.get() << ": ";
  std::cout.write(name.data(), static_cast<std::streamsize>(name.size()));
  std::cout
    << "\n  Compute capability: " << cc.major_cap() << '.' << cc.minor_cap() << "\n  Total memory      : " << std::fixed
    << std::setprecision(0) << total_mem / (1024.0 * 1024.0) << " MB"
    << "\n  Max threads/block : " << traits.max_threads_per_block
    << "\n  Max shared memory : " << traits.max_shared_memory_per_block << " bytes\n"
    << std::defaultfloat << std::setprecision(6);
}

void print_tile_plan(int tile_rows, int tile_alignment, int num_tiles, size_t budget, size_t total_mem)
{
  std::cout
    << "\nTile plan:\n"
    << "  Image          : " << image_width << " x " << image_height << " (" << std::fixed << std::setprecision(0)
    << image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0) << " MB)\n"
    << "  GPU budget     : " << budget / (1024.0 * 1024.0) << " MB (60% of " << total_mem / (1024.0 * 1024.0)
    << " MB)\n"
    << "  Tile rows      : " << tile_rows << " (aligned to " << tile_alignment << ")\n"
    << "  Number of tiles: " << num_tiles << "\n\n"
    << std::defaultfloat << std::setprecision(6);
}

void print_allocation_info(size_t device_total, size_t gpu_budget, size_t tile_pixels, int tile_rows)
{
  std::cout
    << std::fixed << std::setprecision(1) << "  Device pool: " << device_total / (1024.0 * 1024.0) << " MB (initial), "
    << gpu_budget / (1024.0 * 1024.0) << " MB (max)\n"
    << "  Tile size: " << tile_pixels << " pixels (" << tile_rows << " rows x " << image_width << " cols)\n"
    << "  Pinned host: image=" << image_pixels * sizeof(pixel_t) / (1024.0 * 1024.0)
    << " MB, histogram=" << num_bins * sizeof(int) << " bytes\n\n"
    << std::defaultfloat << std::setprecision(6);
}

void print_pool_stats(tile_buffers& bufs)
{
  const auto reserved = bufs.device_pool.get().attribute(cuda::memory_pool_attributes::reserved_mem_current);
  const auto used     = bufs.device_pool.get().attribute(cuda::memory_pool_attributes::used_mem_current);
  std::cout
    << std::fixed << std::setprecision(1) << "  Device pool: reserved=" << reserved / (1024.0 * 1024.0)
    << " MB, used=" << used / (1024.0 * 1024.0) << " MB\n"
    << std::defaultfloat << std::setprecision(6);
}

iqr_result compute_iqr(cuda::std::span<const histogram_count_t> hist, size_t total)
{
  auto find_percentile = [&](float pct) {
    const auto target = static_cast<histogram_count_t>(static_cast<double>(total) * pct);
    histogram_count_t cumulative{};
    for (size_t i = 0; i < hist.size(); ++i)
    {
      cumulative += hist[i];
      if (cumulative >= target)
      {
        return static_cast<int>(i);
      }
    }
    return static_cast<int>(hist.size()) - 1;
  };
  return {find_percentile(0.25f), find_percentile(0.75f)};
}

void print_pass_stats(double ms, long long total_selected, double mean_selected, float global_min, float global_max)
{
  // Note: times measured via cuda::timed_event are approximate GPU-side measurements.
  std::cout
    << std::fixed << std::setprecision(1) << "  Pass time: ~" << ms << " ms\n"
    << "  Pixels above threshold: " << total_selected << " / " << image_pixels << " ("
    << 100.0 * total_selected / image_pixels << "%)\n"
    << std::setprecision(4) << "  Selected range: [" << global_min << ", " << global_max << "], mean=" << mean_selected
    << '\n'
    << std::defaultfloat << std::setprecision(6);
}

void print_sanity_check(iqr_result orig, iqr_result eq)
{
  const bool ok = eq.width() > orig.width();
  std::cout
    << "=== Sanity check ===\n"
    << "  Original IQR (25th-75th):   [" << orig.p25 << ", " << orig.p75 << "] (span " << orig.width() << ")\n"
    << "  Equalized IQR (25th-75th):  [" << eq.p25 << ", " << eq.p75 << "] (span " << eq.width() << ")\n"
    << "  Equalization spread distribution: " << (ok ? "YES" : "NO") << "\n\n";
}

void print_summary(int num_tiles, int tile_rows, double pass1_ms, double pass2_ms, bool ok)
{
  std::cout
    << "=== Summary ===\n"
    << "  Image:          " << image_width << " x " << image_height << '\n'
    << "  Tiles:          " << num_tiles << " (" << tile_rows << " rows each)\n"
    << std::fixed << std::setprecision(1) << "  Pass 1 (hist):  ~" << pass1_ms << " ms\n"
    << "  Pass 2 (stats): ~" << pass2_ms << " ms\n"
    << "  Total pipeline: ~" << pass1_ms + pass2_ms << " ms\n"
    << "  Result:         " << (ok ? "PASSED" : "FAILED") << '\n'
    << std::defaultfloat << std::setprecision(6);
}

bool write_bmp(const char* filename, cuda::std::span<const pixel_t> data, int width, int height)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file)
  {
    std::cerr << "Failed to open " << filename << " for writing\n";
    return false;
  }

  // BMP rows must be padded to a 4-byte boundary.
  const int row_stride = (width + 3) & ~3;
  const int pixel_size = row_stride * height;
  const int file_size  = 54 + 256 * 4 + pixel_size; // header + palette + pixels

  auto put2 = [&](int v) {
    const char b[2] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF)};
    file.write(b, 2);
  };
  auto put4 = [&](int v) {
    const char b[4] = {static_cast<char>(v & 0xFF),
                       static_cast<char>((v >> 8) & 0xFF),
                       static_cast<char>((v >> 16) & 0xFF),
                       static_cast<char>((v >> 24) & 0xFF)};
    file.write(b, 4);
  };

  // File header (14 bytes).
  file.put('B');
  file.put('M');
  put4(file_size);
  put4(0); // reserved
  put4(54 + 256 * 4); // pixel data offset (after header + palette)

  // DIB header (BITMAPINFOHEADER, 40 bytes).
  put4(40); // header size
  put4(width);
  put4(height);
  put2(1); // color planes
  put2(8); // bits per pixel (8-bit indexed)
  put4(0); // no compression
  put4(pixel_size);
  put4(2835); // horizontal resolution (72 DPI)
  put4(2835); // vertical resolution
  put4(256); // palette entries
  put4(0); // all colors important

  // Grayscale palette: 256 entries of (B, G, R, 0).
  for (int i = 0; i < 256; ++i)
  {
    const char c = static_cast<char>(i);
    file.put(c);
    file.put(c);
    file.put(c);
    file.put(0);
  }

  // Pixel data — BMP stores rows bottom-to-top.
  const char pad[3]   = {0, 0, 0};
  const int pad_bytes = row_stride - width;
  for (int y = height - 1; y >= 0; --y)
  {
    file.write(reinterpret_cast<const char*>(&data[static_cast<size_t>(y) * width]), width);
    if (pad_bytes > 0)
    {
      file.write(pad, pad_bytes);
    }
  }

  if (!file)
  {
    std::cerr << "Failed to write " << filename << '\n';
    return false;
  }

  std::cout << "  Wrote " << filename << " (" << width << " x " << height << ")\n";
  return true;
}
