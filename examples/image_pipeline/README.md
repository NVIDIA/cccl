
# Image Processing Pipeline Example

A multi-file example showcasing the CCCL Runtime and CUB APIs working together
in a semi-realistic tiled image processing pipeline.

## What it does

The example generates a synthetic 65K x 65K (~4 GB) grayscale space observation
on the GPU, then processes it in tiles that fit in GPU memory:

1. **Pass 1 - Histogram**: Upload each tile, compute per-tile histograms with
   `cub::DeviceHistogram`, download and accumulate into a global histogram.

2. **Host interlude**: Compute Otsu's threshold (optimal foreground/background
   split) and build a histogram equalization lookup table from the CDF.

3. **Pass 2 - Equalize + Analyze**: For each tile, apply the equalization LUT
   (`cub::DeviceTransform`), normalize to float (`cub::DeviceTransform`),
   compute thresholded count/min/max/sum (`cub::DeviceReduce::TransformReduce`),
   and GPU-downscale a preview (`cub::BlockReduce`).

4. **Output**: Write `input_preview.bmp` and `equalized_preview.bmp` (1024 x
   1024 previews). The equalized image reveals nebula structure and stars that
   are barely visible in the dark original.

## CCCL APIs demonstrated

| File | APIs |
|------|------|
| `image_pipeline.h` | Shared constants and buffer/plan structs using `cuda::device_buffer`, `cuda::host_buffer`, `cuda::mr::shared_resource`, and spans |
| `detail.h` | Example-local declarations for image generation, preview output, reporting, and downscaling helpers |
| `detail.cu` | Synthetic image generation with `cuda::launch`/`cuda::distribute`, tile preview downscaling, `cuda::copy_bytes`, memory-pool statistics, and BMP output |
| `main.cu` | `cuda::devices`, `cuda::device_ref`, `cuda::device_attributes`, `cuda::arch_traits_for`, `cuda::device_memory_pool`, `cuda::memory_pool_properties`, `cuda::mr::shared_resource`, `cuda::make_buffer`, `cuda::make_pinned_buffer`, `cuda::copy_bytes`, `cuda::copy_configuration`, `cuda::fill_bytes`, `cuda::stream`, `cuda::timed_event`, `stream.wait(...)`, `buffer.first()`, `buffer.subspan()`, `buffer.get_unsynchronized()`, CUB `DeviceHistogram`, `DeviceTransform`, `DeviceReduce::TransformReduce`, and `BlockReduce` |
| `CMakeLists.txt` | Standalone CMake/CPM setup for consuming CCCL from a chosen repository and tag |

## Building and running

```bash
cd examples/image_pipeline
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
./build/image_pipeline
```

The example requires CUDA Toolkit 13.1 or newer for the CCCL Runtime APIs used
by the sample. The standalone CMake project uses the vendored `cmake/CPM.cmake`
helper to fetch CCCL; override `CCCL_REPOSITORY` and `CCCL_TAG` to build against
a local checkout or a specific branch.

The example requires ~4 GB of pinned host memory for the full image and uses
60% of GPU memory for the per-tile working set. It should run on any GPU with
at least 4 GB of memory.
