//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
    fptool_custom.cpp - Reduced floating-point formats emulated on native FP64
    ==========================================================================

    An fp_custom value is a double that rounds to a narrower format after every operation,
    so a computation can be run as though the hardware had that format. Both field sizes
    are chosen independently:

        fp64_custom<exponent_bits, mantissa_bits>

    Sizes equal to the native ones, 11 and 52, switch the emulation off and leave plain
    double arithmetic, so the named formats come out as:

        FP64   fp64_custom<>        FP16   fp64_custom<5, 10>
        FP32   fp64_custom<8, 23>   BF16   fp64_custom<8, 7>
        TF32   fp64_custom<8, 10>   PO2    fp64_custom<11, 0>

    Reducing the mantissa rounds the value; zero mantissa bits leave only the implicit
    leading 1, so every result snaps to a power of two. Reducing the exponent narrows the
    range instead, clamping what no longer fits to infinity or to zero.

    The sizes can come from the template arguments or from a variable, and this example
    runs one kernel per choice:

      compile-time    fp64_custom<8, 23> and friends. The format is part of the type, so
                      the emulation is specialized and the arithmetic for it is generated
                      inline, with the native format costing nothing at all.

      run-time        fp64_custom<fp_custom_dynamic_size, fp_custom_dynamic_size>, whose
                      sizes are read from a global that a setter changes between launches.
                      One binary then covers a sweep of formats, at the cost of the sizes
                      no longer being known to the optimizer.

    Both produce the same numbers for the same sizes, which the output shows by running
    the same expression through both paths.

    Demonstrated:

      - the named formats, and how far each one rounds the same sum
      - construction: the native format takes a double implicitly, a reduced one marks the
        value entering it
      - mixed arithmetic, a reduced value combined directly with a plain double
      - the reduced exponent range, where a value too large becomes infinity
      - sweeping the mantissa size at run time without recompiling, and reading back the
        size in effect
      - the same source running on host and device, which is also where the two ways of
        setting a run-time size differ: host code sets its own size directly, device code
        has it set from the host on a stream
*/
#include <cstdio>

// One header for the whole feature.
#include <cuda/fptool>

// The CCCL FP component lives in cuda::experimental (later cuda::), abbreviated here
// rather than pulled in with a using-directive. Everything this example names is specific
// to the component and so carries the prefix; note in particular that the size setters
// and getters take only an int and a stream, so argument-dependent lookup could not find
// them unqualified even if one wanted it to.
namespace cudax = cuda::experimental;

// The compile-time formats, in the order they are printed. The kernel evaluates the same
// expression in each, so the rows can be read against one another.
static constexpr int format_count = 6;

// The mantissa sizes the run-time kernel sweeps. 52 is the native size, which leaves the
// value alone; 0 keeps only the implicit leading 1. The sizes in between match the
// mantissas of the named formats above, so the two kernels can be compared row by row.
static constexpr int sweep_count                 = 5;
static constexpr int sweep_mantissa[sweep_count] = {52, 23, 10, 7, 0};

// One set of results, as doubles so the host can print them without knowing which format
// produced them. A reduced fp_custom is held in a double throughout, so this conversion
// is exact and is not what the rounding below comes from.
struct fp_custom_results
{
  double a, b; // the two inputs

  double formats[format_count]; // a + b, once per compile-time format
  double mixed; // a reduced value combined with a plain double
  double huge_native; // a large value, in the native format
  double huge_reduced; // the same value, in a format whose exponent cannot hold it

  double dynamic[sweep_count]; // a + b again, at each run-time mantissa size
};

// The compile-time formats.
__host__ __device__ void compile_time_formats(fp_custom_results* out)
{
  // The two inputs, held as plain doubles so that each format below is built from the same
  // pair of numbers written once.
  const double da = 1.123456789012345;
  const double db = 2.987654321098765;

  // Construction. The native format is a double in every respect and takes one implicitly.
  // A reduced format is narrower than the source, so the value is marked as entering it -
  // the braces are the only edit a body of double code needs.
  const cudax::fp64_custom<> a = da;
  const cudax::fp64_custom<> b = db;

  const cudax::fp64_custom<8, 23> a_fp32{da};
  const cudax::fp64_custom<8, 23> b_fp32{db};
  const cudax::fp64_custom<8, 10> a_tf32{da};
  const cudax::fp64_custom<8, 10> b_tf32{db};
  const cudax::fp64_custom<5, 10> a_fp16{da};
  const cudax::fp64_custom<5, 10> b_fp16{db};
  const cudax::fp64_custom<8, 7> a_bf16{da};
  const cudax::fp64_custom<8, 7> b_bf16{db};
  const cudax::fp64_custom<11, 0> a_po2{da};
  const cudax::fp64_custom<11, 0> b_po2{db};

  // The same addition in every format. Each result is rounded to the format it was
  // computed in, which is the whole of what the emulation does: the operands are reduced,
  // the native FP64 operation runs, and the result is reduced again.
  const cudax::fp64_custom<> sum_native    = a + b;
  const cudax::fp64_custom<8, 23> sum_fp32 = a_fp32 + b_fp32;
  const cudax::fp64_custom<8, 10> sum_tf32 = a_tf32 + b_tf32;
  const cudax::fp64_custom<5, 10> sum_fp16 = a_fp16 + b_fp16;
  const cudax::fp64_custom<8, 7> sum_bf16  = a_bf16 + b_bf16;
  const cudax::fp64_custom<11, 0> sum_po2  = a_po2 + b_po2;

  // Mixed arithmetic. A scalar operand needs no cast whatever the sizes are, and the
  // result stays in the reduced format rather than widening to double.
  const cudax::fp64_custom<8, 7> mixed = a_bf16 * 2.0;

  // The exponent decides the range rather than the precision. 1e300 is an ordinary double
  // and stays one in the native format, but no 5-bit exponent reaches it, so the value
  // becomes infinity. Note where that happens: the constructor stores the number
  // unreduced, and the first arithmetic operation is what clamps it.
  const cudax::fp64_custom<> huge_native = 1e300;
  const cudax::fp64_custom<5, 10> huge_reduced{1e300};
  const cudax::fp64_custom<5, 10> huge_used = huge_reduced * 1.0;

  // The single conversion point. Coming out is always exact: the value was held in a
  // double the whole time, already rounded to its format.
  out->a = static_cast<double>(a);
  out->b = static_cast<double>(b);

  out->formats[0] = static_cast<double>(sum_native);
  out->formats[1] = static_cast<double>(sum_fp32);
  out->formats[2] = static_cast<double>(sum_tf32);
  out->formats[3] = static_cast<double>(sum_fp16);
  out->formats[4] = static_cast<double>(sum_bf16);
  out->formats[5] = static_cast<double>(sum_po2);

  out->mixed = static_cast<double>(mixed);

  out->huge_native  = static_cast<double>(huge_native);
  out->huge_reduced = static_cast<double>(huge_used);
} // compile_time_formats

// The run-time counterpart. Both sizes come from globals instead of from the template
// arguments, so this one type covers every format and the kernel below is compiled once.
using fp_dynamic = cudax::fp64_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;

// Called once per size in the sweep, with the size already set by the caller. Nothing here
// mentions a format: the same instructions produce a different result each time.
__host__ __device__ void run_time_formats(fp_custom_results* out, int slot)
{
  // Explicit construction, as for any reduced format. With dynamic sizes the format a
  // value is entering is not even known at compile time, so it is always spelled out.
  const fp_dynamic a{1.123456789012345};
  const fp_dynamic b{2.987654321098765};

  const fp_dynamic sum = a + b;

  out->dynamic[slot] = static_cast<double>(sum);
} // run_time_formats

__global__ void compile_time_kernel(fp_custom_results* out)
{
  compile_time_formats(out);
}

__global__ void run_time_kernel(fp_custom_results* out, int slot)
{
  run_time_formats(out, slot);
}

// `where` names the side that produced the results, since a single run reports both.
static void print_compile_time(const char* where, const fp_custom_results& r)
{
  static const char* const names[format_count] = {
    "FP64  fp64_custom<>",
    "FP32  fp64_custom<8, 23>",
    "TF32  fp64_custom<8, 10>",
    "FP16  fp64_custom<5, 10>",
    "BF16  fp64_custom<8, 7>",
    "PO2   fp64_custom<11, 0>",
  };

  printf("\n");
  printf("================================================================================\n");
  printf("  compile-time formats on the %s - the sizes are template arguments\n", where);
  printf("================================================================================\n");

  printf("\ninputs\n");
  printf("  a = %.17g\n", r.a);
  printf("  b = %.17g\n", r.b);

  printf("\na + b, computed in each format\n");
  for (int i = 0; i < format_count; ++i)
  {
    printf("  %-28s %.17g\n", names[i], r.formats[i]);
  }

  printf("\nmixed arithmetic, a reduced value and a plain double\n");
  printf("  a * 2.0, in BF16             %.17g\n", r.mixed);

  printf("\nexponent range, on a value no reduced exponent can hold\n");
  printf("  1e300, in the native format  %.17g\n", r.huge_native);
  printf("  1e300, used in FP16          %.17g\n", r.huge_reduced);
}

static void print_run_time(const char* where, const fp_custom_results& r, const int* sizes_in_effect)
{
  printf("\n");
  printf("================================================================================\n");
  printf("  run-time formats on the %s - the sizes come from a variable\n", where);
  printf("================================================================================\n");

  printf("\na + b again, sweeping the mantissa size between launches\n");
  printf("  %-10s %-10s %s\n", "requested", "in effect", "a + b");
  for (int i = 0; i < sweep_count; ++i)
  {
    printf("  %-10d %-10d %.17g\n", sweep_mantissa[i], sizes_in_effect[i], r.dynamic[i]);
  }

  printf("\nthe same numbers the compile-time formats produced at those mantissa sizes,\n");
  printf("the exponent having been left native throughout\n");
}

int main()
{
  // Both entry points are __host__ __device__, so the build carries a host and a
  // device copy of each and one run can report both. The runtime sizes are the
  // one place where the two sides differ in more than execution space: host code
  // sets a size directly, device code sets a per-device one through a stream.
  fp_custom_results host_results{};
  int host_sizes[sweep_count] = {};

  compile_time_formats(&host_results);

  // Host code owns its own copy of the sizes and sets them directly, with no stream to
  // order the write against.
  for (int i = 0; i < sweep_count; ++i)
  {
    cudax::fp_custom_set_host_mantissa_size(sweep_mantissa[i]);

    run_time_formats(&host_results, i);

    host_sizes[i] = cudax::fp_custom_get_host_mantissa_size();
  }

  print_compile_time("host", host_results);
  print_run_time("host", host_results, host_sizes);

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
  {
    cudaGetLastError(); // discard the sticky error so it cannot mask a later one
    printf("\nno CUDA device available, so only the host results are shown\n\n");
    return 0;
  }

  // Name the GPU the device results came from. Its architecture is worth having next to
  // the numbers, since that is what decides which instructions the arithmetic is built on.
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, device_id);
  printf("\nthe device results below come from %s, sm_%d%d\n", props.name, props.major, props.minor);

  fp_custom_results* r;
  int sizes_in_effect[sweep_count] = {};

  cudaMallocManaged(&r, sizeof(fp_custom_results));

  // A device size is per-device state, so the setter takes the stream that says which
  // device to write it on and orders the write against the kernels that read it. No
  // synchronization of our own is needed between a set and the launch that follows it.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // The same two entry points, this time from kernels.
  compile_time_kernel<<<1, 1, 0, stream>>>(r);

  for (int i = 0; i < sweep_count; ++i)
  {
    cudax::fp_custom_set_device_mantissa_size(sweep_mantissa[i], stream);

    run_time_kernel<<<1, 1, 0, stream>>>(r, i);

    sizes_in_effect[i] = cudax::fp_custom_get_device_mantissa_size(stream);
  }

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaFree(r);
    return 1;
  }

  print_compile_time("device", *r);
  print_run_time("device", *r, sizes_in_effect);

  cudaFree(r);

  printf("\n");

  return 0;
} // main
