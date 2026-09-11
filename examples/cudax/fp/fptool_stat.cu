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
    fptool_stat.cu - Arithmetic statistics collected from a multi-precision computation
    ===================================================================================

    An fpmp2_stat type is a drop-in replacement for the fpmp2 type it instruments: it
    computes bit-identical results, and additionally records on the device what the
    arithmetic did. Swapping the type is the only change a computation needs.

        fp32mp2_stat   instruments fp32mp2, float-float
        fp64mp2_stat   instruments fp64mp2, double-double

    with _low, _mid and _high variants of each, matching the fpmp2 accuracy levels. This
    example runs the same computation once per width, so the two kernels below differ only
    in the type they are written in.

    Instrumenting is selective: an fpmp2_stat type can be used for the variables under
    suspicion while the rest of the computation stays plain fpmp2, and only the operations
    that touch an instrumented value are recorded. Two rules govern what that ends up
    covering:

      - an operation is counted when either of its operands is instrumented
      - its result is instrumented too, so the property propagates along an expression
        until a plain variable absorbs it

    So marking one variable records everything downstream of it, not only the operations
    it appears in directly - useful to know when reading a record back.

    The measured computation is a Kahan-compensated sum of a partial geometric series,
    with the compensation term as the single instrumented variable, because whether the
    compensation is doing anything is the question worth asking about such a loop. The
    accumulation is drawn in through the compensation and is counted; the term progression
    never touches it and stays out, which is why no multiplications appear in the record
    even though the loop performs one per term.

    Compensating also makes the cancellation counters non-zero on a computation where
    nothing is wrong, so reading them means separating a deliberate cancellation from a
    defect. And comparing the two widths shows which numbers depend on the format: the
    operation counts do not, the cancellation and limb-gap numbers do.

    Counting covers the arithmetic operators, including their compound and atomic forms.
    The composite builtins and the math functions from <cuda/fptool> - sqrt, rsqrt, fma,
    renormalize, exp and the rest - are deliberately left out, since their internals would
    swamp the record. They work normally, and this example calls exp() to show that the
    totals stay attributable to the operators actually written in the source.

    The record holds three kinds of information:

      - how many operations ran, by kind: add, sub, mul, div
      - numerical events, meaning operations that were degenerate rather than merely
        imprecise: full and partial cancellation, underflow to zero, overflow
      - a summary of the values that passed through each operand slot and through the
        result: exponent range, zero/inf/NaN/denormal counts, and the gap between the hi
        and lo limbs, which says how much of the double-word range the pair was using

    The counters are deliberately shared by every instantiation, so a measurement has to
    be bracketed: clear the record, run the region of interest, read it back. That is why
    the two widths below are measured one after the other rather than together. Both calls
    take a cuda::stream_ref, which says both which device holds the record and where the
    copies belong in the program's ordering; the read waits on that stream, so no
    synchronization of our own is needed.

    Demonstrated:

      - instrumenting one variable of interest and leaving the rest as plain fpmp2
      - bracketing a measurement with fpmp2_stat_reset_device_data() and
        fpmp2_stat_read_device_data()
      - reading the operation counters, the event counters and the per-slot value summary
      - the same source running on host and device
*/
#include <cstdio>
#include <exception>

// One header for the whole feature: the analysis types and the math functions
// that go with them.
#include <cuda/fptool>

// The CCCL runtime, for the device half of the example: finding a device, a
// stream to submit on, buffers for the results, the kernel launches, and the
// copy that brings the results back.
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include <cuda/stream>

// The CCCL FP component lives in cuda::experimental (later cuda::),
// abbreviated here rather than pulled in with a using-directive. Type names
// and the component's own entry points - here the reset and read functions,
// which take only a stream and so cannot be found by argument-dependent
// lookup at all - carry the cudax:: prefix; the standard-named math functions
// are left unqualified.
namespace cudax = cuda::experimental;

// Terms of the geometric series the measured computation sums.
static constexpr int fptool_stat_series_terms = 64;

// One kernel result. The value is reported as both limbs, since the limb gap is
// one of the things the record describes, and as doubles so the host can print
// it without knowing which width produced it.
struct fptool_stat_results
{
  double hi, lo;
};

// The float-float series.
__host__ __device__ void float_float_series(fptool_stat_results* out)
{
  // The working type is the only thing that separates this from an
  // uninstrumented computation: fp32mp2_stat in place of fp32mp2. Everything
  // below is ordinary fpmp2 arithmetic and produces the same results it would
  // without the instrumentation. Only the variable under suspicion is
  // instrumented. The compensation term is the one worth watching: it exists to
  // carry rounding error, so whether it is doing anything is exactly the
  // question the record can answer. Everything else stays plain fp32mp2 and so
  // stays out of the record - the term progression below is a sequence of exact
  // powers of two with nothing to learn from it.
  cudax::fp32mp2 sum{0.0f};
  cudax::fp32mp2_stat compensation{0.0f};
  cudax::fp32mp2 term{1.0f};
  const cudax::fp32mp2 ratio{0.5f};

  // Kahan summation of a partial geometric series. An operation is counted when
  // either of its operands is instrumented, and its result carries the
  // instrumentation onward, so touching compensation is enough to pull the
  // whole compensation chain into the record: y is instrumented because
  // compensation is, and t in turn because y is. The write-back sum = t is what
  // stops the spread, converting back to plain.
  //
  // What the counters then say about this data is the interesting part. Full
  // cancellation fires once per term, because compensation comes out exactly
  // zero every time: a double-word type sums this series exactly, so there was
  // no rounding error to carry, and the record is what tells us the
  // compensation is dead weight here. Partial cancellation counts the t - sum
  // step instead, which loses more than half the significand once the term
  // drops below one limb's worth of the running sum - after 24 terms at this
  // width, so 64 - 24 = 40 of them.
  for (int i = 0; i < fptool_stat_series_terms; ++i)
  {
    const cudax::fp32mp2_stat y = term - compensation;
    const cudax::fp32mp2_stat t = sum + y;
    compensation                = (t - sum) - y;
    sum                         = t;

    // Plain on both sides, so this multiplication never reaches the record: mul
    // stays at 0 however many terms the loop runs. That is the whole point of
    // instrumenting selectively.
    term = term * ratio;
  }

  // Applying the correction the loop left pending, which is what finalizes a
  // Kahan sum. It touches compensation, so it is counted.
  const cudax::fp32mp2_stat total = sum - compensation;

  // A division, and a transcendental to show what is not counted: exp()
  // contributes nothing to the record, while the division and the addition
  // around it each count once, both having total as an operand. The totals
  // therefore account for exactly the instrumented operators written here: 64 +
  // 1 additions, 3 x 64 + 1 subtractions, 1 division, and no multiplications at
  // all.
  const cudax::fp32mp2_stat scaled = total / cudax::fp32mp2{3.0f};
  const cudax::fp32mp2_stat value  = scaled + exp(ratio);

  // The single conversion point, as in the other fp examples.
  out->hi = static_cast<double>(value.hi());
  out->lo = static_cast<double>(value.lo());
} // float_float_series

// The double-double series.
__host__ __device__ void double_double_series(fptool_stat_results* out)
{
  // The same sequence on the wider type; only the type name and the literal
  // suffixes differ. The operation counts come out identical, since they depend
  // on the operators written rather than on the width. The width-dependent
  // numbers are the ones to compare: partial cancellation drops to 64 - 53 =
  // 11, one limb here being 53 bits rather than 24, and the limb gaps widen to
  // match.
  cudax::fp64mp2 sum{0.0};
  cudax::fp64mp2_stat compensation{0.0};
  cudax::fp64mp2 term{1.0};
  const cudax::fp64mp2 ratio{0.5};

  for (int i = 0; i < fptool_stat_series_terms; ++i)
  {
    const cudax::fp64mp2_stat y = term - compensation;
    const cudax::fp64mp2_stat t = sum + y;
    compensation                = (t - sum) - y;
    sum                         = t;
    term                        = term * ratio;
  }

  const cudax::fp64mp2_stat total  = sum - compensation;
  const cudax::fp64mp2_stat scaled = total / cudax::fp64mp2{3.0};
  const cudax::fp64mp2_stat value  = scaled + exp(ratio);

  out->hi = static_cast<double>(value.hi());
  out->lo = static_cast<double>(value.lo());
} // double_double_series

__global__ void float_float_kernel(fptool_stat_results* out)
{
  float_float_series(out);
}

__global__ void double_double_kernel(fptool_stat_results* out)
{
  double_double_series(out);
}

// Every counted operation feeds one value into each slot, so the operation
// count is the total these shares are shares of.
static void print_value_slot(const char* name, const cudax::fpmp2_stat_value& slot, unsigned long long total)
{
  if (total == 0)
  {
    printf("  %-8s no values sampled\n", name);
    return;
  }

  // The limb gap is normalized: 0 or 1 means a tightly normalized pair,
  // negative means the limbs overlap, larger means accuracy held in reserve. A
  // range still holding the sentinels reset() armed it with was never sampled,
  // which happens to the gap when every value in the slot had a zero lo limb
  // and so no gap to measure.
  if (slot.min_hi_lo_gap <= slot.max_hi_lo_gap)
  {
    printf("  %-8s exponent range [%d .. %d], hi/lo limb gap [%d .. %d]\n",
           name,
           slot.min_exp,
           slot.max_exp,
           slot.min_hi_lo_gap,
           slot.max_hi_lo_gap);
  }
  else
  {
    printf("  %-8s exponent range [%d .. %d], hi/lo limb gap never sampled\n", name, slot.min_exp, slot.max_exp);
  }

  // All three shares are of the same total, so they can be read against each
  // other. An inverted pair, lo outweighing hi, is the subset of the overlaps
  // that makes a value read wrong rather than merely imprecise.
  const double overlap_pct = (total > 0) ? 100.0 * (double) slot.overlap_count / (double) total : 0.0;
  const double invert_pct  = (total > 0) ? 100.0 * (double) slot.invert_count / (double) total : 0.0;
  const double zero_lo_pct = (total > 0) ? 100.0 * (double) slot.zero_lo_count / (double) total : 0.0;
  printf(
    "           zero=%llu  inf=%llu  nan=%llu  infnan=%llu  denorm=%llu"
    "  overlap=%llu (%.1f%%)  invert=%llu (%.1f%%)  zero_lo=%llu (%.1f%%)\n",
    slot.zero_count,
    slot.inf_count,
    slot.nan_count,
    slot.infnan_count,
    slot.denorm_count,
    slot.overlap_count,
    overlap_pct,
    slot.invert_count,
    invert_pct,
    slot.zero_lo_count,
    zero_lo_pct);

  // The tightest pair seen, i.e. the one that used the least of the double-word
  // range. The sentinels are still in place if nothing was sampled.
  if (slot.min_hi_lo_gap <= slot.max_hi_lo_gap)
  {
    printf("           tightest pair: hi=%.17g  lo=%.17g\n", slot.min_hi_lo_gap_sample_hi, slot.min_hi_lo_gap_sample_lo);
  }
}

// `where` names the side that produced the results, since a single run reports
// both. stats is null for the host side, where the computation runs but nothing
// is collected: the record lives in device memory and is reached through the
// stream-ordered API, so host-side arithmetic has nowhere to count.
static void print_results(
  const char* where,
  const char* type_name,
  const char* description,
  const fptool_stat_results& r,
  const cudax::fpmp2_stat_data* stats)
{
  printf("\n");
  printf("====================================================================="
         "===========\n");
  printf("  %s on the %s - %s\n", type_name, where, description);
  printf("====================================================================="
         "===========\n");

  printf("\nresult of the measured computation\n");
  printf("  hi               = %.17g\n", r.hi);
  printf("  lo               = %.17g\n", r.lo);

  if (stats == nullptr)
  {
    printf("\nno counters: the record is device memory, so nothing is collected here\n");
    return;
  }

  printf("\noperation counters\n");
  printf("  total            = %llu\n", stats->ops_count);
  printf("  add              = %llu\n", stats->add_count);
  printf("  sub              = %llu\n", stats->sub_count);
  printf("  mul              = %llu\n", stats->mul_count);
  printf("  div              = %llu\n", stats->div_count);

  printf("\nnumerical events\n");
  printf("  cancellation, full    = %llu\n", stats->full_cancel_count);
  printf("  cancellation, partial = %llu\n", stats->partial_cancel_count);
  printf("  underflow to zero     = %llu\n", stats->underflow_count);
  printf("  overflow              = %llu\n", stats->overflow_count);

  printf("\nvalue statistics, over the %llu counted operations\n", stats->ops_count);
  print_value_slot("arg[0]", stats->arg[0], stats->ops_count);
  print_value_slot("arg[1]", stats->arg[1], stats->ops_count);
  print_value_slot("result", stats->result, stats->ops_count);
} // print_results

// Whether this machine can run the device half of the example. Neither a missing
// GPU nor a driver that cannot initialize is a failure here: the host results
// stand on their own, so both fall back to reporting only those. Enumerating the
// devices is what first touches the driver, so it is also where an installation
// that cannot be used surfaces.
static bool device_available()
try
{
  return cuda::devices.size() != 0;
}
catch (const cuda::cuda_error&)
{
  return false;
}

// Where a kernel leaves its results, and how they get back to the host. CUDA 12.9
// brought the pinned memory pool, and on those toolkits the kernel writes memory the
// host can read directly, so draining the stream is all that stands between the
// launch and the numbers. Before that version there is no such pool, so the buffer
// lives on the device and cuda::copy_bytes carries it back. The choice is plumbing
// only: the arithmetic being demonstrated, and everything printed, is the same.
#if CUDART_VERSION >= 12090

template <class Results>
[[nodiscard]] static auto make_results_buffer(cuda::stream_ref stream, cuda::device_ref)
{
  return cuda::make_pinned_buffer<Results>(stream, 1, cuda::no_init);
}

template <class Results, class Buffer>
static void read_results(cuda::stream_ref stream, const Buffer& buffer, Results& results)
{
  stream.sync();
  results = buffer[0];
}

#else // ^^^ CUDA 12.9 and up ^^^ / vvv below CUDA 12.9 vvv

template <class Results>
[[nodiscard]] static auto make_results_buffer(cuda::stream_ref stream, cuda::device_ref device)
{
  return cuda::make_device_buffer<Results>(stream, device, 1, cuda::no_init);
}

template <class Results, class Buffer>
static void read_results(cuda::stream_ref stream, const Buffer& buffer, Results& results)
{
  cuda::copy_bytes(stream, buffer, cuda::std::span<Results>{&results, 1});
  stream.sync();
}

#endif // below CUDA 12.9

int main()
try
{
  // The series are __host__ __device__, so the build carries a host and a device
  // copy of each and one run can report both. Only the device side carries
  // counters, so the host block shows the same arithmetic with the statistics
  // section empty.
  fptool_stat_results host_float_float{};
  fptool_stat_results host_double_double{};

  float_float_series(&host_float_float);
  double_double_series(&host_double_double);

  print_results("host", "fp32mp2_stat", "instruments fp32mp2, float-float", host_float_float, nullptr);
  print_results("host", "fp64mp2_stat", "instruments fp64mp2, double-double", host_double_double, nullptr);

  if (!device_available())
  {
    printf("\nno CUDA device available, so only the host results are shown\n\n");
    return 0;
  }

  // Name the GPU the device results came from. Its architecture is worth having next to
  // the numbers, since that is what decides which instructions the arithmetic is built on.
  const cuda::device_ref device = cuda::devices[0];
  const auto device_name        = device.name();
  const auto cc                 = device.attribute(cuda::device_attributes::compute_capability);
  printf("\nthe device results below come from %.*s, sm_%d%d\n",
         static_cast<int>(device_name.size()),
         device_name.data(),
         cc.major_cap(),
         cc.minor_cap());

  // The clear and the read place their copies through a stream, so the whole
  // measurement runs on one. The counter API takes it directly.
  cuda::stream stream{device};

  // A one-value buffer per width for the kernels to write, and the host values each
  // is read back into once its kernel has run.
  auto float_float   = make_results_buffer<fptool_stat_results>(stream, device);
  auto double_double = make_results_buffer<fptool_stat_results>(stream, device);

  fptool_stat_results device_float_float{};
  fptool_stat_results device_double_double{};

  // One thread does all of it, since the point is the arithmetic rather than
  // the parallelism.
  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>());

  // One bracketed measurement per width. The second bracket starts from a
  // cleared record, which is what makes its counts attributable to fp64mp2_stat
  // alone; without the clear the two runs would accumulate together, since the
  // counters are shared.
  cudax::fpmp2_stat_reset_device_data(stream);

  // Run the float-float kernel on the device.
  cuda::launch(stream, config, float_float_kernel, float_float.data());
  read_results(stream, float_float, device_float_float);

  const cudax::fpmp2_stat_data float_float_record = cudax::fpmp2_stat_read_device_data(stream);

  cudax::fpmp2_stat_reset_device_data(stream);

  // Run the double-double kernel on the device.
  cuda::launch(stream, config, double_double_kernel, double_double.data());
  read_results(stream, double_double, device_double_double);

  const cudax::fpmp2_stat_data double_double_record = cudax::fpmp2_stat_read_device_data(stream);

  print_results("device", "fp32mp2_stat", "instruments fp32mp2, float-float", device_float_float, &float_float_record);
  print_results(
    "device", "fp64mp2_stat", "instruments fp64mp2, double-double", device_double_double, &double_double_record);

  printf("\n");

  return 0;
} // main
catch (const cuda::cuda_error& e)
{
  printf("CUDA error: %s\n", e.what());
  return 1;
}
catch (const std::exception& e)
{
  printf("error: %s\n", e.what());
  return 1;
}
