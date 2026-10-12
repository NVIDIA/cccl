.. _libcudacxx-extended-api-fp-fpmp-example:

fpmp: Examples
==============

Two programs, with different purposes. The first is the example that ships with CCCL, which shows
what each operation does and is the place to start. The second is the benchmark behind the figures
on the :ref:`fpmp page <libcudacxx-extended-api-fp-fpmp>`, which shows what the arithmetic costs
and is the place to go once the question is whether a type is worth using.

The example that ships with CCCL
--------------------------------

`examples/cudax/fp/fpmp.cu <https://github.com/NVIDIA/cccl/blob/main/examples/cudax/fp/fpmp.cu>`__
is a complete program covering the type's surface: construction and the conversion rules,
arithmetic and comparisons, the accuracy levels side by side with ``hi`` and ``lo`` printed
separately, ``renormalize``, the accuracy-explicit ``add<>``, and the math functions. Every
operation is performed twice, once on the host and once in a kernel, and both results are printed,
so the claim that one source serves both can be checked rather than taken on faith.

It builds with the other CCCL examples, and the
`sibling programs <https://github.com/NVIDIA/cccl/tree/main/examples/cudax/fp>`__ do the same for
``fpemu`` and the ``fptool`` types. Those examples are also where the CCCL runtime API usage worth
copying lives — device enumeration that degrades gracefully when there is no GPU, buffers that
release themselves, and a launch configuration built with ``cuda::make_config``.

Measuring a type against ``double``
-----------------------------------

The program below computes

.. math::

    \pi = 4 \int_0^1 \frac{dx}{1 + x^2}

by the midpoint rule, once in native ``double`` and once in a type of your choosing, and reports
for each the value, how many decimal digits of it are correct, and how long the kernel took. It is
deliberately a poor numerical method and a good benchmark: five arithmetic operations per term with
no memory traffic worth speaking of, so what it prices is arithmetic pipelines rather than
bandwidth, and both runs execute identical code.

Three details in it are worth lifting out, because they are what separates a measurement from a
number:

- **The first launch of a kernel is where its module gets loaded**, and that happens inside the
  launch, so timing it prices the load rather than the arithmetic. Left uncorrected it is worth
  tens of milliseconds — enough to make whichever type runs first look far slower than it is. Hence
  the short untimed run before the timed one.
- **Every step stays in the type being measured, the final reduction included.** A sequential sum
  of 131072 partials in the type under test contributes rounding error of its own, so a reduction
  done in something wider would report the loop alone rather than what a program written in that
  type actually costs. It is also the part of the run that separates the three ``fp32mp2`` accuracy
  levels, since a long chain of additions is far more sensitive to the trailing limb than the five
  operations of a single term are.
- **The reference is a double-double**, since a ``double`` reference cannot rank types more
  accurate than a ``double``, which ``fp64mp2`` is.

.. code-block:: cuda

    #include <cuda/algorithm>
    #include <cuda/buffer>
    #include <cuda/devices>
    #include <cuda/fpmp> // the reference is fp64mp2, whatever type is measured
    #include <cuda/launch>
    #include <cuda/std/span>
    #include <cuda/stream>

    #include <cmath>
    #include <cstdio>
    #include <exception>
    #include <vector>

    // The one thing to change. The rest of the program is generic, and native double is
    // always the baseline it is compared against.
    //
    //   <cuda/fpmp>    cudax::fp32mp2         the default float pair, 46 significand bits
    //                  cudax::fp32mp2_low     same width, cheapest arithmetic
    //                  cudax::fp32mp2_high    same width, most accurate arithmetic
    //                  cudax::fp64mp2         double pair, 104 significand bits
    //   <cuda/fpemu>   cudax::fp64emu         double emulated on integer and FP32 units
    //                  cudax::fp64emu_unpacked   the same, kept unpacked through the chain
    //   <cuda/fptool>  cudax::fp64_custom<8, 10>   a narrower format, here BF16's widths
    //
    // float and double work here too, as yardsticks rather than as subjects: they show what
    // the hardware does without the component.
    //
    // The type and the header that declares it travel together, so both come from the build
    // line, which is how one type is swapped for another without editing the file:
    // -DPI_FP_T=cudax::fp64emu_unpacked -DPI_FP_HEADER='<cuda/fpemu>'
    #ifndef PI_FP_T
    #  define PI_FP_T cudax::fp32mp2
    #endif

    #ifndef PI_FP_HEADER
    #  define PI_FP_HEADER <cuda/fpmp>
    #endif

    #include PI_FP_HEADER

    namespace cudax = cuda::experimental;

    // The label printed for the type is the type's own spelling, so there is nothing to keep
    // in step with it. Variadic because a type like fp64_custom<8, 10> reaches the inner
    // macro as two arguments, and # on __VA_ARGS__ puts the comma back.
    #define PI_STRINGIZE_(...) #__VA_ARGS__
    #define PI_STRINGIZE(...)  PI_STRINGIZE_(__VA_ARGS__)

    using fp_t                    = PI_FP_T;
    static constexpr auto kFpName = PI_STRINGIZE(PI_FP_T);

    // The term count is a power of two so that the step 1/n is exact and every grid point is
    // representable in a type with enough significand for it; -DPI_LOG2_TERMS=nn varies it,
    // which is how one checks whether a type has stopped resolving the grid.
    #ifndef PI_LOG2_TERMS
    #  define PI_LOG2_TERMS 26
    #endif

    static constexpr int kBlocks       = 512;
    static constexpr int kThreads      = 256;
    static constexpr long long kTerms  = 1LL << PI_LOG2_TERMS;
    static constexpr int kThreadsTotal = kBlocks * kThreads;

    // pi as a double-double: hi is double(pi), lo is the remainder.
    static constexpr double kPiHi = 3.14159265358979311600e0;
    static constexpr double kPiLo = 1.22464679914735317722e-16;

    template <class T>
    __global__ void pi_kernel(long long n, T* partials)
    {
      const int tid      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int nthreads = static_cast<int>(gridDim.x * blockDim.x);

      // Conversions from double are written out because a narrowing one into these types has
      // to be explicit, and they are loop invariant in any case.
      const T h    = T(1.0 / static_cast<double>(n));
      const T four = T(4.0);
      const T one  = T(1.0);

      // This thread walks the midpoints tid, tid + nthreads, ... rather than a contiguous
      // block of them, so neighbouring threads stay on neighbouring grid points. The step is
      // added rather than the grid point recomputed, which keeps the loop to the five
      // operations being priced with no integer-to-float conversion among them.
      T x        = (T(static_cast<double>(tid)) + T(0.5)) * h;
      const T dx = T(static_cast<double>(nthreads)) * h;

      T acc = T(0.0);
      for (long long i = tid; i < n; i += nthreads)
      {
        acc += four / (one + x * x);
        x += dx;
      }

      partials[tid] = acc;
    }

    // Widens a result to a double-double without assuming anything about its layout: the cast
    // takes the leading double, and subtracting it back off recovers whatever the type held
    // below that, exactly. For double the remainder is zero and this costs two operations.
    template <class T>
    [[nodiscard]] static cudax::fp64mp2 to_dd(const T& value)
    {
      const double head = static_cast<double>(value);
      const double tail = static_cast<double>(value - T(head));
      return cudax::fp64mp2(head) + cudax::fp64mp2(tail);
    }

    template <class T>
    [[nodiscard]] static double correct_digits(const T& value)
    {
      const cudax::fp64mp2 error = to_dd(value) - cudax::fp64mp2(kPiHi, kPiLo);
      const double magnitude     = std::fabs(static_cast<double>(error));
      return magnitude > 0.0 ? -std::log10(magnitude / kPiHi) : 99.0;
    }

    // Everything here is a device timing, so a missing GPU or a driver that cannot initialize
    // means there is nothing to report. Enumerating the devices is what first touches the
    // driver, so it is also where such an installation surfaces.
    [[nodiscard]] static bool device_available()
    try
    {
      return cuda::devices.size() != 0;
    }
    catch (const cuda::cuda_error&)
    {
      return false;
    }

    struct run_result
    {
      double value;
      double digits;
      double milliseconds;
    };

    template <class T>
    [[nodiscard]] static run_result run(cuda::stream_ref stream, cuda::device_ref device)
    {
      auto partials     = cuda::make_device_buffer<T>(stream, device, kThreadsTotal, cuda::no_init);
      const auto config = cuda::make_config(cuda::grid_dims(kBlocks), cuda::block_dims(kThreads));

      // A short untimed run first, so that the module load lands outside the timed window.
      cuda::launch(stream, config, pi_kernel<T>, kTerms / 64, partials.data());
      stream.sync();

      // A timed_event records on the stream as it is constructed, so these bracket the launch.
      cuda::timed_event start{stream};
      cuda::launch(stream, config, pi_kernel<T>, kTerms, partials.data());
      cuda::timed_event stop{stream};

      std::vector<T> host(kThreadsTotal);
      cuda::copy_bytes(stream, partials, cuda::std::span<T>{host.data(), host.size()});
      stream.sync();

      // The partials are summed in T, not in a wider type. A reduction more accurate than the
      // type being measured would report the loop alone, whereas a program written in T has to
      // pay for its own summation too, so keeping every step in T is what the type costs end to
      // end. It is also what separates the accuracy levels: the trailing limb matters far more
      // to a chain of this many additions than to the five operations of a single term, and
      // low, which is the level that skips the renormalization, is hit hardest.
      T sum = host[0];
      for (size_t i = 1; i < host.size(); ++i)
      {
        sum = sum + host[i];
      }
      const cudax::fp64mp2 value = to_dd(sum) * cudax::fp64mp2(1.0 / static_cast<double>(kTerms));

      const auto elapsed = stop - start;
      return {static_cast<double>(value), correct_digits(value), elapsed.count() / 1.0e6};
    }

    int main()
    try
    {
      if (!device_available())
      {
        printf("no usable CUDA device; this sample reports device timings, so there is nothing to do\n");
        return 0;
      }

      const cuda::device_ref device = cuda::devices[0];
      cuda::stream stream{device};

      // Name the GPU, since that is what decides the numbers. CUDA's device 0 need not be the
      // one nvidia-smi lists first, so on a machine with several GPUs this is the only
      // reliable label for the figures; CUDA_VISIBLE_DEVICES picks a different one.
      const auto device_name = device.name();
      const auto cc          = device.attribute(cuda::device_attributes::compute_capability);

      printf("pi by the midpoint rule, %lld terms, %d threads\n", kTerms, kThreadsTotal);
      printf("on %.*s, sm_%d%d\n\n",
             static_cast<int>(device_name.size()),
             device_name.data(),
             cc.major_cap(),
             cc.minor_cap());

      const run_result native   = run<double>(stream, device);
      const run_result selected = run<fp_t>(stream, device);

      printf("%-30s %-22s %8s %11s\n", "type", "value", "digits", "time (ms)");
      printf("%-30s %-22.17g %8.2f %11.3f\n", "double", native.value, native.digits, native.milliseconds);
      printf("%-30s %-22.17g %8.2f %11.3f\n", kFpName, selected.value, selected.digits, selected.milliseconds);

      if (selected.milliseconds > 0.0)
      {
        printf("\n%s is %.2fx the speed of native double, for %+.1f digits\n",
               kFpName,
               native.milliseconds / selected.milliseconds,
               selected.digits - native.digits);
      }

      return 0;
    }
    catch (const std::exception& failure)
    {
      printf("failed: %s\n", failure.what());
      return 1;
    }

Building it against a CCCL checkout takes all three include roots from that one checkout, because
the toolkit ships its own copies of CUB and Thrust and their version checks reject being paired
with a libcudacxx from somewhere else:

.. code-block:: bash

    nvcc -std=c++20 -arch=native \
      -I <cccl>/libcudacxx/include -I <cccl>/cub -I <cccl>/thrust pi.cu -o pi

On an RTX 6000 Ada, where FP64 runs at a fraction of FP32, the default type gives:

.. code-block:: text

    pi by the midpoint rule, 67108864 terms, 131072 threads
    on NVIDIA RTX 6000 Ada Generation, sm_89

    type                           value                    digits   time (ms)
    double                         3.1415926535897984        14.78       1.297
    cudax::fp32mp2                 3.1415926535931646        11.97       0.132

    cudax::fp32mp2 is 9.82x the speed of native double, for -2.8 digits

Ten times the throughput of ``double`` for around three of its digits, on a computation that is
nothing but arithmetic. Rebuilding with ``-DPI_FP_T=cudax::fp32mp2_low`` or
``...=cudax::fp32mp2_high`` walks the accuracy levels — 9.5 and 12.7 digits respectively, at 16×
and 6× — ``-DPI_FP_T=float`` shows what the hardware does unaided, and
``-DPI_FP_T=cudax::fp64mp2`` costs roughly five times ``double`` for 2.4 digits more than it. The
:ref:`fpmp page <libcudacxx-extended-api-fp-fpmp>` collects those figures across three
architectures.

Two things to expect when running it. Absolute times move with clocks and with what else is on the
GPU, so the ratio is the stable quantity, not the milliseconds. And ``-DPI_LOG2_TERMS=nn`` is worth
a look: ``double`` holds near 14 digits as terms are added, while ``float`` *degrades* — 5.2 digits
at 2\ :sup:`20` terms against 2.8 at 2\ :sup:`28`, so the extra work buys less answer rather than
more. With 24 significand bits it can neither place the grid points nor carry the sum. That
divergence is the wall the component exists to get past, and it takes one rebuild to see.
