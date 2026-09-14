CCCL FP Component Examples
==========================

The CCCL FP component provides floating-point types that give you arithmetic the hardware
does not offer directly: more precision than a `double`, `double` precision without FP64
units, less precision than any hardware implements, or the same arithmetic with a record of
what it did. Every type works in both host and device code from the same source.

Each example below prints the inputs and the result of every operation it performs. Nothing
is checked against a reference — the point is to show what each interface looks like in use,
so the output is meant to be read.

The components
--------------

| Component | Gives you | Documentation |
|---|---|---|
| **FPMP** | More mantissa than the hardware has, from pairs of IEEE floats — `fp32mp2` (~46 bits), `fp64mp2` (~104 bits) | [README_fpmp.md](README_fpmp.md) |
| **FPEMU** | IEEE-754 double precision built from integer and float operations, for hardware with limited or absent FP64 — `fp64emu` | [README_fpemu.md](README_fpemu.md) |
| **FP Tool: custom formats** | Any narrower format emulated on native FP64, to find out how much precision an algorithm needs — `fp64_custom<E, M>` | [README_fptool_custom.md](README_fptool_custom.md) |
| **FP Tool: statistics** | The same arithmetic as `fpmp2`, bit for bit, plus a record of the operations and numerical events — `fp32mp2_stat`, `fp64mp2_stat` | [README_fptool_stat.md](README_fptool_stat.md) |

Three of these change the arithmetic; the fourth observes it.

Which one do I want?
--------------------

**More precision than `double`.** `fp64mp2` — a double-double reaching ~104 mantissa bits
out of FP64 operations, where there is usually no binary128 hardware and a software
binary128 is expensive.

**Roughly `double` precision, but FP64 is slow on the target.** Two answers, and the choice
turns on what you need from `double`:

- `fp32mp2` if ~46 mantissa bits are enough and you can live with `float`'s exponent range.
  It is 8 bytes and runs on FP32 units.
- `fp64emu` if you need `double`'s full semantics — its range, its specials, and results
  identical to `double` bit for bit. Also 8 bytes in the packed form.

**Less precision, on purpose.** `fp64_custom<E, M>` rounds to a narrower format after every
operation, so a computation runs as though the hardware had that format. This is how to find
out whether an algorithm survives BF16, and whether it is the mantissa or the dynamic range
that matters.

**To know what the arithmetic is currently doing.** `fp32mp2_stat` / `fp64mp2_stat` compute
identical results and record operation counts, cancellation and overflow events, and what
the operands looked like — including whether the second limb is carrying anything at all.

The examples
------------

| Example | Shows |
|---|---|
| [`fpmp.cu`](fpmp.cu) | The `fpmp2` interface over both widths, one kernel each: arithmetic, mixed-type operations, math functions, the accuracy levels, and `renormalize()` |
| [`fpemu.cu`](fpemu.cu) | Both `fpemu` representations, one kernel each, and the accumulation where the unpacked form's deferred rounding separates it from `double` |
| [`fptool_custom.cu`](fptool_custom.cu) | The named formats at compile time, and the same sweep driven by a run-time size setter |
| [`fptool_stat.cu`](fptool_stat.cu) | A Kahan-compensated sum with one instrumented variable, and how to read the resulting record |

Each one reports twice, once from the host and once from the device. The functions that do
the arithmetic are `__host__ __device__`, so a single build carries both copies and a single
run exercises both, which is also the clearest demonstration that these types need no
separate host implementation.

A convention shared by all four
-------------------------------

The CCCL FP component lives in the `cuda::experimental` namespace, which will be promoted to
`cuda::` later. The examples abbreviate it rather than pulling it in wholesale:

```c++
namespace cudax = cuda::experimental;
```

Two spellings then appear, and the split is deliberate. Type names and the component's own
functions — `renormalize`, the accuracy-selecting `add<>`, the statistics and size control
functions — carry the `cudax::` prefix, since they have no counterpart for `double` and so
only occur in code written against the component to begin with. The standard-named math
functions are left **unqualified** and found by argument-dependent lookup:

```c++
cudax::fp64mp2 x{2.0};
auto r = sqrt(x);                  // unqualified: ADL finds the fpmp2 overload
auto s = cudax::renormalize(x);    // component-specific: qualified
```

That is what lets an existing body of `double` code keep its call sites unchanged when the
type underneath it is swapped.

Building and running
--------------------

These sources are built by the enclosing project in [`..`](..), which acquires CCCL through
CPM and so builds the way a consumer of the released library would. From that directory:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

That produces one executable and one CTest per example. Each prints its report and returns
success, so a passing CTest run means every example ran. A single one can be selected with
`ctest -R fpmp`, or its executable run directly to read the output.

A CUDA compiler is required, since the sources are CUDA and always contain kernels. The
architecture follows `CMAKE_CUDA_ARCHITECTURES`, which the enclosing project defaults to
`native`. Running without a visible GPU is not an error: the device section is skipped and
the host report is still printed.

The arithmetic is the subject of these examples, but the code around it is written the way
CCCL would have you write it: `cuda::devices` to find a GPU, a `cuda::stream` to submit on,
`cuda::make_pinned_buffer` for the results the kernels write back, and `cuda::launch` for the
kernels themselves.
