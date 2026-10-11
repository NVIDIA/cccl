.. _libcudacxx-extended-api-fp-fptool-stat:

fpmp2_stat: What the Arithmetic Actually Did
============================================

An ``fpmp2_stat`` type is a drop-in replacement for the ``fpmp2`` type it instruments. It computes
**bit-identical results** — it is the same arithmetic, wrapped — and additionally records on the
device what that arithmetic did: how many operations of each kind ran, which numerical events
occurred, and what the operands and results looked like.

It answers questions that are otherwise guesswork. Is the second limb of this pair actually
carrying information, or am I paying 16 bytes to store a ``double``? Is the ``low`` accuracy level
safe on this data? Did the region I think I measured run the number of operations I think it did?
Is anything silently corrupt?

Because results are bit-identical by construction, an instrumented run can be compared against a
plain one — and if they differ, the cause is a race or an uninitialized value, not a rounding
difference.

.. seealso::
   :ref:`fptool overview <libcudacxx-extended-api-fp-fptool>` — the header shared with
   ``fp64_custom``, and why neither is a type to ship in.

   :ref:`fpmp <libcudacxx-extended-api-fp-fpmp>` — the types being instrumented, and the accuracy
   levels these wrappers select among.

The types
---------

.. list-table::
   :widths: 36 32 32
   :header-rows: 1

   * - **Instrumented type**
     - **Wraps**
     - **Accuracy**

   * - ``fp32mp2_stat``
     - ``fp32mp2``
     - ``def``

   * - ``fp32mp2_stat_low``
     - ``fp32mp2_low``
     - ``low``

   * - ``fp32mp2_stat_mid``
     - ``fp32mp2_mid``
     - ``mid``

   * - ``fp32mp2_stat_high``
     - ``fp32mp2_high``
     - ``high``

   * - ``fp64mp2_stat``
     - ``fp64mp2``
     - ``def``

   * - ``fp64mp2_stat_low``
     - ``fp64mp2_low``
     - ``low``

   * - ``fp64mp2_stat_mid``
     - ``fp64mp2_mid``
     - ``mid``

   * - ``fp64mp2_stat_high``
     - ``fp64mp2_high``
     - ``high``

Since ``fpmp2_accuracy::def`` is ``mid``, ``fp32mp2_stat`` and ``fp32mp2_stat_mid`` are the same
type, not two.

The wrapper keeps the layout promise that makes it a drop-in: the same ``sizeof`` and ``alignof`` as
the wrapped type, still trivially copyable, and ``cuda::std::numeric_limits`` specialized to report
exactly what the wrapped type reports. ``base_type`` names the type being instrumented.

Moving between the two
~~~~~~~~~~~~~~~~~~~~~~

At the same accuracy level, conversion is implicit in **both** directions, so an instrumented value
passes to an interface expecting the plain type and back without ceremony. ``as_fpmp2()`` gets at
the wrapped value directly — the ``const`` overload for handing it to an API taking ``fpmp2``, and a
mutable overload whose modifications are, by their nature, not instrumented.

Conversions that change the accuracy level are explicit in every direction, mirroring ``fpmp2``
itself, and copy the limbs without renormalizing: ``fp32mp2_stat_low(x)``.

Selective instrumentation
-------------------------

The wrapper does not have to cover the whole computation. Two rules govern how far it reaches:

1. An operation is counted when **either** operand is instrumented.
2. The **result is instrumented too**, so the property propagates along an expression until a plain
   variable absorbs it.

That is what makes the tool precise rather than merely loud. Instrument the one variable under
suspicion and leave the rest as plain ``fpmp2``, and the counters describe that variable's
arithmetic instead of the whole kernel's:

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    cudax::fp32mp2      sum{0.0f}, term{1.0f};
    cudax::fp32mp2_stat compensation{0.0f};   // the only instrumented variable

    auto y = term - compensation;   // counted: compensation is instrumented, y becomes so
    auto t = sum + y;               // counted: y is instrumented, t becomes so
    sum    = t;                     // plain variable: the spread stops here
    term   = term * ratio;          // never counted, both sides plain

Mixing an instrumented value with a plain one, or with a built-in scalar, works in either order and
is counted, returning the instrumented type.

What is counted
~~~~~~~~~~~~~~~

Four operations, individually, with ``ops_count`` their sum:

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - **Counter**
     - **Fed by**

   * - ``add_count``
     - ``+``, ``+=`` (pair and single-limb), ``++``, ``atomicAdd``

   * - ``sub_count``
     - ``-``, ``-=`` (pair and single-limb), ``--``, ``atomicSub``

   * - ``mul_count``
     - ``*``, ``*=``

   * - ``div_count``
     - ``/``, ``/=``

Increment and decrement route through ``+= 1`` and ``-= 1``, so they land in ``add_count`` and
``sub_count`` rather than in counters of their own.

What is deliberately not counted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unary negation, since it is exact. All comparisons. ``renormalize``. ``sqrt`` and ``rsqrt``. ``fma``
and ``mad``. Every math function — ``exp``, ``log``, ``pow``, ``sin`` and the rest. The warp
shuffles, being thread cooperation rather than arithmetic. And the mutable ``as_fpmp2()`` accessor,
which hands out a reference whose modifications are invisible.

The composites are excluded on purpose: the operations inside them would swamp the counters of the
algorithm under study. What they call internally is invisible either way, since those internals use
plain ``fpmp2`` and never reach the record.

Running a measurement
---------------------

The header is opt-in, because the counter record is a mutable object at namespace scope with one
copy shared by every translation unit, and every instrumented operation updates it. Define
``CCCL_ENABLE_FPTOOL`` for the whole project — see
:ref:`the component page <libcudacxx-extended-api-fp-fptool>` for why it has to be the whole
project:

.. code-block:: bash

    nvcc -DCCCL_ENABLE_FPTOOL ...

.. code-block:: cuda

    #include <cuda/fptool>

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    cudax::fpmp2_stat_reset_device_data(stream);
    cuda::launch(stream, config, my_kernel, ...);
    const auto record = cudax::fpmp2_stat_read_device_data(stream);

Both functions take a stream and are host-only. The reset enqueues a cleared record, so counting
starts with the next kernel on that stream and **no synchronization is needed**. The read enqueues
the copy back and then synchronizes, so everything already enqueued on that stream is finished and
counted — work on *other* streams is your responsibility. Both throw ``cuda::cuda_error`` on a
failed copy. The stream also identifies the device, the record being per-device state.

Three things to plan around
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Collection is device-only.** The same source compiles and runs on the host, where the wrapper is a
transparent pass-through and nothing is gathered. A program that does part of its arithmetic on the
host will therefore report a count below what its closed form predicts, with no diagnostic. The
control functions do not exist at all in a host-only build or under NVRTC, so guard the measurement
rather than the computation.

**The counters are shared by every instantiation.** ``fp32mp2_stat`` and ``fp64mp2_stat`` write to
the same record, as do all eight aliases, and a program using more than one sees their operations
summed. Measure one type at a time, bracketing each run with its own reset. That sharing extends
across translation units only under relocatable device code; compiled as a whole program, each
translation unit gets its own copy of the record, and a reset or read then sees only its own — which
shows up as a count below what the algorithm predicts.

**Use a small grid.** The record is device-wide and updated with atomics, and same-address atomic
throughput does not scale with the number of SMs, so an instrumented run gains nothing from a large
grid. A single block of 32 threads and a few thousand iterations exercise the same code paths as a
full run.

Bracket the region tightly. Setup kernels, warm-up iterations and validation passes otherwise land
in the same record and confuse every share you compute from it.

There is no macro that switches instrumentation off. The way to turn it off is to switch the type
alias back to plain ``fpmp2``, which is worth keeping behind a build flag.

What gets recorded
------------------

``fpmp2_stat_read_device_data`` returns an ``fpmp2_stat_data`` holding nine counters and four value
summaries.

Numerical events
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - **Counter**
     - **Means**

   * - ``full_cancel_count``
     - An additive operation returned an exact zero

   * - ``partial_cancel_count``
     - An additive operation dropped more than half the significand in magnitude, without reaching
       zero

   * - ``underflow_count``
     - A multiply or divide returned an exact zero

   * - ``overflow_count``
     - Any of the four operations returned a non-finite result — NaN as well as infinity

All four require both operands to be finite and non-zero, which is why a division by zero is not
counted as an overflow and a zero operand passing through a multiply is not counted as an underflow.
The additive/multiplicative split needs no heuristic: the difference of two distinct values is a
non-zero multiple of the smaller one's ulp and so never rounds to zero, which makes an additive zero
*proof* that the operands were equal and opposite, while multiplication and division have nothing to
cancel at all.

The partial-cancellation threshold is half the **pair's** significand, ``digits / 2`` — so 23 bits
for ``fp32mp2`` and 52 for ``fp64mp2``, being half of 46 and 104 rather than any single limb's field
width. It measures the drop in *magnitude*, which is not quite the same as bits gone from the
representation.

The value slots
~~~~~~~~~~~~~~~

Each counted operation contributes one value to each of three slots: ``arg[0]``, ``arg[1]`` and
``result``. For a binary operator the mapping is left-then-right; for compound assignment the old
value of the target is ``arg[0]`` and the operand ``arg[1]``; for ``atomicAdd`` the old value from
memory is ``arg[0]``. A fourth slot, ``arg[2]``, is reserved for a future ternary operation and is
never written — treat it as reserved rather than reading it.

Each slot carries:

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - **Field**
     - **Records**

   * - ``max_exp``, ``min_exp``
     - Range of the leading limb's exponent

   * - ``zero_count``
     - Both limbs zero

   * - ``zero_lo_count``
     - Finite, non-zero ``hi``, zero ``lo``

   * - ``inf_count``, ``nan_count``
     - An infinite or NaN limb, once per value

   * - ``infnan_count``
     - Both limbs infinite with opposite signs

   * - ``denorm_count``
     - A subnormal limb, once per value

   * - ``overlap_count``
     - The limb gap was negative

   * - ``invert_count``
     - ``abs(lo) > abs(hi)``, strictly

   * - ``max_hi_lo_gap``, ``min_hi_lo_gap``
     - Range of the gap between the limbs

   * - ``min_hi_lo_gap_sample_hi``, ``min_hi_lo_gap_sample_lo``
     - The limbs of a value that lowered the gap minimum

Note that ``max`` is declared before ``min`` in both ranges, which is easy to get backwards when
writing a ``printf``.

Not every field is sampled on every value. The exponent range, the gap block and ``invert_count``
are only recorded for finite, non-zero values, so a NaN contributes to ``nan_count`` but not to the
ranges — and an inverted NaN pair is never counted as inverted. The gap additionally requires both
limbs to be non-zero: a value with a zero ``lo`` has no gap and is covered by ``zero_lo_count``
instead. The gap itself is measured against the **limb's** mantissa width, in contrast to the
partial-cancellation threshold above, which uses the pair's.

Reading a record
----------------

The one interpretation mistake to avoid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The cancellation counters count occurrences, not damage.** A cancelling subtraction is itself
exact and loses no accuracy of its own; what makes cancellation harmful is rounding already present
in the operands, which a single operation cannot see. Kahan and Neumaier summation run up one
cancellation per term *while being at their most accurate* — the cancellation is how they work. A
high ``full_cancel_count`` is not an error count, and treating it as one will send you chasing a
problem that isn't there. The :ref:`example below <libcudacxx-extended-api-fp-fptool-stat-example>`
is deliberately such a case.

Sentinels
~~~~~~~~~

The four range fields are armed inverted so that an untouched range is detectable: ``max_exp`` and
``max_hi_lo_gap`` start at ``INT_MIN``, ``min_exp`` and ``min_hi_lo_gap`` at ``INT_MAX``. An
unsampled range therefore prints as ``[2147483647 .. -2147483648]`` if you print it raw. The test
for "nothing sampled" is ``min > max``.

The two gap-sample fields have **no sentinel**. They are zero-initialized, so ``hi=0 lo=0`` is
indistinguishable from a real sample, and there is no way to tell from those fields alone. Gate on
``min_hi_lo_gap <= max_hi_lo_gap`` before reporting them.

The counters have no sentinels: zero means zero.

Which field answers which question
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 46 54
   :header-rows: 1

   * - **Question**
     - **Look at**

   * - Is the pair type earning its cost?
     - ``zero_lo_count`` as a share of ``ops_count``

   * - Am I measuring the intended region?
     - ``ops_count`` against your own closed form

   * - Is ``low`` accuracy safe on this data?
     - ``overlap_count``, ``min_hi_lo_gap``, ``invert_count``

   * - Where does the precision go?
     - ``full_cancel_count``, ``partial_cancel_count``

   * - Is the dynamic range near an edge?
     - ``min_exp``/``max_exp``, ``denorm_count``

   * - Is anything silently corrupt?
     - ``invert_count``

Two fields deserve singling out. ``zero_lo_count`` is the best single indicator of whether a pair
type is earning its cost: a high share means the second limb is sitting idle and a cheaper type may
do. And ``invert_count`` is the one number that should always be zero — the tail outweighing the
head means anything reading the pair through ``hi`` alone gets the wrong answer, and every later
operation inherits it.

Validating ``ops_count`` against a closed form is the cheapest sanity check available, and it is
worth doing first. Comparing the same workload across ``_low``, ``_mid`` and ``_high`` is the most
informative use of the tool.

Ranges go with the counter beside them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A range says what the worst case was and never how often it happened, so read each one next to its
counter: ``denorm_count`` with the bottom of the exponent range, ``overlap_count`` and
``invert_count`` with the bottom of the gap range. A minimum of −3 seen once in a billion values and
one seen in every second value call for opposite responses.

``denorm_count`` doubles as a domain check: the ``low`` and ``mid`` accuracy levels support the
normal range only, so any subnormal means such a configuration is being used outside its domain.

Two caveats on shares. ``overlap_count`` and ``invert_count`` are gated on having a gap to measure,
so dividing them by ``ops_count`` under-estimates them — the true denominator is not exposed. And
overlap and inversion are not exclusive to the ``low`` level: arithmetic only produces them there,
but the two-limb constructor can produce them at any accuracy level.

Cost and limits
---------------

An instrumented run is **two to three orders of magnitude slower** than the plain type. Every
operation updates the same device-wide record with atomics, and same-address atomic throughput does
not scale with the number of SMs. Statistics are a diagnostic, not something to leave switched on.

Two further limits worth knowing. The gap *sample* is best-effort under concurrency: two threads
lowering the minimum simultaneously may leave the sample of either, though ``min_hi_lo_gap`` itself
is exact. And atomics are unsupported in tile code, so the instrumented types are unavailable there.

A measurement workflow
----------------------

1. Change only the type alias, leaving the algorithm alone.
2. Bracket the region tightly, so setup and validation passes stay out of the record.
3. Reduce the problem size and the grid — one block of 32 threads is plenty.
4. Validate ``ops_count`` against your own closed form before reading anything else.
5. Read the per-slot counts as shares of ``ops_count``, minding the two exceptions above.
6. Compare the same workload across ``_low``, ``_mid`` and ``_high``.
7. Keep a plain-type run alongside. Results are bit-identical by construction, so a mismatch means
   a race or an uninitialized value rather than a rounding difference.

.. _libcudacxx-extended-api-fp-fptool-stat-example:

The example
-----------

`fptool_stat.cu <https://github.com/NVIDIA/cccl/blob/main/examples/cudax/fp/fptool_stat.cu>`__ runs
the same computation once per width, so its two kernels — ``float_float_series`` and
``double_double_series`` — differ only in the type they are written in, with a separate
reset/run/read bracket for each because the counters are shared. It reports from the host as well as
from the device, which shows the other half of the device-only rule: the same instrumented code
compiles and runs unchanged on the host and gathers no counters there.

The measured computation is a **Kahan-compensated sum** over 64 terms, with the compensation term as
the single instrumented variable, and both choices are deliberate. Whether the compensation is doing
anything is the question worth asking about such a loop, and instrumenting only that variable is what
lets the counters answer it. Its ``fp64mp2_stat`` run reports:

.. code-block:: text

    operation counters
      total            = 259
      add              = 65
      sub              = 193
      mul              = 0
      div              = 1

    numerical events
      cancellation, full    = 64
      cancellation, partial = 11
      underflow to zero     = 0
      overflow              = 0

    value statistics, over the 259 counted operations
      arg[0]   exponent range [-63 .. 1], hi/lo limb gap [1 .. 11]
               zero=1  inf=0  nan=0  infnan=0  denorm=0  overlap=0 (0.0%)  invert=0 (0.0%)  zero_lo=234 (90.3%)
               tightest pair: hi=2  lo=-1.1102230246251565e-16
      arg[1]   exponent range [-63 .. 1], hi/lo limb gap [1 .. 10]
               zero=66  inf=0  nan=0  infnan=0  denorm=0  overlap=0 (0.0%)  invert=0 (0.0%)  zero_lo=183 (70.7%)
               tightest pair: hi=2  lo=-1.1102230246251565e-16
      result   exponent range [-63 .. 1], hi/lo limb gap [1 .. 11]
               zero=64  inf=0  nan=0  infnan=0  denorm=0  overlap=0 (0.0%)  invert=0 (0.0%)  zero_lo=181 (69.9%)
               tightest pair: hi=2  lo=-1.1102230246251565e-16

Four things in that record are worth walking through, because each one is a reading the tool exists
to support.

``mul`` **is zero**, on a series whose terms are formed by multiplication. That is selective
instrumentation working as described: the series multiplication stays plain on both sides, so it is
never counted however many terms run, and the counters describe the compensation rather than the
kernel. The counts also validate against a closed form, which is step 4 of the workflow above:
64 terms of Kahan summation are one addition and three subtractions each, and the code then
finalizes the sum with one more subtraction, scales it with one division, and adds one ``exp()``
result — giving exactly the 65, 193 and 1 reported. The ``exp()`` itself contributes nothing, being
on the not-counted list.

``cancellation, full = 64`` **is one per term, and nothing is wrong.** This is the interpretation
trap made concrete: a compensated sum is *most* accurate exactly when it is cancelling, because the
cancellation is how it recovers the rounding error of the previous step. Read as an error count this
record would look alarming; read correctly it says the compensation is doing its job on every
iteration.

``zero_lo`` **at 90.3% says the pair is barely being used** by this variable. Nine times in ten the
instrumented operand's second limb was empty, which is the signal that a double-double is more type
than the compensation term needs. That is the ``zero_lo_count`` reading from the table above, on a
case where it has something to say.

``overlap`` **and** ``invert`` **are both zero**, and the gap range never drops below 1, so the
limbs stay properly separated throughout — the check that says this data would also be safe at the
``low`` accuracy level, which is the level that can produce overlap.

Running the same source as ``fp32mp2_stat`` gives an instructive contrast. The operation counters
are identical to the last digit, the algorithm and its control flow being unchanged, but
``cancellation, partial`` rises from 11 to 40 and the limb gap widens from ``[1 .. 11]`` to
``[1 .. 40]``. Both numbers are predictable, which is what makes them a good check on the tool.
The series halves each term, so the term falls below one limb's worth of the running sum after as
many iterations as that limb has mantissa bits — 53 for ``fp64mp2``, 24 for ``fp32mp2`` — and every
iteration after that loses more than half the significand in the ``t - sum`` step. Out of 64 terms
that leaves 64 − 53 = 11 partial cancellations at the wider type and 64 − 24 = 40 at the narrower
one, exactly as reported.

That is what "compare the same workload across types" produces: the operation counts confirm you
are measuring the same thing, and the event counters show what changed — here, that the same
algorithm crosses into lossy territory forty iterations early when the pair is built from
``float``.
