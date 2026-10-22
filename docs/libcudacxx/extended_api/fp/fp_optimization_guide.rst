.. _libcudacxx-extended-api-fp-optimization-guide:

FP Optimization Guide - Choosing a Type and Using It Well
=========================================================

The CCCL FP component offers several ways to get arithmetic the hardware does not sell directly, and
they are not interchangeable: each one spends a different pipe, holds a different number of bits,
and keeps or drops different parts of ``double``'s semantics. This guide is the cross-component
view. It says how to choose between them, how the types combine inside one kernel, how to get the
most out of each, and how to use the two instruments so that those decisions rest on measurements
rather than guesses. The component pages - :ref:`fpmp <libcudacxx-extended-api-fp-fpmp>`,
:ref:`fpemu <libcudacxx-extended-api-fp-fpemu>`,
:ref:`fptool custom <libcudacxx-extended-api-fp-fptool-custom>` and
:ref:`fptool stat <libcudacxx-extended-api-fp-fptool-stat>` - describe each type; this document
assumes them and does not repeat them.

.. _libcudacxx-extended-api-fp-optimization-guide-choosing:

Choosing the type
-----------------

Four questions settle most choices, and they are worth answering in this order.

1. **What does the target do to FP64?** The rate at which the part runs FP64 against FP32 decides
   more than anything in the code does. The CUDA Best Practices Guide gives it per compute
   capability, as results per clock per SM: 1:2 on A100, H100 and B200, and 1:64 on every
   part that rations FP64 - L40S, B300, RTX PRO 6000 Blackwell and the consumer GPUs - where FP64
   issues 2 results per clock per SM against 128 for FP32. Everything below ``double`` in the table
   is a way of avoiding the FP64 pipe and pays only where it is rationed; ``fp64mp2`` spends it and
   pays only where it is not.
2. **How many significand bits does the computation need?** ``float`` has 24, ``fp32mp2`` 46,
   ``double`` 53, ``fp64emu_unpacked`` 62 between operations, ``fp64mp2`` 104, binary128 113 - all
   counted with the implicit leading bit, as ``cuda::std::numeric_limits::digits`` counts them, so
   the pair figures are 2×24 − 2 and 2×53 − 2 rather than stored-bit totals. When the answer is not
   known, :ref:`fp64_custom <libcudacxx-extended-api-fp-optimization-guide-fptool>` exists to find
   it.
3. **Which of** ``double`` **'s semantics must survive?** Its exponent range, which ``fp32mp2`` does
   not provide; bit-for-bit reproducibility, which only ``double`` itself and packed ``fp64emu`` at
   ``high`` offer; subnormals, infinities and NaNs full IEEE-754 compliance which the ``low`` and
   ``mid`` levels of both arithmetic components drop.
4. **What else does the kernel need?** Transcendentals, which ``fpmp`` has and ``fpemu`` does not;
   atomics; bytes per value, which decide bandwidth and occupancy; and whether the same code has to
   run on the host.

The table puts the answers together. Speed figures are from the π integration the component pages
measure, relative to native ``double`` on the same part, and "rationed" means a part whose FP64 rate
is 1:64.

.. list-table::
   :widths: 32 8 18 43
   :header-rows: 1

   * - **Need**
     - **FP64 on the target**
     - **Reach for**
     - **Why**

   * - More than 104 significand bits, or a wider exponent range than ``double``'s
     - any
     - software binary128 (``__float128``) on the device sm_100 and later
     - nothing narrower holds it

   * - Beyond ``double``, up to 104 bits
     - full rate
     - ``fp64mp2``
     - 7.8× faster than software binary128 per ``mid`` addition on B200, 17.8× per division

   * - Beyond ``double``, up to 104 bits
     - rationed
     - software binary128, or ``double`` with a compensated algorithm
     - binary128 is integer code and runs 1.4–4× faster than ``fp64mp2`` there; :ref:`fp64mp2
       <libcudacxx-extended-api-fp-optimization-guide-fp64mp2>` has the register-pressure exception

   * - An x87 ``long double`` port, 64-bit significand
     - full rate
     - ``fp64mp2``
     - 104 bits covers 64 with room; the exponent range does not carry over

   * - Extra precision between operations only, result stored as ``double``
     - rationed
     - ``fp64emu_unpacked``
     - 62-bit intermediates rounded once at the pack; about 1.5× ``double``

   * - ``double`` results bit for bit, without the FP64 pipe
     - rationed
     - ``fp64emu``, packed, at its default ``high``
     - identical results off the FP64 pipe; near ``double``'s speed only in small kernels

   * - About ``double`` precision, ``float``'s range is enough
     - rationed
     - ``fp32mp2``
     - 46 bits in 8 bytes on the FP32 pipe; 5.5–16× ``double`` by level

   * - ``float`` precision over ``double``'s exponent range
     - rationed
     - ``fp64emu_low``
     - the cheapest emulated arithmetic, about 2× ``double``, reaching ±10\ :sup:`308`

   * - Up to ``double`` precision
     - full rate
     - native ``double``
     - every emulation only slows the mix down

   * - To find out how much precision an algorithm needs
     - any
     - ``fp64_custom<E, M>``
     - sweeps formats within ``double`` layout, ~ same cost

   * - To know what the pair arithmetic is doing
     - any, on the device
     - ``fp32mp2_stat``, ``fp64mp2_stat``
     - bit-identical results + statistic record

Three forks in that table deserve more than a row:

1. **Around** ``double`` **on a rationed HW target**, three types compete, and the trade is sharp.
   ``fp32mp2`` is the fastest by a wide margin and carries a full math API, but holds 46 bits in
   ``float``'s exponent range, so it overflows at 10\ :sup:`38` and asks for a cast wherever a
   ``double`` enters. Packed ``fp64emu`` keeps every bit of ``double``'s semantics and converts as
   freely as ``double`` does, but at its best it only matches hardware ``double``'s speed and in a
   large kernel it falls behind - its point is exactness without the FP64 pipe, not throughput.
   ``fp64emu_unpacked`` is the one that beats ``double`` on both speed and accuracy over a chain, at
   16 bytes per value, with explicit conversions, without bit-for-bit agreement with ``double``.

2. **Above** ``double``, the choice between ``fp64mp2`` and ``fp64emu_unpacked`` is a choice about
   how much extra precision and where it lives. The pair carries 104 bits in the stored value, so a
   result keeps them; the unpacked form carries 62 between operations and rounds to 53 at the pack,
   so only the chain benefits. The second is far cheaper and runs on the FP32 and integer pipes, so
   it is the one to consider on a rationed part when the goal is a ``double`` result more accurate
   than ``double`` arithmetic would give, rather than a wider result.

3. **Below** ``double``, ``fp64emu_low`` is ``float``-precision arithmetic with ``double``'s range,
   for computations such as power series whose intermediates leave ``float``'s exponent long before
   the sum converges.

.. _libcudacxx-extended-api-fp-optimization-guide-mixed:

Mixed cases
-----------

The types are designed to be mixed, some real kernels end up using two of them: one for the bulk of
the work and one for the part that sets the accuracy. The patterns below are the ones the component
documentation already relies on, collected in one place.

- **A fast level for the kernel, a careful step inside it.** The accuracy level is part of an
  ``fpmp`` type, but ``add<>``, ``sub<>``, ``mul<>``, ``div<>``, ``fma<>`` and ``mad<>`` take it per
  operation and return the operand type, so a kernel written in ``fp32mp2_low`` can run one addition
  at ``high`` without changing anything else. Argument reductions and the subtraction of nearly
  equal values are the usual candidates. The reverse mixing - a ``low`` value handed to ``mid`` or
  ``high`` code - is handled by the conversion, which renormalizes on the way; see
  :ref:`Accuracy levels <libcudacxx-extended-api-fp-fpmp-accuracy>`.

- ``double`` **data, a wider accumulator.** Inputs that are exact as ``double`` stay in ``double``
  arrays; only the running sum, or the sum of products, outgrows them. An ``fp64mp2`` accumulator
  fed with ``acc += d`` is compensated summation written for you, at 10 FP64 instructions per term,
  and the terms never leave their 8-byte storage. On a rationed part the same role goes to
  ``fp64emu_unpacked``, which absorbs the summation error in its guard bits and rounds once at the
  end.

- **Packed storage, unpacked arithmetic.** With ``fpemu``, the 8-byte packed form belongs in memory
  and the 16-byte unpacked form in registers: unpack once on load, run the chain, pack once on
  store. ``fma``, ``mad``, ``dot`` and ``cmul`` are fused for the same reason, so that intermediate
  products are not rounded and repacked between steps. Alternating the two representations inside a
  loop pays the pack/unpack tax the unpacked form exists to avoid, and a ``double`` literal in an
  unpacked expression is such an alternation, since entering the unpacked form is a conversion
  written out; hoist constants as ``fp64emu_unpacked`` values.

- **Emulated arithmetic, native transcendentals.** ``fpemu`` has ``+ - * /``, ``fma``, ``mad``,
  ``sqrt``, ``dot`` and ``cmul``, and nothing else. A kernel that needs an occasional ``exp`` or
  ``sin`` converts to ``double``, calls the native function and converts back - one FP64 call per
  element, acceptable where the call is rare - moreover this case may get performance benefits from
  GPU pipes parallelization - see the next case.

- **Two pipes at once.** On a rationed part the FP64 units sit idle while ``fpemu`` runs on the FP32
  and integer pipes, and would still deliver their own rate if asked, so a reduction split between
  native ``double`` and ``fpemu`` could in principle sum the two throughputs -
  :ref:`the fpemu page <libcudacxx-extended-api-fp-fpemu>` measures ``fp64emu_unpacked_mid`` at
  1.78–1.96× ``double``, so the sum would be 2.8–3.0× ``double``. No such kernel has been built or
  measured, and the two halves share issue bandwidth, registers and the scheduler, so treat it as an
  upper bound to test rather than a result to plan on. Packed ``high`` is the partner when the split
  must not change the result: it holds ``double``'s layout and returns ``double``'s bits, so the
  handover costs nothing and the two halves agree. The unpacked form is the partner when throughput matters more, its pack boundary
  being where the handover would sit.

- **Size first, then choose.** When the precision a computation needs is not known,
  ``fp64_custom<11, M>`` answers it before any type is committed to: sweep ``M`` downward with the
  reference reduction kept wide, find where the digits give way, and read the result against the
  table above - 24 bits point at ``float`` or ``fp64emu_low``, up to 46 at ``fp32mp2``, 53 at
  ``double`` or ``fp64emu``, more at ``fp64mp2``. A separate sweep of the exponent axis says whether
  ``float``'s range would have been the problem.

.. _libcudacxx-extended-api-fp-optimization-guide-fpmp:

Using fpmp efficiently
----------------------

``fp32mp2`` is written to take ``double``'s place in existing code, and ``fp64mp2`` to take the
place of a wider CPU type - binary128, or the x87 80-bit ``long double`` of code being ported to the
GPU - but neither has the cost model of the type it replaces: a pair operation is a run of hardware
operations on the limbs, and how long a run depends on the operation and on the accuracy level. The
table gives the length, in instructions on the floating-point pipe of the component type, from the
SASS counts in the :doc:`specification <fpmp_spec>` - one representative count per operation. The
algorithms are the same for both widths, so the counts are too; what differs is the pipe they
occupy, FP32 for ``fp32mp2`` and FP64 for ``fp64mp2``, and that is the whole of the hardware
question.

.. list-table::
   :widths: 38 19 25 19
   :header-rows: 1

   * - **Operation**
     - ``low``
     - ``mid``
     - ``high``

   * - ``+``, ``-``
     - 8
     - 11
     - 20

   * - ``+=``, ``-=`` with a one-limb scalar
     - 7
     - 10
     - 13

   * - ``*``
     - 5
     - 9
     - 9

   * - ``mad``
     - 13
     - 16
     - 29

   * - ``fma``
     - 16
     - 19
     - 37

   * - ``/`` on ``fp32mp2``
     - 6
     - 13
     - 23, plus 23 integer

   * - ``/`` on ``fp64mp2``
     - 11, plus 27 integer
     - 18, plus 25 integer
     - 28, plus 49 integer

   * - ``sqrt``, one implementation for all levels
     - —
     - 18 (``fp32mp2``), 23 plus 25 integer (``fp64mp2``)
     - —

   * - ``double`` → ``fp32mp2``
     - —
     - 25, integer & fp32
     - —

   * - ``fp32mp2`` → ``double``
     - —
     - 40, integer & fp32
     - —

Three things follow from the table before any type-specific advice:

1. Multiplication costs the same at ``mid`` and ``high``, so ``high`` is paid for on additions and
   divisions.

2. Division is the expensive basic operation, and the one ``high`` makes most expensive.

3. And for ``fp32mp2``, a conversion to or from ``double`` costs as much as two to four additions -
   on B300 the two conversions run at 1290 and 734 billion per second against 2782 billion ``mid``
   additions.

The statistics type, ``fp32mp2_stat`` / ``fp64mp2_stat`` from ``<cuda/fptool>``, is the tool several
of the points below lean on. It computes bit-identical results and records, per instrumented
variable, the operation mix, the cancellation and overflow events, and how the limbs were placed;
:ref:`Using fptool <libcudacxx-extended-api-fp-optimization-guide-fptool>` below and
:ref:`the fpmp2_stat page <libcudacxx-extended-api-fp-fptool-stat>` say how to read the record. It
is a diagnostic rather than a mode: an instrumented run is two to three orders of magnitude slower,
so instrument one variable on a reduced problem and build without it for the runs that count.

.. _libcudacxx-extended-api-fp-optimization-guide-fp32mp2:

fp32mp2
~~~~~~~

- **Use it only where FP64 throughput is much lower than FP32.** The type pays for its digits in
  FP32 instructions, and the bargain holds only where those are cheap relative to FP64. On the
  FP64-rationed parts - L40S, RTX PRO 6000 Blackwell, B300, consumer GPUs generally - a ``mid``
  addition runs 5.6× faster than a native ``double`` addition, and the well-known π integration
  benchmark on :ref:`the fpmp page <libcudacxx-extended-api-fp-fpmp-performance>` comes out
  8.6–10.8× ahead. On a full-rate part such as B200 the same addition runs at 0.36× of native
  ``double``: ``double`` is both faster and more accurate there, and the float pair has nothing to
  offer. The code is portable; the win is not. Decide from the FP64:FP32 rate of the target, and
  keep ``double`` reachable behind a type alias.

- **Keep** ``double`` **↔** ``fp32mp2`` **conversions out of hot loops.** A conversion is
  arithmetic, not a register move: about 25 instructions into the pair and 40 out of it. They hide
  in two places. One is a ``double`` *variable* in a pair expression - ``x * scale`` with ``double
  scale`` - which the mixed-type operator converts on every evaluation before running the full pair
  operation; a literal can be folded at compile time, a variable cannot. Hoist such scalars into
  ``fp32mp2`` values before the loop. The other is storage. An array of ``fp32mp2`` is 8 bytes per
  element, exactly like ``double``, so keeping the data in pair form costs no bandwidth and saves a
  conversion on every load and store; convert once, at the I/O boundary.

- **Stay at** ``mid``, **and drop to** ``low`` **only on the paths that dominate the profile.**
  ``mid`` renormalizes after every operation, so results do not degrade along a chain, at 11
  instructions per addition against 8 for ``low`` and 20 for ``high``. ``low`` saves a quarter on
  additions and nearly half on multiplications by skipping the closing ``fast_two_sum``, and the
  price is drift: the limbs are free to overlap, and along a long chain part of the trailing limb
  stops carrying information. The remedy is ``renormalize()``, three instructions that restore the
  invariant exactly, placed where the chain ends rather than where each operation does: at the end
  of the inner loop, every so many iterations of a long accumulation, or where the value is handed
  to code that assumes normalization - a conversion out of ``low`` into ``mid`` or ``high`` does
  this on its own. ``fp32mp2_stat_low`` says whether ``low`` is safe on your data, and where:
  ``overlap_count`` and ``min_hi_lo_gap`` show how often and how far the limbs drift, and
  ``invert_count`` has to stay zero. The pattern is the ``low`` type for the kernel and
  ``add<fpmp2_accuracy::high>`` for the one step that needs more, as in
  :ref:`Accuracy levels <libcudacxx-extended-api-fp-fpmp-accuracy>`.

- **Use** ``high`` **only on accuracy-critical steps - reductions above all.** ``high`` recovers the
  rounding error committed when the two trailing limbs are added, and that error only grows into the
  result when something amplifies it; cancellation is the usual amplifier. It costs 20 instructions
  per addition against 11, and 46 per division against 13, while multiplication is 9 at both levels
  - a product-heavy kernel gains nothing from ``high`` and pays for it on every addition. Where it
  pays is where nearly equal values are subtracted, where a long sum carries terms of both signs,
  and in argument reductions. ``fp32mp2_stat`` points at those places: ``partial_cancel_count`` and
  ``full_cancel_count`` count the operations where significance was lost, per instrumented variable.
  Then raise the level per operation, with ``add<>``, ``sub<>`` and the rest on the critical step,
  rather than raising the type for the whole kernel.

- **The conversion macros trade FP64 instructions for registers.**
  ``CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP`` and ``CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE``, both on by
  default, implement the ``double`` ↔ ``fp32mp2`` conversions in integer and FP32 arithmetic so that
  no FP64 instruction sits on the hot path. The cast-based conversions they replace are two and
  three FP64 operations: shorter, with fewer live registers, but on the rationed FP64 pipe. Setting
  either to ``0`` is worth trying in two cases - a large kernel whose register count or spills, as
  ``-Xptxas -v`` reports them, are limiting occupancy, and a part where FP64 is cheap anyway, where
  the integer path is extra instructions for nothing. Measure before keeping it: on a 1:64 part a
  few FP64 operations per element can cost more than the registers buy back. The macros change the
  bodies of inline functions, so set them for the whole build rather than for one file, and they
  affect ``fp32mp2`` only; an ``fp64mp2`` conversion is FP64 either way.

- ``CCCL_FPMP_EXPLICIT_CASTS=0`` **is for the first compile of a migration, not for the build you
  ship.** The default makes every narrowing conversion into the pair a visible cast; switching it
  off lets a large ``double`` codebase compile against ``fp32mp2`` without touching the call sites.
  The conversions still take the accurate two-limb path, so what is saved is edit churn rather than
  correctness - but the diagnostics it silences are exactly the list the second point says to work
  through: every place a ``double`` enters a pair expression and a conversion runs. With the macro
  off, a ``double`` intermediate in a hot loop, or a helper that still takes ``double``, costs a
  conversion in each direction and FP64 arithmetic in between, with nothing to say so. The genuinely
  lossy conversions - ``int64_t`` and ``fp64mp2`` into ``fp32mp2`` - pass silently as well. Use it
  to get a large port compiling, then turn it back on and work through the errors; each one is a
  conversion site worth knowing about.

- **Avoid the atomics, even though they are provided.** ``atomicAdd`` and ``atomicSub`` on
  ``fp32mp2`` are a 64-bit ``atomicCAS`` loop: each attempt reads the pair, performs a ``high``
  addition and tries to exchange, and under contention a failed exchange means the whole addition is
  redone. Hardware ``atomicAdd`` on ``float`` or ``double`` is one instruction resolved in L2. The
  gap is small while every thread hits a different address and large when many hit the same one,
  which is the reduction pattern. Reduce first: the ``__shfl_down_sync`` family is overloaded for
  the pair, so a warp reduces in five shuffle steps and a block finishes in shared memory, leaving
  one atomic per block - or none, with per-block partials and a second pass, which also makes the
  result deterministic. The type is trivially copyable with the usual operators, so generic block
  and warp reductions accept it as they do any user-defined type.

- **Smaller things that add up:**

  - **Prefer the one-limb accumulate.** ``acc += f`` with a ``float`` operand takes a dedicated
    path, 10 instructions against 11 at ``mid`` and 13 against 20 at ``high``, and builds no
    temporary pair.
  - **Prefer** ``mad`` **to a multiply and an add, and use** ``fma`` **for its rounding, not by
    habit.** ``mad(a, b, c)`` is a ``low`` multiply followed by an addition at the type's level, 16
    instructions at ``mid`` against 20 for the two operations written out. ``fma`` is 19 at ``mid``
    and 37 at ``high``, more than the separate operations at ``high``, so it belongs where the fused
    rounding of the main term matters.
  - **Hoist divisions.** Division is the most expensive basic operation and the least accurate. A
    loop-invariant divisor is a reciprocal computed once and multiplied, at the price of one extra
    rounding, and where a square root feeds a division, ``rsqrt`` and a multiply replace the pair.
  - **Mind the register count.** Each pair is two registers, and the ``high`` algorithms carry more
    temporaries. ``-Xptxas -v`` or Nsight Compute shows spills; ``__launch_bounds__`` and the
    conversion macros above are the levers.
  - **The fast-math flags do not reach the arithmetic.** The error-free transformations are written
    with the ``_rn`` intrinsics - ``__fadd_rn``, ``__fmul_rn``, ``__fmaf_ieee_rn`` and their
    ``double`` counterparts - and the seeds with explicit approximation instructions, so ``--fmad``
    contraction and the ``--use_fast_math`` division and square-root replacements do not alter them;
    flush-to-zero touches only the subnormal range, which ``low`` and ``mid`` do not support in any
    case.
  - **Template the kernel on the type.** The level is part of the type, so a kernel templated on it
    compares ``low``, ``mid``, ``high`` and ``double`` from the build line with no change to the
    code, which is how the figures above were taken.

.. _libcudacxx-extended-api-fp-optimization-guide-fp64mp2:

fp64mp2
~~~~~~~

- **Use it only on GPUs with full-rate FP64.** ``fp64mp2`` spends FP64 throughput - 11 FP64
  instructions per ``mid`` addition, 9 per multiplication - so the price of its digits is set by how
  the part rations that pipe. Where FP64 runs at half the FP32 rate, as on B200, H100 and A100, the
  price is fair: on B200 a ``mid`` addition sustains 1347 giga-additions per second and is 7.8×
  faster than software binary128, and division is 17.8× faster. On B300, with FP64 at 1/64 of FP32,
  the same addition sustains 45 - a thirtieth - and software binary128, which is integer code and
  does not compete for the FP64 units, is 1.4–4× faster than ``fp64mp2`` operation for operation. On
  such a part the choice is usually ``fp128`` for the digits or ``double`` for the speed, with
  ``fp64mp2`` the worst of both - usually, because those figures come from small kernels. Software
  binary128 is long integer code with many values live at once, and a large kernel already near its
  register budget can spill where the pair, at four registers per value and a dozen instructions per
  operation, does not; once the spills sit in the inner loop the ordering can reverse, and
  ``fp64mp2`` comes out ahead on rationed FP64 after all. The same code size shows up at build time,
  since every ``fp128`` operation instantiated adds noticeably to compile time. So measure in the
  real kernel, with ``-Xptxas -v`` on both builds, rather than from the table. The host side has no
  such rationing, so CPU code is on the good side of this line.

- **Conversions are free; bandwidth and registers are not.** Unlike the float pair, ``fp64mp2``
  converts to and from ``double`` at no cost worth measuring: ``double`` into the pair is a limb
  copy with a zero tail, and the pair into ``double`` is at most one FP64 addition, so the
  conversion macros above play no part here. What the wider type costs is 16 bytes per value against
  8, and four registers against two. Keep data that is ``double`` precision to begin with in
  ``double`` arrays, and promote where the extra width is needed - typically the accumulator, since
  the inputs are exact as ``double`` and it is the running sum, or sum of products, that outgrows
  it. ``acc += d`` with a ``double`` operand is the cheap path for exactly that, 10 FP64
  instructions at ``mid`` against 11 for a pair addition after promotion and 13 against 20 at
  ``high``. Convert the result back once, when it leaves the kernel. Where whole arrays must be
  ``fp64mp2``, the type is 16-byte aligned, so each element is one 128-bit load or store: keep them
  contiguous and aligned, and budget half the elements per byte of bandwidth. The exception to
  "free" is the quad interchange: ``fp64mp2`` ↔ ``__float128`` is software in both directions, about
  1200 instructions on the device, and belongs at the I/O boundary.

- **The accuracy levels follow the float pair, with one extra check.** The costs are the same - 8,
  11 and 20 FP64 instructions per addition - and so is the advice: ``mid`` by default, ``low`` on
  the dominant path with ``renormalize()`` where the chain ends, ``high`` per operation where
  cancellation is possible, and ``fp64mp2_stat`` to find both. The extra check is ``zero_lo_count``.
  The point of the type is digits beyond ``double``; if that counter is a large share of
  ``ops_count``, the second limb is idle for most of the data, the computation is not using what it
  pays for, and ``double`` would give the same answer faster.

- **Math functions are where the time goes.** The arithmetic, ``sqrt``, ``rsqrt``, ``fma`` and
  ``mad`` have dedicated double-double implementations; the transcendentals currently not. As
  :ref:`Math function accuracy <libcudacxx-extended-api-fp-fpmp-math-accuracy>` describes, they
  evaluate in a binary128 backend where one is active and in ``double`` where not, and the two cases
  call for opposite handling.

- **Division costs more than on the float pair.** A ``mid`` division is 18 FP64 and 25 integer
  instructions, ``high`` 28 and 49, with the integer part doing the branch-free scaling that keeps
  reciprocals in range. On B200 a ``mid`` division has 80% of the throughput of a ``mid`` addition
  and a ``high`` division 44%, so the hoisting advice above applies with more force: reciprocals
  once, ``rsqrt`` and a multiply rather than one over ``sqrt``.

- **The atomics need sm_90 and are slower still.** ``atomicAdd`` on ``fp64mp2`` is a 128-bit
  compare-exchange loop, which exists on Hopper and later only - elsewhere the call fails to link
  with an error naming the requirement - while hardware ``double`` ``atomicAdd`` has been one
  instruction since sm_60, so the gap to the native type is wider than for the float pair. The
  remedy is the same: shuffle and shared-memory reduction first, one atomic per block at most, or
  per-block partials and a second pass for a deterministic result.

- **It is the type to port 80-bit** ``long double`` **code to.** CPU code written for the x87
  double-extended format - ``long double`` on Linux/x86 - meets a GPU that has no such format in
  device code. ``fp64mp2`` carries 104 significand bits against that format's 64, so the precision
  survives the port with room to spare, and on a full-rate FP64 part it is the cheapest way to carry
  it: a software binary128 would pay for 113 bits where 64 were asked for. What does not survive is
  exponent range, ``double``'s ±10\ :sup:`308` against x87's ±10\ :sup:`4932`, so code that chose
  the format for its range rather than its mantissa needs rescaling or binary128 - see the note on
  retiring 80-bit code under :ref:`Where this pays off <libcudacxx-extended-api-fp-fpmp-80-bit>`.

- **Know when the alternative is better.** Two cases call for something other than ``fp64mp2`` even
  on a full-rate part. A computation that needs more than 104 significand bits, or a wider exponent
  range than ``double``'s, needs binary128 and nothing else will do. And a computation whose only
  ill-conditioned part is a sum or a dot product may get its digits from compensated ``double``
  arithmetic - Kahan, pairwise summation, the Ogita–Rump–Oishi dot product in the references - which
  is built from the same error-free transformations as the pair. Where only the accumulation needs
  the extra width, ``fp64mp2`` for the accumulator alone, as above, is that algorithm written for
  you, at 10 FP64 instructions per term.

.. _libcudacxx-extended-api-fp-optimization-guide-fpemu:

Using fpemu efficiently
-----------------------

``fpemu`` is ``double`` built from 32-bit integer and ``float`` operations. Its costs are the pack
and unpack at the boundaries and the integer work inside, and almost everything below is about
keeping the first out of the loop and matching the second to what the computation needs. The
measured figures live on :ref:`the fpemu page <libcudacxx-extended-api-fp-fpemu>`; the advice here
is qualitative.

- **The unpacked form can beat native** ``double`` **on both speed and accuracy; the packed form
  rarely does in real applications.** ``fp64emu_unpacked`` pays the pack and unpack once per chain
  rather than once per operation and keeps 9 guard bits between operations, so on a rationed part a
  chain of it runs ahead of hardware ``double`` and lands closer to the exact answer. That is the
  form that competes with native ``double`` in real applications, and sometimes wins. Packed
  ``fp64emu`` is a different proposition. The π integration on :ref:`the fpemu page
  <libcudacxx-extended-api-fp-fpemu>` puts it close to parity with hardware ``double`` on RTX 6000
  Ada, RTX PRO 6000 Blackwell and B300, with the unpacked form at about 1.5× it, but that kernel is
  the exception rather than the rule: a few operations per term, no memory traffic, nothing else
  competing for registers. A real kernel is larger and rarely bound by arithmetic alone, and each
  emulated operation replaces one FP64 instruction with a run of integer and FP32 instructions and
  several more live registers, so in a large kernel the packed form usually means register pressure,
  spills and a net loss against the FP64 pipe, slow as that pipe is. Its uses are the ones the
  unpacked form cannot serve: ``double`` work moved onto the FP32 and integer pipes alongside a
  saturated FP64 pipe, see :ref:`Mixed cases <libcudacxx-extended-api-fp-optimization-guide-mixed>`;
  results that must match ``double`` bit for bit while the FP64 pipe is avoided; and a target with
  no FP64 arithmetic at all. On a full-rate part hardware ``double`` is the fast path for
  everything, and emulating it only slows the mix down.

- **Know what the default buys.** ``fpemu`` defaults to ``high``, the opposite of ``fpmp``, so
  packed ``fp64emu`` reproduces ``double`` bit for bit out of the box - and at best runs at about
  ``double``'s speed. That is the right type when the requirement is exact ``double`` semantics off
  the FP64 pipe, and the wrong one when the requirement is speed: for speed, step up to the unpacked
  form, step down a level, or both.

- **Chains go unpacked.** Everything the unpacked form gains, in speed and in accuracy, comes from
  paying the pack and unpack once per chain instead of once per operation, and from the 9 guard bits
  that survive between operations. Enter it once, leave it once, and keep the intermediate values in
  it. The explicit constructor is the reminder: every ``fp64emu_unpacked{x}`` in a loop body is a
  repacking that was probably avoidable.

- **Budget 16 bytes per unpacked value.** Four registers rather than two, and twice the bytes if it
  is ever stored. Store packed, compute unpacked, and watch ``-Xptxas -v`` in kernels with many live
  values; the unpacked form is the one that costs occupancy.

- **Step down with the range in mind.** ``mid`` is 1–2 ulp and ``low`` about ``float``'s precision;
  ``mid`` runs at 1.4–1.6× ``double`` packed and 1.8–2.0× unpacked, ``low`` at 1.7–2.0× and
  2.0–2.3×. In the packed form both handle the normal range only and drop the subnormal, infinity
  and NaN handling that ``high`` carries; the unpacked form keeps full-range pack and unpack at
  every level, so stepping down there is the safer move. ``low``'s error is biased rather than
  scattered - a long accumulation in ``low`` tracks plain ``float`` - so its place is a
  range-limited loop body with a wider accumulator elsewhere, not the accumulator itself.

- **Use the fused operations.** ``fma``, ``mad``, ``dot`` and ``cmul`` exist because composing them
  from ``*`` and ``+`` would round and repack between the steps. Where an expression has one of
  those shapes, call the fused form.

- **Directed rounding is a packed-only feature.** The ``rz``, ``ru`` and ``rd`` spellings of the
  intrinsics exist for the packed representation; the unpacked form rounds to nearest and nothing
  else. Interval arithmetic and its relatives therefore stay packed.

- **Decide what "reproduces double" means before choosing.** Only packed ``high`` returns
  ``double``'s values. A chain in the unpacked form behaves like x87 extended precision stored at
  the end: more accurate, and different. Code validated bit for bit against a ``double`` reference
  needs the packed form, or a reference computed the same way.

.. _libcudacxx-extended-api-fp-optimization-guide-fptool:

Using fptool
------------

The two instruments answer the two questions the previous sections keep raising: **how much
precision does this computation need**, and **what is the pair arithmetic actually doing**. Both are
opt-in behind ``CCCL_ENABLE_FPTOOL``, defined for the whole build; both cost rather than save; and
both are used the same way - change a type alias, read the answer, change it back. The macro is
there because both tools keep their state in global memory and touch it from inside the arithmetic:
``fp64_custom`` reads its run-time field sizes there, and ``fpmp2_stat`` writes its counter record
there through atomics on every operation. Global objects reached that way can cause problems in a
complex parallel application, so the gate makes sure they leave the target application together with
the research work that brought them in rather than staying behind.
:ref:`The fp64_custom page <libcudacxx-extended-api-fp-fptool-custom>` and
:ref:`the fpmp2_stat page <libcudacxx-extended-api-fp-fptool-stat>` describe them; this section says
how they fit into an optimization pass.

.. _libcudacxx-extended-api-fp-optimization-guide-fptool-custom:

fp64_custom: how much precision is enough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fp64_custom<E, M>`` is a ``double`` that rounds to ``M`` mantissa bits and ``E`` exponent bits
before and after every operation, so an algorithm can be run as though the hardware had that format.
It is the tool for the second question in
:ref:`Choosing the type <libcudacxx-extended-api-fp-optimization-guide-choosing>`.

- **Sweep the mantissa with the range held native.** ``fp64_custom<11, M>`` reduces the significand
  and clamps nothing, so a sweep over ``M`` shows where the computation itself gives way. The
  run-time sizes, ``fp_custom_dynamic_size`` with the setters, make the sweep one binary and one
  loop; the compile-time sizes make the native format cost nothing.

- **Keep the reference reduction wider than the format under test.** Reduce the per-thread partials
  in ``fp64mp2`` or the like, so the sweep measures the loop rather than the summation. That gives
  the best result each format can produce, which is what a sizing question wants.

- **Then sweep the exponent separately.** ``fp64_custom<E, 52>`` keeps the precision and narrows the
  range, which separates an algorithm that needs bits from one that needs headroom - the distinction
  between ``fp32mp2``, which has ``float``'s range, and ``fp64emu_low``, which has ``double``'s.

- **Know what is not reduced.** Construction stores the value unreduced, comparisons compare
  unreduced bits, and every math function except ``sqrt`` and ``fma`` resolves through ``operator
  double()`` and computes at full FP64 precision, silently. A sensitivity study that leans on
  ``exp`` or ``fabs`` is measuring less than it thinks.

- **It is cheap to run.** Every reduced format costs about 10% over plain ``double`` on the π
  integration, so a full sweep is an afternoon's measurement rather than a project. The run-time
  sizes pay a memory read per reduction, and under whole-program compilation each translation unit
  has its own copy of them, so a size set in one is not seen by the others; relocatable device code
  shares them.

.. _libcudacxx-extended-api-fp-optimization-guide-fptool-stat:

fpmp2_stat: what the arithmetic did
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fp32mp2_stat`` and ``fp64mp2_stat`` wrap the ``fpmp2`` types, compute bit-identical results, and
record on the device how many operations of each kind ran, which numerical events occurred and how
the limbs were placed. Several points in
:ref:`Using fpmp efficiently <libcudacxx-extended-api-fp-optimization-guide-fpmp>` lean on them;
this is the order to read the record in.

1. **Validate** ``ops_count`` **against a closed form.** Most kernels have one. A count that
   disagrees means the region is not the one intended, something was optimized away, or part of the
   arithmetic ran on the host, where nothing is collected.
2. **Read** ``zero_lo_count`` **as a share of** ``ops_count``. A large share means the second limb
   was idle for most of the data and the pair type is not earning its cost - ``fp32mp2`` might be
   only ``float``'s job (``fp64mp2`` ``double``'s).
3. **Compare** ``_low``, ``_mid`` **and** ``_high`` **records on the same workload.**
   ``overlap_count``, ``invert_count`` and ``min_hi_lo_gap`` on the ``_low`` record say whether the
   fast level is safe on this data and how far the limbs drift, which is where to put
   ``renormalize()``. The ``invert_count`` must be zero at every level - important.
4. **Read the cancellation counters as locations, not damage.** ``partial_cancel_count`` and
   ``full_cancel_count`` mark the operations where significance was lost, which is where ``high``
   pays and where a reformulation might. A compensated sum cancels on every term while being at its
   most accurate, so the count is a map of where to look rather than an error count.
5. **Check the exponent range.** ``min_exp``, ``max_exp`` and ``denorm_count`` say whether the data
   approaches the edges: ``fp32mp2`` inherits ``float``'s range, and the ``low`` and ``mid`` levels
   support the normal range only, so any subnormal means a configuration is being used outside its
   domain.

Instrument one variable and leave the rest plain - the property spreads along an expression from
either operand until a plain variable absorbs it, so the counters describe that variable's
arithmetic rather than the kernel's. Run on a reduced problem with a small grid, one block of 32
threads being enough, since every operation updates one device-wide record through atomics and an
instrumented run is two to three orders of magnitude slower than the plain type. Bracket the region
tightly with ``fpmp2_stat_reset_device_data`` and ``fpmp2_stat_read_device_data``, measure one type
at a time because all eight aliases share the record, and keep the alias behind a build flag so the
instrumentation cannot ship.

.. _libcudacxx-extended-api-fp-optimization-guide-measuring:

Measuring
---------

Every recommendation above ends in the same place: the figures in these documents are one benchmark
on a handful of parts, and the computation that matters is yours. The component types are built to
make that measurement cheap.

- **Template the kernel on the type.** The level, the width and the component are all part of the
  type, so a kernel templated on it compares ``float``, ``double``, every ``fpmp`` and ``fpemu``
  variant and the ``fptool`` instruments from the build line with no change to the code. One alias
  is the whole configuration, and the π integration behind the figures in these pages is written
  that way.

- **Measure end to end, with the accumulator where it lives.** Both the ``fpmp`` and the ``fpemu``
  measurements show the reduction deciding most of the accuracy column. A measurement that reduces
  in a wider type than the one under test reports the loop's accuracy; one that reduces in the same
  type reports the program's. Neither is wrong, but only one matches the program you are going to
  ship.

- **Report digits and time together.** A faster type that loses the digits the computation needed
  has not won, and a slower one that gains digits the computation cannot use has not either. The
  tables in the component pages pair the two columns for that reason; a measurement of your own
  should do the same.

- **Check the pipe you meant to use.** For an ``fp32mp2`` or ``fpemu`` kernel on a rationed part,
  the loop body should carry no FP64 instructions - the ``fp32mp2`` conversions keep a few on a cold
  fallback path, and that is all:

  .. code-block:: bash

     cuobjdump --dump-sass kernel.o | grep -cE 'DADD|DMUL|DFMA|DSETP|F2D|D2F|F2F\.F64|F2F\.F32\.F64'

  A count beyond those few points at a hidden conversion - a ``double`` variable in a pair
  expression, a ``double`` literal that did not fold, a helper that still takes ``double``, or
  ``CCCL_FPMP_EXPLICIT_CASTS=0`` letting one through - and the same count on an ``fp64mp2`` kernel
  says how much of the FP64 pipe each element costs.

- **Check registers and spills.** ``-Xptxas -v`` reports both per kernel. ``fp64mp2`` and
  ``fp64emu_unpacked`` double the register footprint of the value they replace, and the ``high``
  algorithms carry more temporaries. A kernel that spills is usually a kernel to restructure before
  it is a kernel to speed up, and it is also the one case where software binary128 and ``fp64mp2``
  can trade places on a rationed part.

- **Compare against the type you are replacing, on the same part.** The question is never whether
  ``fp32mp2`` is fast but whether it is faster than ``double`` on the GPU the program runs on, and
  the ratios in these documents move by an order of magnitude between a 1:2 and a 1:64 part. Measure
  the baseline in the same run.
