.. _libcudacxx-extended-api-fp-fptool:

fptool: Instruments for Floating-Point Arithmetic
=================================================

.. toctree::
   :hidden:
   :maxdepth: 1

   fp64_custom <fptool_custom>
   fpmp2_stat <fptool_stat>

The other two sub-components give you arithmetic to compute with. ``fptool`` gives you two
instruments for finding out what a computation *needs* and what it is *doing* — questions that are
otherwise answered by guesswork, or by rewriting the algorithm in a narrower type and hoping the
rewrite was faithful.

.. list-table::
   :widths: 30 34 36
   :header-rows: 1

   * - **Tool**
     - **Answers**
     - **How**

   * - :ref:`fp64_custom\<E, M\> <libcudacxx-extended-api-fp-fptool-custom>`
     - How little precision does this algorithm need? Is it the mantissa or the dynamic range that
       matters?
     - Rounds to an arbitrary exponent and mantissa width after every operation, so the
       computation runs as though the hardware had that format

   * - :ref:`fp32mp2_stat, fp64mp2_stat <libcudacxx-extended-api-fp-fptool-stat>`
     - What did the arithmetic actually do? Is the second limb carrying anything? Is anything
       silently corrupt?
     - Wraps an ``fpmp2`` type, computes bit-identical results, and records the operations and
       numerical events on the device

The difference between them is worth stating plainly, because it decides how each is used:
**``fp64_custom`` changes the arithmetic, and the ``fpmp2_stat`` types only observe it.** One is
there to make results worse in a controlled way, so you can see how much worse the algorithm
tolerates. The other is there to leave results alone — bit-identical to the type it wraps, by
construction — so that an instrumented run can be compared against a plain one, and a difference
between them means a race or an uninitialized value rather than a rounding change.

Neither is a type to ship in
----------------------------

Both are diagnostics, and both cost rather than save:

- ``fp64_custom`` emulates a narrow format *on native FP64*, adding a rounding step to each
  operand and to each result. Modelling BF16 with it is therefore slower than ``double``, not
  faster — in the integration :ref:`measured here <libcudacxx-extended-api-fp-fptool-custom-pi>`
  every reduced format costs about 10% over plain ``double``. What it buys is the answer to
  "would this work in BF16", on hardware that has no BF16 arithmetic, for a format that no
  hardware need ever implement.
- an instrumented ``fpmp2_stat`` run is **two to three orders of magnitude slower** than the plain
  type, every operation updating one device-wide record through atomics.

So the shape of a study is: reach for one of these to answer a question, read the answer, and put
it back. Both are designed for that — ``fp64_custom`` is a drop-in for ``double`` and the
``fpmp2_stat`` types are drop-ins for the ``fpmp2`` types they wrap, so in each case the change
is to a type alias and the algorithm is left alone. Keeping that alias behind a build flag is the
arrangement worth copying — and the header itself is behind one, ``CCCL_ENABLE_FPTOOL``, so that
a leftover include cannot quietly carry the tools into a shipping build.

Using the header
----------------

The feature is opt-in, for the reason above. Both tools keep mutable state at namespace scope —
``fp64_custom``'s runtime field sizes and the ``fpmp2_stat`` counter record — and because those
are variable templates, one copy is shared by every translation unit that includes the header. That
sharing is what makes the tools work, and it is also why an ``#include`` left behind after a study
is not free: the state, and for ``fpmp2_stat`` the atomic traffic on every operation, go into the
binary with it. So ``<cuda/fptool>`` refuses to compile until a project asks for it:

.. code-block:: bash

    nvcc -DCCCL_ENABLE_FPTOOL ...

.. important::
   Define it for the whole project rather than per file. The state above is shared across
   translation units, so a build where only some of them have opted in is the one configuration to
   avoid. This is the same rule the other :ref:`CCCL configuration macros <cccl-config>` follow.

.. code-block:: cuda

    #include <cuda/fptool>

One header carries both tools and their math functions, as ``<cuda/fpmp>`` does. The one
consequence to know is that including it costs about a fifth more than the types alone, since the
statistics math wrappers rest on the fpmp math surface.

Both tools work in host and device code from the same source, but with an asymmetry that matters
in opposite directions:

- ``fp64_custom`` keeps **independent host and device sizes**, deliberately, so a full-precision
  host reference can run alongside a reduced device computation.
- ``fpmp2_stat`` collection is **device-only**. The same source compiles and runs on the host, where
  the wrapper is a transparent pass-through and nothing is gathered — so a program doing part of
  its arithmetic on the host reports a count below what its closed form predicts, with no
  diagnostic.

Everything either tool names is specific to the component and carries the namespace. Unlike
``fpmp`` and ``fpemu``, there is little here to find by argument-dependent lookup: the setters,
getters and readout functions take only an ``int`` or a stream, so there is no operand of a
component type to look in.

.. seealso::
   :ref:`fp64_custom <libcudacxx-extended-api-fp-fptool-custom>` — **emulating a narrower format
   on native FP64**, for sensitivity studies.

   :ref:`fpmp2_stat <libcudacxx-extended-api-fp-fptool-stat>` — **recording what the arithmetic
   did**, without changing what it computes.
