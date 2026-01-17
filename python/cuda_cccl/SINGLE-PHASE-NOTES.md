---
# vim:set tw=78
title: "Single-Phase CUDA Cooperative Notes"
categories:
  - cuda.cooperative
author: "Trent Nelson"
date: 07/30/2025
#image: "images/pytorch-and-python-free-threading.png"
description: |
    Notes on the single-phase CUDA Cooperative development.
format:
  html:
    css: extend.css
    code-annotations: select
    code-line-numbers: true
    grid:
      gutter-width: 1rem
---

# CUDA Cooperative Today: Two-Phase

The `cuda.cooperative` module today supports a two-phase approach for using
accelerated CUB C++ primitives.  A primitive is created outside the kernel,
and then used within the kernel.  For example:

```python
import numpy as np
from numba import cuda
import cuda.cccl.cooperative as coop

# These variables need to be defined prior to the kernel, either as
# globals (in this case), or local function freevars (via closure).
dtype = np.int32
threads_per_block = 128
items_per_thread = 4

block_load = coop.block.load(
    dtype=dtype,
    threads_per_block=threads_per_block,
    items_per_thread=items_per_thread
)

block_store = coop.block.store(
    dtype=dtype,
    threads_per_block=threads_per_block,
    items_per_thread=items_per_thread
)

# The `.files` of every primitive (LTO IR blobs) need to be explicitly
# passed to to the `@cuda.jit` decorator.
@cuda.jit(link=block_load.files + block_store.files)
def elaborate_gpu_memcpy(d_in, d_out):
    # Would be nice if `dtype` could be inferred from the `d_in`
    # argument, but `cuda.local.array` requires a known literal value
    # at this point.
    thread_data = cuda.local.array(items_per_thread, dtype=dtype)
    block_load(d_in, thread_data)
    block_store(d_out, thread_data)
```

# CUDA Cooperative Tomorrow: Single-Phase

Single-phase, generally, refers to the ability to directly use the primitives
within the kernel without needing to create them outside the kernel first,
e.g.:


```python
from numba import cuda
import cuda.cccl.cooperative.experimental as coop

# No need to `link` anything anymore, this is all handled behind the scenes
# automatically.
@cuda.jit
def elaborate_gpu_memcpy(d_in, d_out, items_per_thread):
    # Note the use of our new `coop.local.array()` helper, which can
    # handle `items_per_thread` as a kernel parameter, as well as figure
    # out the `dtype` from the `d_in` argument on the fly.
    thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
    coop.block.load(d_in, thread_data)
    coop.block.store(d_out, thread_data)
```

::: {.callout-note title="References"}
The original issue capturing some discussion and motivation is here:
[Issue #4776](https://github.com/NVIDIA/cccl/issues/4776).  For the past few
months development has occurred in the branch represented by this
[WIP PR](https://github.com/NVIDIA/cccl/pull/4954).
:::

At the time of writing, the single-phase investigation has resulted in support
for the following primitives:

1. New `coop.(local|shared).array()` helpers that can handle non-literal
   arguments, e.g. `items_per_thread`, and infer the `dtype` from the first
   argument passed to the primitive.

2. `coop.block.load()` and `coop.block.store()` primitives.

3. `coop.block.scan()` primitive (albeit without support for block prefix
    callbacks or custom types yet).

4. `coop.block.histogram()` primitive, which is the first working example
    of a parent/child primitive in `cuda.cooperative` (i.e. it has persistent
    state).

5. `coop.block.run_length()`: initial scaffolding, but switched to histo
    for easier testing of parent/child work.

## Concepts & Terms

Reviewing existing terms:

- **Two-phase**: A CUB primitive that was created outside of the kernel.  The
  only time this will be necessary going forward is when a user needs to know
  temp storage and alignment requirements for one or more CUB primitives prior
  to kernel launch, such that a shared memory allocation within the kernel can
  be sized accordingly and subsequently shared between multiple primitives
  (with `cuda.syncthreads()` calls added as necessary).  Example:

```python
dtype = np.int32
dim = 128
items_per_thread = 16

# Primitive constructed outside a CUDA kernel.
block_load = coop.block.load(dtype, dim, items_per_thread)

@cuda.jit
def kernel(d_in, d_out, items_per_thread):
    thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
    # And called within via the returned "invocable".
    block_load(d_in, thread_data)
    ...

```
- **Invocable**: In the existing two-phase approach, this refers to the
  callable function object returned by the current primitives, which can then
  be called from within the kernel, i.e. the `block_load` above is an invocable
  instance that has the appropriate typing and lowering wired up to it in order
  for Numba/Numba-CUDA to handle it correctly.  These aren't exposed to users
  in the same way for either single-phase **or** two-phase going forward.

- **Single-phase**: A CUB primitive that can be used directly within the kernel
  without needing to create an invocable outside the kernel first, e.g.:

```python
@cuda.jit
def kernel(d_in, d_out, items_per_thread):
    thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
    coop.block.load(d_in, thread_data, items_per_thread)
```

New terms:

- **"One-shot" Primitive**
  - A CUB primitive without persistent state (i.e. most of them;
    load/store/scan/reduce, etc.).

- **Parent/Child Primitives**:
  - A CUB primitive struct that contains persistent state.  *Parent* refers to
    the constructor invocation.  *Child* refers to one or more methods called
    against that parent instance (i.e. the `this` pointer in C++ terms), e.g.
    histogram, run-length decode, flag discontinuity.  Example:

```python
@cuda.jit
def kernel(...):
    thread_data = coop.local.array(...)
    smem_histogram = coop.shared.array(...)

    # Load thread data...
    ...

    # Parent instance construction.
    histo = coop.block.histogram(thread_data, smem_histogram)

    # Child invocation corresponding to `BlockHistogram::InitHistogram()`
    # instance method.
    histo.init()

    # Child invocation corresponding to `BlockHistogram::Composite()`
    # instance method.
    histo.composite(thread_data)

```

## Concrete Example

The most complex application of the new single-phase support is demonstrated
by the following pytest which exercises the greatest number of moving parts
to date:

```python
from functools import reduce
from operator import mul

import numpy as np
import pytest
from numba import cuda

import cuda.cccl.cooperative.experimental as coop


def get_histogram_bins_for_type(np_type):
    dtype = np.dtype(np_type)
    bins = 1 << (dtype.itemsize * 8)
    return bins if dtype.kind == "u" else bins >> 1


@pytest.mark.parametrize("item_dtype", [np.uint8, np.int8])
@pytest.mark.parametrize("counter_dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("threads_per_block", [32, 128, (4, 16), (4, 8, 16)])
@pytest.mark.parametrize("items_per_thread", [2, 4, 8])
@pytest.mark.parametrize(
    "num_total_items",
    [
        1 << 10,  # 1KB
        1 << 12,  # 4KB
        1 << 15,  # 32KB
        1 << 19,  # 512KB
        1 << 23,  # 8MB
        1 << 28,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        coop.BlockHistogramAlgorithm.ATOMIC,
        coop.BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram_histo_single_phase_2(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    num_total_items,
    algorithm,
):
    # Example of what a two-phase histogram instance creation would look like,
    # purely for the purpose of demonstrating how much simpler the single-phase
    # construction is because we can infer so many of the other parameters
    # on the fly.
    histo_two_phase = coop.block.histogram(
        item_dtype=np.int8,
        counter_dtype=np.int32,
        dim=128,
        items_per_thread=4,
        algorithm=coop.BlockHistogramAlgorithm.ATOMIC,
        bins=256,
    )

    # N.B. The example above is still valid; the single-phase functionality
    #      does not remove the ability to construct these primitives outside
    #      of kernels--which you would do if you needed to access temp storage
    #      or alignment ahead of time.
    print(f"temp storage bytes: {histo_two_phase.temp_storage_bytes}")
    print(f"temp storage alignment: {histo_two_phase.temp_storage_alignment}")

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        # Create the "parent" histogram instance.  Note that we only need
        # to provide two parameters--the local register array from which we
        # plan to read, and the corresponding shared-memory bin count array.
        #
        # We can infer all of the other arguments required by the two-phase
        # constructor (see `histo_two_phase` above) automatically, as follows:
        #
        #   item_dtype: inferred from thread_samples.dtype (which was inferred
        #               from d_in.dtype).
        #
        #   counter_dtype: inferred from smem_histogram.dtype (which was inferred
        #                  from d_out.dtype).
        #
        #   dim: inferred from the grid launch dimensions of the current kernel.
        #
        #   items_per_thread: inferred from the thread_samples.shape (which was
        #                     obtained via the items_per_thread kernel param).
        #
        #   algorithm: defaults to ATOMIC.
        #
        #   bins: inferred from smem_histogram.shape (which was inferred from
        #         d_out.shape).
        histo = coop.block.histogram(thread_samples, smem_histogram)

        # Initialize the histogram.  This corresponds to the CUB instance
        # method BlockHistogram::InitHistogram.  It is a "child" call of a
        # parent instance.  (Our shared-memory array also comes pre-zeroed
        # so this technically isn't necessary.)
        histo.init()

        cuda.syncthreads()

        # Loop through tiles of the input data, loading chunks optimally via
        # block load, then updating the histogram accordingly.  To simplify
        # the example, we don't have any corner-case handling for undesirable
        # shapes (e.g. items per block (and thus, block offset) not a perfect
        # multiple of total items, etc.).
        while block_offset < num_total_items:
            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
            )

            # This is the second "child" call against the parent histogram
            # instance, corresponding to the the CUB C++ instance method:
            #
            #   BlockHistogram::Compute(
            #       T& thread_samples[ITEMS_PER_THREAD],
            #       CounterT& histogram[BINS]
            #   )
            #
            # Note that we don't need to furnish the smem_histogram as the
            # second parameter here--that was already provided to the
            # histogram's constructor, and thus, is accessible to us behind
            # the scenes when we need to wire up this composite call.
            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        # Final block atomic update to merge our block counts into the user's
        # output results array (d_out).
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

    # Kernel ends.  Remaining code is test scaffolding.

    bins = get_histogram_bins_for_type(item_dtype)

    threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = blocks_per_grid

    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)

    # Sanity check sum of histo bins matches total items.
    assert np.sum(actual) == num_total_items
    assert np.sum(expected) == num_total_items

    # Verify arrays match.
    np.testing.assert_array_equal(actual, expected)
```

1. You no longer need to pass anything to the `@cuda.jit` decorator
   (i.e. no `link=` argument).  The single-phase machinery handles all of the
   necessary linking and code generation behind the scenes.  (Technically,
   there's nothing to even provide to the decorator anyway at that point,
   as no primitives have appeared yet.)
2. The shared memory for the histogram `smem_histogram` can be sized directly
   off the `d_out` argument, which is a device array of the correct shape and
   dtype (i.e. the histogram bins).  (The `alignment=128` isn't strictly
   necessary, I just included it to verify alignment was getting plumbed all
   the way through correctly.)
3. The local array for the thread samples `thread_samples` can be sized
   directly off the `items_per_thread` argument, which is a kernel parameter
   that can be passed in at kernel launch time.  The `dtype` is inferred from
   `d_in.dtype`, as those are the *"items"* being counted.

## Implementation

In order to achieve the new single-phase functionality, we have to hook into
Numba's [Stage 5a: Rewrite Typed IR](https://numba.readthedocs.io/en/stable/developer/architecture.html#rewrite-typed-ir)
phase of the compilation pipeline.

::: {.callout-note collapse="true" title="Regarding order of implementation..."}

It can be helpful to be aware of the order I got things working when reviewing
all of the new content in the dev branch.  At a top-level, the work includes
a) at-one-point-definitely-working (but potentially broken right now)
implementations of block load and store that worked in both single-phase and
two-phase mode, b) new coop local/shared array primitives, c) block scan
in single-phase only with a lot of restrictions, d) a
non-functional-but-probably-close-to-functional `coop.block.run_length()` parent
primitive and supporting `decode()` child method, and e)
the following concrete things:

1. Block load/store primitives: `coop.block.load()` and `coop.block.store()`.
   I initially only got this working in "single-phase" land; two-phase support
   was absolutely not functional anymore via the same interface (you could
   get the same behavior by creating an invocable directly via a `create()`
   class method, but I didn't want that to be the final proposal).  Getting
   the single-phase rewriting of these primitives was a pretty big milestone,
   and paved the way for the general approach regarding creation of new
   synthentic Numba IR nodes and subsequent block reconstruction.  Huge amount
   of pain experienced at every level, particularly typing and lowering.

2. The new "coop" local/shared array helper functions that accepted non-literal
   arguments and rewrote them on the fly to `cuda.(local|shared).array()`
   equivalents once concrete shapes and dtypes were known (thanks to the
   rewriter being called after type-inference).  This also necessitated a
   change to `numba-cuda` to support access to information about kernel launch
   parameters and grid dimensions, etc.  See:
   [PR #280: Implement a thread-local means to access kernel launch config](https://github.com/NVIDIA/numba-cuda/pull/288).  This also required very hand-written rewriting
   implementation that didn't leverage anything from the block load/store
   rewriting code, other than the general concept.  The first point where
   block load/store and the coop local/shared arrays work was this
   excitedly-named commit:
   [WORKING CHECKPOINT COMMIT!](https://github.com/NVIDIA/cccl/pull/4954/commits/137d680aac52e7c49fa0089a99e0bed6a7211204).

3. Figuring out a way to have the two-phase and single-phase paradigms use
   the same namespace, i.e. block load is always accessible via
   `coop.block.load()` regardless of whether you're within a CUDA kernel or
   outside of it.  This was incredibly non-trivial to get working!  But it
   does now work, as long as the underlying primitive has been ported to the
   new `BasePrimitive`-derived class implementation (plus all of the other
   scaffolding now required to support single-phase/two-phase typing/lowering
   etc.).  Only block load/store support this dual interface at the moment
   (and I may have already broken it recently).

4. Figuring out how to get parent/child struct stuff working using the
   run-length decode primitive.  Failed to get it working, had to review
   the innards of the catch2 test to get a better appreciation for how it
   was meant to be called.  Saw that it called exclusive scan internally,
   so decided to switch gears and port `coop.block.scan()` to single-phase
   first.

5. Ported `coop.block.scan()` to single-phase.  Noticed some common patterns
   occurring in the typing, rewriting and lowering phases by this point, so
   was able to implement a couple of more abstract methods that should apply
   to more classes (versus authoring rewriting code by hand for every
   primitive).  Got the basic exclusive-sum functionality working, then
   was dragged back into hell trying to get the `BlockPrefixCallbackOp` support
   working in single-phase... which I failed to do.  I did manage to end up
   with a CUDA kernel that passes typing and lowering though, which is not
   nothing.  It crashes cryptically when the block prefix callback attempts
   to access the `self_ptr[0]` (which it needs to do in order to do anything
   useful, like keep track of a running total).

6. Switched gears again to histogram: at least that had a pretty simple
   parent/child interface that would be easier to debug.  Finally got that
   working end-to-end after a huge amount of hacking.  Found more opportunities
   for abstraction (yay) and the three histogram support classes (one for the
   constructor, init and composite) all leverage the exact same rewriting
   code that has been sufficiently abstracted/genericized to work with all of
   them.

The point of mentioning all of this is that the newer stuff is looking much more
refined than the older stuff.  The block load/store has a lot of warts and
battle scars... histogram is cleaner at all levels (typing/rewriting/lowering),
and I am now back on the *"We could definitely try to get Numbast to
auto-generate all this nonsense"*-train.

:::

### \_rewrite.py

The entirety of the rewriting logic currently lives in the
[\_rewrite.py](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_rewrite.py)
file.

#### CoopNodeRewriter

The main work-horse is the [CoopNodeRewriter](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_rewrite.py#L2799)
class, which derives from Numba's `numba.core.rewrites.Rewrite` class, and
registers itself automatically by way of the
`numba.core.rewrites.register_rewrite` decorator.  A pseudo-code walk-through
of the main logic is provided below, which should help when reviewing the
actual implementation.

```python
from numba.core.rewrites import Rewrite, register_rewrite
@register_rewrite("after-inference")
class CoopNodeRewriter(Rewrite):
    def match(self, func_ir, block, typemap, calltypes, **kw):
        # This routine is responsible for examining all of the individual
        # instructions present in a block and determining if they're of
        # interest to us.
        #
        # Specifically, is this instruction representing an operation
        # that we want to rewrite?  We ascertain this by checking for
        # all the different ways we can receive things that we need to
        # rewrite at this stage.
        #
        # From a performance perspective, the goal of this routine is
        # to determine if it *doesn't* need to rewrite anything as quickly
        # as possible so that control can be returned back to Numba.
        # If there's nothing of interest in the block, there should be
        # negligible overhead.  (So, there are various early-exit paths
        # in the implementation to facilitate this.)
        #
        # If we do find an instruction of interest, we need to categorize
        # it, i.e. is it a single-phase one-shot primitive?  Is it a child
        # method invocation against an as-yet unseen parent primitive?  Is
        # it an invocation attempt of a two-phase primitive (i.e. one that
        # was created outside of the kernel--we need special handling to
        # detect and rewrite those nodes now too).
        #
        # Once we can categorize the instruction, we capture a bucketload
        # of local variables, stash them in the appropriate `CoopNode`-
        # derived class for that instruction (e.g. `CoopBlockLoadNode`),
        # register the node against the instruction's target name, and
        # return True, which tells Numba that we want our `apply()` func
        # to be called immediately after we return, because we plan on
        # returning a new block of instructions.
        #
        # Otherwise, we return False, which tells Numba we didn't find
        # anything of interest and don't plan on rewriting anything, so
        # no need to call our apply() function after this.
        #
        # Pseudo-code follows.

        for instr in block.body:
            if not isinstance(instr, ir.Assign):
                # We don't care about things that aren't assign nodes.
                continue

            # Have we already seen a node for this target?
            target_name = instr.target.name
            node = self.nodes.get(target_name)
            if node:
                # Handle this quirky corner-case where we've already created
                # a node for an instruction before we've seen it.  This will
                # happen for parent nodes that were constructed in blocks
                # preceding child method invocations--i.e. the exact setup
                # as our histogram example above.  This is because, for
                # whatever reason, Numba calls all of the registered rewriters
                # with blocks in the reverse order they appear, so we end up
                # seeing `histo.composite(thread_samples)` before we even have
                # any idea what `histo` is.  Ditto for `histo.init()`.
                if node.is_parent:
                    # Some logic to determine if we need to still rewrite,
                    # then move onto the next instruction.
                    ...
                    continue

                # Otherwise, validate a whole bunch of invariants that, if
                # you look at the actual implementation, were an artifact of
                # handling lots of bizarre things I wasn't ever expecting
                # (like seeing children before parents, match() being called
                # multiple times for blocks we've already seemingly seen, etc.).
                ...
                # And then continue to the next instruction.
                continue

            if instr_involves_a_module_we_are_interested_in(instr):
                # Set up some stuff to track numba and numba.cuda modules that
                # appear as we may need to synthetically inject, for example,
                # the numba.cuda module into block if we're dealing with some
                # coop.(local|shared).array() nodes (that we need to rewrite
                # into `cuda.(local|shared).array()` equivalents).
                ...

            # At this point, we don't care about any instructions that aren't
            # `call` instructions, which are bundled as `ir.Expr` with an
            # op == "call".
            rhs = instr.value
            if not isinstance(rhs, ir.Expr):
                # Don't care about non-expressions.
                continue

            expr = rhs
            if expr.op != "call":
                # Don't care about expressions that aren't calls (i.e.
                # function invocations).
                continue

            # We're dealing with an instruction that's a function call at
            # this point.
            func_name = expr.func.name
            func = typemap[func_name]

            # Next up is a huge amount of complex logic to determine if this
            # is a function call we care about, and if so, what kind (one-shot,
            # single-phase vs two-phase, parent/child etc.).

            ...

            if not_a_function_we_care_about(instr):
                continue

            # If we get here, we've found a function call that we want to
            # rewrite.  The local variable `node_class` will be an instance
            # of the `CoopNode`-derived class that corresponds to that func,
            # e.g. `CoopBlockLoadNode` for `coop.block.load()`, etc.  We then
            # create a new instance of the node passing in keyword arguments
            # that capture everything under the sun, then call into the node
            # to have it "refine" the match, i.e.: do primitive-specific stuff
            # in support of a subsequent rewrite.  This includes things like
            # argument validation, argument inference (i.e. the process of
            # how we infer all of the histogram keyword arguments from just
            # `histo = coop.block.histogram(thread_samples, smem_histogram)`),
            # etc.
            node = node_class(
                every=variable,
                under=the,
                sun=gets,
                stashed=here,
            )
            self.nodes[target_name] = node
            node.refine_match(rewriter)
            we_want_apply = True
            continue

        # At the end of our enumeration of all instructions, we return the
        # `we_want_apply` boolean, which we will have set to True if one of
        # the many triggers prompting a rewrite were encountered.
        return we_want_apply

    def apply(self):
        # Construct an entirely new block of Numba IR instructions
        # according to whatever we determined we wanted to rewrite
        # during the match() phase above.
        #
        # In practice, what this translates to is simply creating a
        # new block and copying instructions over--unless it's an
        # instruction we're rewriting, in which case, we do a
        # switcharoo and inject our rewritten IR nodes.
        #
        # Simplified pseudo-code of the logic follows.
        new_block = ir.Block(
            self.current_block.scope,
            self.current_block.loc,
        )

        for instr in self.current_block.body:
            if not_an_instruction_we_care_about(instr):
                new_block.append(instr)
                continue

            # Get the "node" for our rewritten instruction (this is
            # our Swiss-Army knife class that basically captures all
            # the glue we need to rewrite stuff.
            target_name = instr.target.name
            node = self.nodes[instr.target.name]

            # Defer to the node itself for actually accomplishing the
            # rewrite, by way of the `CoopNode.rewrite()` class being
            # responsible for returning a sequence of new Numba IR
            # instructions to inject into the block.
            for new_instr in node.rewrite(self):
                new_block.append(new_instr)

        # Return the new "rewritten" block back to Numba.  (This kicks off
        # another rewrite match pass with all the registered rewriters, as
        # you may have injected new nodes that a rewriter now recognizes.)
        return new_block
```

#### Side-bar: The Numba IR

You can view the [Post-typed Numba IR](histo-numba-ir-pre-rewrite.html) to get an idea
of what the entire function IR looks like in our histogram example above.  The
linked color-coded IR dump is for the entire kernel, which has a handful of
blocks per the control flow (i.e. the early-exit bounds check, the while loop,
etc.).  Numba calls our rewriter's `match()` routine with blocks in the reverse
order they appear in the IR.  (A block is delineated by a `label <N>:` until
either a block terminator (i.e. `ret`) or another label.)


#### Node Classes

(Line numbers are relative to commit ref
[c7150e8e7](https://github.com/tpn/cccl/blob/c7150e8e777a710565c32c5fe3bbbe460d8d7f1e/python/cuda_cccl/cuda/cccl/cooperative/experimental/_rewrite.py).)

The base node class from which all primitives must inherit is
[CoopNode](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_rewrite.py#L751).  This is a Swiss-Army Knife
class, which is a fancy way of saying it's a `@dataclass` with a plethora of
attributes that probably don't have any typing annotations.

The histogram implementation needs three derived classes, one for the parent
constructor, and one for each child instance method (`init` and `composite`).

##### CoopBlockHistogramNode

The [CoopBlockHistogramNode](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_rewrite.py#L1891)
class represents the "parent" constructor, and is responsible for taking a
line of Python that looks like this:

```python
histo = coop.block.histogram(thread_samples, smem_histogram)
```

Which looks like this in Numba IR:

```{=html}
<code>
<span class="ansi38-15"> </span><span class="ansi38-15"> </span><span class="ansi38-15"> </span><span class="ansi38-15"> </span><span class="ansi38-81">histo</span><span class="ansi38-204"> = </span><span class="ansi38-81">call</span><span class="ansi38-15"> $504load_attr.40(</span>
<span class="ansi38-15">        </span><span class="ansi38-81">thread_samples</span><span class="ansi38-15">,</span>
<span class="ansi38-15">        </span><span class="ansi38-81">smem_histogram</span><span class="ansi38-15">,</span>
<span class="ansi38-15">        </span><span class="ansi38-148">func</span><span class="ansi38-15">=</span><span class="ansi38-15">$504load_attr.40<span class="ansi38-15">,</span>
<span class="ansi38-15">        </span><span class="ansi38-148">args</span><span class="ansi38-15">=[</span>
<span class="ansi38-15">            </span><span class="ansi38-148">Var</span><span class="ansi38-15">(</span><span class="ansi38-81">thread_samples</span><span class="ansi38-15">, </span><span class="ansi38-81">test_block_histogram</span><span class="ansi38-15">.</span><span class="ansi38-81">py</span><span class="ansi38-15">:</span><span class="ansi38-141">158</span><span class="ansi38-15">),</span>
<span class="ansi38-15">            </span><span class="ansi38-148">Var</span><span class="ansi38-15">(</span><span class="ansi38-81">smem_histogram</span><span class="ansi38-15">, </span><span class="ansi38-81">test_block_histogram</span><span class="ansi38-15">.</span><span class="ansi38-81">py</span><span class="ansi38-15">:</span><span class="ansi38-141">157</span><span class="ansi38-15">)</span>
<span class="ansi38-15">        ],</span>
<span class="ansi38-15">        </span><span class="ansi38-148">kws</span><span class="ansi38-15">=(),</span>
<span class="ansi38-15">        </span><span class="ansi38-148">vararg</span><span class="ansi38-15">=</span><span class="ansi38-81">None</span><span class="ansi38-15">,</span>
<span class="ansi38-15">        </span><span class="ansi38-148">varkwarg</span><span class="ansi38-15">=</span><span class="ansi38-81">None</span><span class="ansi38-15">,</span>
<span class="ansi38-15">        </span><span class="ansi38-148">target</span><span class="ansi38-15">=</span><span class="ansi38-81">None</span><span class="ansi38-15">)</span><span class="ansi38-15"> [</span>
<span class="ansi38-15">    '</span><span class="ansi38-15">$504load_attr.40<span class="ansi38-15">',</span>
<span class="ansi38-15">    '</span><span class="ansi38-81">histo<span class="ansi38-15">',</span>
<span class="ansi38-15">    '</span><span class="ansi38-81">smem_histogram<span class="ansi38-15">',</span>
<span class="ansi38-15">    '</span><span class="ansi38-81">thread_samples<span class="ansi38-15">'</span>
<span class="ansi38-15">]</span>
</code>
```

And extracting/inferring enough information to construct an instance of the
*"implementation"* (i.e. the actual underlying two-phase primitive that is
accessible outside of CUDA kernels via the `coop.block.histogram()` API)
that looks like this:

```python
    # self.impl_class = `<class 'coop.block.histogram'>`
    self.instance = self.impl_class(
        item_dtype=self.item_dtype,
        counter_dtype=self.counter_dtype,
        items_per_thread=self.items_per_thread,
        bins=self.bins,
        dim=self.threads_per_block,
        algorithm=self.algorithm,
        unique_id=self.unique_id,
        node=self,
        temp_storage=self.temp_storage,
    )
```

This is tackled in the `refine_match()` routine, reproduced below.  The
complexity of this task can be appreciated when you view the full Numba IR:

1. Find the `histo = call $504load_attr.40(thread_samples, smem_histogram)`
   line in the IR.

2. Wtf is the value of `thread_samples`?  Well, if we walk backwards
   through the IR we'll eventually find a `thread_samples = ...` line:

```
thread_samples = call $440load_attr.33(
    $460load_deref.35,
    $462load_deref.36
    ...
```

3. Wtf is the value for `$440load_attr.33`?  Well, if we walk backwards through
   the IR we'll eventually find a `$440load_attr.33 = ...` line:

```
$440load_attr.33 = getattr(value=$420load_attr.32, attr=array...
```

4. Ah, an array of some kind!  But wtf is the value for `$420load_attr.32`?
   Well, if we walk backwards through the IR we'll eventually find a
   `$420load_attr.32 = ...`:

```
    $420load_attr.32 = getattr(value=$410load_global.31, attr=local)
```

5. And just before that IR:

```
    $410load_global.31 = global(cuda: <module 'numba.cuda' from ...
```

6. Ah!  So `thread_samples = ` was constructed from a `cuda.local.array()`
   entry point!  But what was the first argument `$460load_deref.35`?
   Well, if we walk backward through the IR...

You get the idea.  For any given call instruction in Numba IR, figuring out
the genesis of all of the pieces participating in that call (the actual
underlying function and all of the arguments) requires traipsing back through
the IR following chains of calls and getattrs until we've found a definitive
root instruction.

Root instructions are going to be one of the following instruction types:

- `ir.Global` or `ir.FreeVar`: these map to Python globals and non-local (i.e.
  closure) variables, respectively.

- `ir.Const`: a constant that Numba injected into the IR to represent a known
  constant (i.e. the literal `4` for example).

- `ir.Arg`: the value derived from a parameter passed to the function.

Once we hit one of those instruction types, we know we've found the "root",
and no longer need to keep walking back through the chain.  (Nor could we
keep walking back any further if we wanted to---those instructions represent
the genesis of whatever it was for which we were originally looking.)

As soon as I tackled trying to support the parent/child stuff, it became
apparent we'd frequently need to perform two key operations:

1. For any given function call we're dealing with, reliably get the arguments
   it was passed, with proper support for all of Python's function argument
   nuances (args, varargs, kwargs, defaults, etc.).  This has already happened
   by the time a derived `CoopNode`-class accesses `self.bound.arguments`;
   but, that's a complex topic in and of itself that we'll discuss later.

2. Getting those arguments from the function call IR are just going to return
   useless `ir.Var(variable_name)` instructions that tell us nothing about
   anything of interest.  So we need to do the elaborate back-tracking
   described above to build up a picture of how exactly that variable was
   actually constructed.  The chain of instructions could be arbitrarily-deep;
   consider long getattr chains: `coop.block.foo.bar.moo.cow.my_cool_array()`.

The latter functionality is handled by a bunch of utility code I wrote that,
given an arbitrary Numba IR instruction, constructs a `RootDefinition` object
that represents everything that participated in the construction of whatever
variable it was fed.

In the `CoopBlockHistogramNode` class's `refine_match()` routine, it needs
to identify whatever was passed as the first argument `items`, which will
return `ir.Var(thread_samples, ...)` in our case, and then get the root for
that variable, which looks like this:

```python
    def refine_match(self, rewriter):
        bound = self.bound.arguments

        # Infer `items_per_thread` and `bins` from the shapes of the items and
        # histogram arrays, respectively.
        items_var = bound["items"]
        items_root = rewriter.get_root_def(items_var)
        items_leaf = items_root.leaf_constructor_call
```

A root node's leaf constructor call is best thought of as 99% likely to be the
thing you actually need at any given point.  In the case of `thread_samples`,
because that was created via `cuda.local.array()`, and our
`get_root_definition()` implementation knows how to handle both `cuda` and
`coop` arrays, it creates an instance of an `ArrayCallDefinition` dataclass
at that point in the chain, and fills it out with all of the stuff we'd care
about for arrays:

```python
@dataclass
class ArrayCallDefinition(CallDefinition):
    """
    Specialization of CallDefinition for array calls.  This is used to capture
    additional information specific to array calls.
    """

    # N.B. We need to mark everything `Optional` and default to None because
    #      our parent `CallDefinition` class had optional fields at the end
    #      (`assign`, and `order`).
    array_type: Optional[types.Array] = None
    array_dtype: Optional[types.DType] = None
    array_alignment: Optional[int] = None
    is_coop_array: bool = False
    shape: Optional[int] = None
```

So we can simply obtain the `items_per_thread` we need to provide to our
`coop.block.histogram()` invocation later by simply using the size/shape
that was used to create the `thread_samples` array, e.g. `4` if we had
done: `thread_samples = coop.local.array(4, dtype=np.int8)`:

```python
    items_per_thread = items_leaf.shape
```

The full `CoopBlockHistogramNode` class follows.  All of its logic is in
support of inferring all of the values needed to create an instance of the
actual underlying `coop.block.histogram()` primitive on the user's behalf.

```python
@dataclass
class CoopBlockHistogramNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram"
    disposition = Disposition.PARENT

    def refine_match(self, rewriter):
        bound = self.bound.arguments

        # Infer `items_per_thread` and `bins` from the shapes of the items and
        # histogram arrays, respectively.
        items_var = bound["items"]
        items_root = rewriter.get_root_def(items_var)
        items_leaf = items_root.leaf_constructor_call
        if not isinstance(items_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected items to be an array call, got {items_leaf!r}"
            )
        items_per_thread = items_leaf.shape
        if not isinstance(items_per_thread, int):
            raise RuntimeError("Could not determine shape of items array.")

        histogram_var = bound["histogram"]
        histogram_root = rewriter.get_root_def(histogram_var)
        histogram_leaf = histogram_root.leaf_constructor_call
        if not isinstance(histogram_leaf, ArrayCallDefinition):
            raise RuntimeError(
                f"Expected histogram to be an array call, got {histogram_leaf!r}"
            )
        bins = histogram_leaf.shape
        if not isinstance(bins, int):
            raise RuntimeError("Could not determine shape of histogram array.")

        self.algorithm = bound.get("algorithm")
        self.temp_storage = bound.get("temp_storage")

        items_ty = self.typemap[items_var.name]
        histogram_ty = self.typemap[histogram_var.name]

        self.items = items_var
        self.items_ty = items_ty
        self.items_root = items_root
        self.item_dtype = items_ty.dtype
        self.items_per_thread = items_per_thread

        self.histogram = histogram_var
        self.histogram_ty = histogram_ty
        self.histogram_root = histogram_root
        self.histogram_dtype = histogram_ty.dtype

        self.counter_dtype = histogram_ty.dtype
        self.bins = bins

        launch_config = rewriter.launch_config
        if launch_config is None:
            return False

        self.threads_per_block = launch_config.blockdim

        # Instantiate an instance now so our children can access it.
        self.instance = self.impl_class(
            item_dtype=self.item_dtype,
            counter_dtype=self.counter_dtype,
            items_per_thread=self.items_per_thread,
            bins=self.bins,
            dim=self.threads_per_block,
            algorithm=self.algorithm,
            unique_id=self.unique_id,
            node=self,
            temp_storage=self.temp_storage,
        )
        self.children = []

        algo = self.instance.specialization
        assert len(algo.parameters) == 1, algo.parameters
        # self.set_no_runtime_args()
        self.runtime_args = tuple()
        self.runtime_arg_types = tuple()
        self.runtime_arg_names = tuple()
        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()

```

##### CoopBlockHistogramInit

```python
@dataclass
class CoopBlockHistogramInitNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.init"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance

        histogram = parent_node.histogram
        histogram_ty = parent_node.histogram_ty
        if histogram_ty != self.typemap[histogram.name]:
            raise RuntimeError(
                f"Expected histogram type {parent_node.histogram_ty!r}, "
                f"got {histogram_ty!r} for {self!r}"
            )

        self.instance = parent_instance.init(self)

        self.runtime_args = [histogram]
        self.runtime_arg_types = [histogram_ty]
        self.runtime_arg_names = ["histogram"]

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
```

##### CoopBlockHistogramComposite

```python
@dataclass
class CoopBlockHistogramCompositeNode(CoopNode, CoopNodeMixin):
    primitive_name = "coop.block.histogram.composite"
    disposition = Disposition.CHILD

    def refine_match(self, rewriter):
        parent_node = self.parent_node
        parent_instance = parent_node.instance
        parent_root_def = parent_node.root_def
        assert self.parent_root_def is parent_root_def, (
            self.parent_root_def,
            parent_root_def,
        )

        bound = self.bound.arguments
        items = bound["items"]
        items_ty = self.typemap[items.name]
        if items_ty != parent_node.items_ty:
            raise RuntimeError(
                f"Expected items type {parent_node.items_ty!r}, "
                f"got {items_ty!r} for {self!r}"
            )

        histogram = parent_node.histogram
        histogram_ty = parent_node.histogram_ty
        if histogram_ty != self.typemap[histogram.name]:
            raise RuntimeError(
                f"Expected histogram type {parent_node.histogram_ty!r}, "
                f"got {histogram_ty!r} for {self!r}"
            )

        self.instance = parent_instance.composite(self, items)

        self.runtime_args = [items, histogram]
        self.runtime_arg_types = [items_ty, histogram_ty]
        self.runtime_arg_names = ["items", "histogram"]

        return

    def rewrite(self, rewriter):
        rd = self.rewrite_details
        return (rd.g_assign, rd.new_assign)

    @cached_property
    def rewrite_details(self):
        return self.do_rewrite()
```

#### Node Rewriting

For the initial block load/store and coop array work, the rewriting logic
was hand-written each time with small derivations to support nuances involved
with each pattern (coop array rewriting requires possibly injecting
`numba.cuda` module instructions, as we synthesize the full
`numba.cuda.local.array()` getattr chain in our rewritten block).

Thankfully by the time I got to authoring the rewriting logic for block
histogram---both parent and child---I saw enough commonality to warrant
a reasonably generic `CoopNode.do_rewrite()` implementation, which is now used
by all of the most-recently-written primitives (basically, histogram and scan).

[CoopNode.do_rewrite()](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_rewrite.py#L1114-L1295):

```python

@dataclass
class CoopNode:

    ...

    def do_rewrite(self):
        if self.is_two_phase:
            assert not self.instance
            instance = self.two_phase_instance
            algo = instance.specialization
            algo.unique_id = self.unique_id
        elif self.is_one_shot:
            # One-shot instances should not have an instance yet.
            if self.instance is not None:
                raise RuntimeError(
                    f"One-shot instance {self!r} already has an instance: "
                    f"{self.instance!r}"
                )
            instance = self.instance = self.impl_class(**self.impl_kwds)
        elif self.is_parent:
            # Parent instances should have an instance.
            instance = self.instance
            if instance is None:
                raise RuntimeError(
                    f"Parent instance {self!r} already has an instance: "
                    f"{self.instance!r}"
                )
        else:
            # Child instances should have an instance (created from the
            # appropriate parent instance method, e.g. `run_length.decode()`).
            assert self.is_child, self
            instance = self.instance
            if instance is None:
                raise RuntimeError(
                    f"Child instance {self!r} does not have an instance set, "
                    "but should have been created from a parent instance."
                )

        expr = self.expr
        assign = self.instr
        assert isinstance(assign, ir.Assign)
        scope = assign.target.scope

        g_var_name = f"${self.call_var_name}"
        g_var = ir.Var(scope, g_var_name, expr.loc)

        existing = self.typemap.get(g_var_name, None)
        if existing:
            raise RuntimeError(f"Variable {g_var.name} already exists in typemap.")

        # Create a dummy invocable we can use for lowering.
        code = dedent(f"""
            def {self.call_var_name}(*args):
                return
        """)
        exec(code, globals())
        invocable = globals()[self.call_var_name]
        mod = sys.modules[invocable.__module__]
        setattr(mod, self.call_var_name, invocable)

        self.invocable = instance.invocable = invocable
        invocable.node = self

        runtime_args = self.runtime_args or tuple()
        runtime_arg_types = self.runtime_arg_types or tuple()

        g_assign = ir.Assign(
            value=ir.Global(g_var_name, invocable, expr.loc),
            target=g_var,
            loc=expr.loc,
        )

        new_call = ir.Expr.call(
            func=g_var,
            args=runtime_args,
            kws=(),
            loc=expr.loc,
        )

        new_assign = ir.Assign(
            value=new_call,
            target=assign.target,
            loc=assign.loc,
        )

        return_type = self.return_type
        if return_type is None:
            return_type = types.void
        existing_type = self.typemap[assign.target.name]
        if existing_type != return_type:
            # I don't fully understand or appreciate why some primitives need
            # this but others don't, i.e. load/store will have a void return
            # type in the typemap... but coop.block.scan() calls will have a
            # `coop.block.scan` return type.  Regardless, if the existing type
            # differs, we need to clear it and set the new return type; if we
            # don't, we'll hit a numba casting error.
            del self.typemap[assign.target.name]
            self.typemap[assign.target.name] = return_type

        algo = instance.specialization
        parameters = algo.parameters
        num_params = len(parameters)
        assert num_params == 1, parameters
        parameters = parameters[0]
        param_dtypes = [p.dtype() for p in parameters]

        # if len(parameters) != len(self.runtime_arg_types):
        #    import debugpy; debugpy.breakpoint()
        #    raise RuntimeError(
        #        f"Expected {len(self.runtime_arg_types)} parameters, "
        #        f"but got {len(parameters)} for {self!r}."
        #    )

        sig = Signature(
            return_type,
            args=runtime_arg_types,
            recvr=None,
            pysig=None,
        )

        self.calltypes[new_call] = sig

        self.codegens = algo.create_codegens()

        outer_node = self

        @register_global(invocable)
        class ImplDecl(AbstractTemplate):
            key = invocable

            def generic(self, outer_args, outer_kws):
                @lower(invocable, types.VarArg(types.Any))
                def codegen(context, builder, sig, args):
                    node = invocable.node
                    cg = node.codegen
                    (_, codegen_method) = cg.intrinsic_impl()
                    res = codegen_method(context, builder, sig, args)

                    if not node.is_child:
                        # Add all the LTO-IRs to the current code library.
                        algo = node.instance.specialization
                        add_ltoirs(context, algo.lto_irs)

                    return res

                return sig

        # If we don't do the following, nothing works!  Numba will crash later
        # with a cryptic key error because some rando' cache won't have an
        # entry for the function we've just registered... unless we do a dummy
        # `func_ty.get_call_type()` call now to prime the cache.  This seems
        # like a red flag that we're doing something wrong overall... but, eh,
        # it all currently works, and many different (simpler) things were
        # tried that didn't work before reaching this final implementation.
        func_ty = types.Function(ImplDecl)

        typingctx = self.typingctx
        result = func_ty.get_call_type(
            typingctx,
            args=runtime_arg_types,
            kws={},
        )
        assert result is not None, result
        check = func_ty._impl_keys[sig.args]
        assert check is not None, check

        self.typemap[g_var.name] = func_ty

        rewrite_details = SimpleNamespace(
            g_var=g_var,
            g_assign=g_assign,
            new_call=new_call,
            new_assign=new_assign,
            sig=sig,
            func_ty=func_ty,
        )

        return rewrite_details
```

[The final rewritten Numba IR](histo-post-rewrite-dump.html).


<br/>
*Remaining content is actively WIP.*
<br/>

### Parent/Child Shim Code & Lowering

The auto-generated C++ shim code exported via `extern "C"` linkage for our
histogram example follows.  This is the exact C++ code passed to `nvrtc`
during lowering/codegen in order to produce an `LTOIR` blob that we can then
dynamically add to the linker as necessary.

```c++
#include <cuda/std/cstdint>
#include <cub/block/block_histogram.cuh>

using histo_0_t = cub::BlockHistogram<
    ::cuda::std::uint8_t, // T
    128, // BLOCK_DIM_X
    2, // ITEMS_PER_THREAD
    256, // BINS
    ::cub::BLOCK_HISTO_ATOMIC, // ALGORITHM
    1, // BLOCK_DIM_Y
    1 // BLOCK_DIM_Z
>;
__device__ constexpr unsigned int histo_0_t_struct_size = sizeof(histo_0_t);
__device__ constexpr unsigned int histo_0_t_struct_alignment = alignof(histo_0_t);

using histo_0_temp_storage_t = typename histo_0_t::TempStorage;
__device__ constexpr unsigned int histo_0_temp_storage_t_bytes = sizeof(histo_0_temp_storage_t);
__device__ constexpr unsigned int histo_0_temp_storage_t_alignment = alignof(histo_0_temp_storage_t);

// Constructor
extern "C" __device__ histo_0_t* histo_0_construct(
    ::cuda::std::int8_t* __restrict__ addr,
    ::cuda::std::uint32_t size
    )
{
    if (size < sizeof(histo_0_t)) {
        return nullptr;
    }
    if (reinterpret_cast<uintptr_t>(addr) % alignof(histo_0_t) != 0) {
         return nullptr;
    }
    return new (addr) histo_0_t();
}

// Destructor
extern "C" __device__ void histo_0_destruct(
    histo_0_t* __restrict__ algorithm
    )
{
    if (algorithm != nullptr) {
        algorithm->~histo_0_t();
    }
}

// Composite
extern "C" __device__ void histo_0_composite(
    histo_0_t* __restrict__ algorithm,
    ::cuda::std::uint8_t (&items)[2],
    ::cuda::std::uint32_t (&histogram)[256]
    )
{
    algorithm->Composite(
        items,
        histogram
    );
}


// InitHistogram
extern "C" __device__ void histo_0_init(
    histo_0_t* __restrict__ algorithm,
    ::cuda::std::uint32_t (&histogram)[256]
    )
{
    algorithm->InitHistogram(
        histogram
    );
}
```

Things that will probably be of interest mostly to Georgii at this point:

1. I am experimenting with unique IDs instead of the elaborate C++ name
   mangling, so instead of the really long symbol names we used to use,
   everything is now prefixed with `histo_0_` in this case (where `histo`
   is the literal name of the variable used in the originating Python
   code).  Additionally, parameter names are now taken from formal Python
   signatures, so it's very easy to eyeball which parameter corresponds
   to which C++ arg (as I've named them as close to the C++ counterparts
   as possible).  Super helpful when debugging the auto-generated C++ code.
   (This is also in support of an optimization goal that I suspect will
   be worthwhile: only doing one `nvrtc` compilation for *all* primitives
   a kernel needs; `nvrtc` compilation is by far the most expensive thing
   in the pipeline, and it appears to have a very high fixed cost unrelated
   to code complexity (nearly 1s even for trivial `code.cu` content).)

2. Our `Pointer()`-type now takes a `restrict: bool` argument, which, when
   True, results in a `__restrict__` annotation being included when the
   parameter's C++ representation is generated.  (And in true anti-YAGI form I
   haven't done any assessment regarding whether or not this makes any
   performance difference (probably not), just YOLO'd it in on a whim.)

3. I always inject four `__device__ constexpr unsigned int ...` lines for struct
   size and alignment, and temp storage size and alignment.  The struct size and
   alignment is new, and is required for the next point.

4. Parent/child implies the underlying CUB struct requires persistence (because
   it manages state).  This requires an allocation of a known size and
   alignment---which we can now obtain thanks to the new variables mentioned
   in the prior point.  The `histo = coop.block.histogram()` "constructor" call
   eventually results in a call to the constructor C++ shim that uses a
   *placement `new`* to in-place construct the C++ struct using the provided
   buffer, including a defensive size and alignment check (which isn't strictly
   necessary as we have full control over all of these parts, however, it's
   a critical invariant that we'd want to fail-fast on if we inadvertently
   break something down the track).

```c++
extern "C" __device__ histo_0_t* histo_0_construct(
    ::cuda::std::int8_t* __restrict__ addr,
    ::cuda::std::uint32_t size
    )
{
    if (size < sizeof(histo_0_t)) {
        return nullptr;
    }
    if (reinterpret_cast<uintptr_t>(addr) % alignof(histo_0_t) != 0) {
         return nullptr;
    }
    return new (addr) histo_0_t();
}
```

5. The actual lowering of the function calls happens in the same spot as it
   did before (this [code block](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_types.py#L1640-L1711),
   which I'll reproduce below), however, it now handles the three types of
   calls we encounter: one-shot (this is the new name for the existing
   behavior), parent and child:

```python

# ... in _types.Algorithm.create_codegen_method.intrinsic_impl.codegen():

    if primitive.is_one_shot:
        # This is the existing code, now referred to as "one-shot"
        # handling code (i.e. handling primitives that don't need
        # a persistent struct) and can just be raw-dogged each time
        # with a one-shot struct instance decl + instance method
        # call.
        function_type = ir.FunctionType(ir.VoidType(), types)
        function = cgutils.get_or_insert_function(
            builder.module, function_type, unique_name
        )
        builder.call(function, arguments)
        if ret is not None:
            return builder.load(ret)
    elif primitive.is_parent:
        assert ret is None
        # We need to stack-alloc sufficient space for the struct
        # and then call the constructor shim with the addr and
        # size parameters injected as the first two arguments.
        primitive = self.primitive
        node = primitive.node
        algo = node.instance.specialization
        names = algo.names
        struct_size = primitive.algorithm_struct_size
        struct_alignment = primitive.algorithm_struct_alignment
        buf_name = f"{names.target_name}_buf"
        buf = cgutils.alloca_once(
            builder,
            ir.IntType(8),
            size=struct_size,
            name=buf_name,
        )
        buf.align = struct_alignment
        addr_ty = ir.PointerType(ir.IntType(8))
        addr = builder.bitcast(buf, addr_ty)
        size_ty = ir.IntType(32)
        size = ir.Constant(size_ty, struct_size)
        # Inject our addr and size parameters as the first two
        # arguments to the intrinsic.  Ditto for the types.
        arguments[:0] = [addr, size]
        types[:0] = [addr_ty, size_ty]

        function_type = ir.FunctionType(addr_ty, types)
        function = cgutils.get_or_insert_function(
            builder.module, function_type, names.constructor_name
        )
        result = builder.call(function, arguments)
        node.cg_details = SimpleNamespace(
            buf=buf,
            addr=addr,
            size=size,
            arguments=arguments,
            types=types,
            result=result,
        )
        return result
    elif primitive.is_child:
        assert ret is None
        primitive = self.primitive
        node = primitive.node
        parent_node = node.parent_node
        parent_cg_details = parent_node.cg_details

        func_name = self.c_name

        parent_ptr = parent_cg_details.result
        arguments.insert(0, parent_ptr)
        ptr_ty = ir.PointerType(parent_ptr.type.pointee)
        types.insert(0, ptr_ty)

        function_type = ir.FunctionType(ir.VoidType(), types)
        function = cgutils.get_or_insert_function(
            builder.module, function_type, func_name
        )
        builder.call(function, arguments)
        return None
    else:
        raise RuntimeError("Invalid primitive state: {primitive!r}")
```

6. So for handling parent constructor calls, we're synthetically injecting two
   additional parameters, a `void*` address (backed by a stack-local `alloca`)
   and a corresponding struct size.  For children, we inject a single pointer
   as the first argument---the parent's address pointer---which the shim
   code now expects, and uses to call the appropriate member function, e.g.:

```c++
// Composite
extern "C" __device__ void histo_0_composite(
    histo_0_t* __restrict__ algorithm,
    ::cuda::std::uint8_t (&items)[2],
    ::cuda::std::uint32_t (&histogram)[256]
    )
{
    algorithm->Composite(
        items,
        histogram
    );
}
```

7. This synthetic parameter injection is now handled by the `Algorithm`'s
   `__init__` routine when called in the "specialized" form (i.e. with no
   template parameters) ([GitHub code link here](https://github.com/tpn/cccl/blob/4776-cuda-coop-single-phase/python/cuda_cccl/cuda/cccl/cooperative/experimental/_types.py#L658-L734)):

```python
class Algorithm:
    def __init__(
        self,
        struct_name,
        method_name,
        c_name,
        includes,
        template_parameters,
        parameters,
        primitive,
        type_definitions=None,
        fake_return=False,
        threads=None,
        unique_id=None,
        temp_storage=None,
    ):
        # Snip attribution initialization.
        ...

        if not template_parameters:
            # We're dealing with a specialized instance.
            if primitive.is_parent:
                # Inject the void *addr and size parameters for structs.
                injected = [
                    Pointer(
                        # numba.types.voidptr,
                        numba.types.int8,
                        name="addr",
                        restrict=True,
                        is_array_pointer=False,
                    ),
                    Value(numba.types.uint32, name="size"),
                ]
                if not self.parameters:
                    self.parameters = [injected]
                else:
                    # Insert the injected parameters at the beginning of each
                    # method's parameters.
                    for method in self.parameters:
                        method[:0] = injected
            elif primitive.is_child:
                # Inject the typed pointer to the parent instance type as
                # the first parameter.
                parent_node = primitive.parent.node
                parent_instance_type = parent_node.return_type
                parent_algo = parent_node.instance.specialization
                parent_names = parent_algo.names
                injected = [
                    Pointer(
                        parent_instance_type,
                        name="algorithm",
                        type_name=parent_names.algorithm_t,
                        restrict=True,
                        is_array_pointer=False,
                    )
                ]
                if not self.parameters:
                    self.parameters = [injected]
                else:
                    # Insert the injected parameter at the beginning of each
                    # method's parameters.
                    for method in self.parameters:
                        method[:0] = injected
```

8. You'd think we'd be able to rely more on Numba's `BoundFunction` scaffolding,
   which our `init()` and `composite()` typing uses... but, I don't think we
   can because of our elaborate on-the-fly rewriting, that results in a
   completely different instance being in flight than whatever Numba thought
   was going to be used as the `histo` parent instance, for example.

## Todo

The following section either lists things I haven't gotten around to tackling
yet, or things I have tried but are broken in the single-phase branch.

1. Get stateful Python callback functions working (i.e. `BlockPrefixCallbackOp`
   for `scan`).  They currently crash CUDA in a very cryptic way.  (Est: 1-3
   days?)

2. Haven't even tried user-defined types support yet, but I can guarantee it
   will need work to play nice with single-phase.  (Est: 2-4 days?)

3. Figure out a solution for argument validation and signature/typing that
   doesn't require so much duplication.  (Est: 2-4 days?)

4. Figure out the documentation story.  This includes both docstrings and
   higher-level guides and whatnot.  (Est: ~5-10+ days, factoring in figuring
   out the solution, and actually updating existing docs and writing new
   docs.)

5. Port all the existing warp and block primitives to the new
   `BasePrimitive`-derived interface.  This needs to be done for everything at
   the same time the new rewriting logic is introduced just due to the fact
   the way primitives are implemented fundamentally changes (i.e. from a Python
   function to a `BasePrimitive`-derived class).

6. ....and implement appropriate typing (`_decls.py`) and node (`_rewrite.py`)
   support for all of these primitives as well---again, a prerequisite for any
   of this stuff to get in.

7. Update all existing tests to ensure they still pass.  Write new tests that
   exercise single-phase variants.  (Est: probably at least 2-3 calendar weeks
   for the last three items.)

8. Spend a bit of time trying to break the rewriter.  I've already come across
   a weird bug where if child primitive is called in multiple blocks, things
   break (the rewriter needs to track multiple instances of the same child and
   essentially coalesce them such that codegen/LTO only handles the child func
   once).

   Additionally, doing quirky stuff like assigning primitives (within a kernel)
   to object attributes or setitem-array stuff will certainly break things now.

   (Est: 2-3 days?)

9. Ensure `temp_storage=` kwargs works again.  Definitely neglected it whilst
   getting single-phase stuff working.  (Est: ~2 days?)
