# TWO-PHASE-CREATE.md

This note documents the *current* (as of Jan 2026) cuda.coop behavior around
single‑phase vs two‑phase usage and the meaning of the host‑side constructor
path. The goal is to
reduce confusion for users and for ourselves while we evolve the API.

---

## 1) Terminology

### Single‑phase (in‑kernel usage)
You call the primitive directly inside a CUDA kernel:

```python
@cuda.jit
def kernel(x, out):
    out[0] = coop.block.sum(x[cuda.threadIdx.x])
```

The compiler pass rewrites the call, instantiates the primitive on the fly,
and triggers NVRTC/LTO codegen as needed. This is flexible and supports
`temp_storage=` (for primitives that currently allow it).

### Two‑phase (host instance, one‑shot primitives)
You construct a primitive *outside* the kernel and call it inside the kernel.
Linking is handled automatically by the rewrite pass; you should **not**
pass `link=` to `@cuda.jit` for coop primitives.

```python
block_sum = coop.block.sum(np.int32, threads_per_block=128)

@cuda.jit
def kernel(x, out):
    out[0] = block_sum(x[cuda.threadIdx.x])
```

This avoids per‑primitive NVRTC at kernel compile time. The call signature is
fixed at creation time. **Today, this path does not accept `temp_storage=`.**

### Two‑phase instance (parent/child primitives)
Some primitives (e.g. block histogram / run‑length decode) are **parent/child**
structs with methods. For these, the “two‑phase” entry point is *not* an
invocable; it is a host‑side instance that you pass into the kernel:

```python
block_hist = coop.block.histogram(
    item_dtype=np.uint8,
    counter_dtype=np.uint32,
    dim=128,
    items_per_thread=4,
    bins=256,
)

@cuda.jit
def kernel(inp, out):
    histo = block_hist()          # parent instance in‑kernel
    histo.init()                  # child method
    histo.composite(...)          # child method
```

This is still “two‑phase” in the sense that the primitive is constructed
outside the kernel, but it is *not* the LTO‑invocable `.create()` path used
by one‑shot primitives.

---

## 2) What `.create()` means today

### One‑shot primitives (most block/warp primitives)
The public API no longer requires `.create()` for one‑shot primitives. Use the
host‑side constructor (e.g., `coop.block.reduce(...)`) and call the resulting
instance inside the kernel.

`.create()` still exists as a legacy/internal path, but it is **not** part of
the recommended public interface and should be avoided in new code.

### Parent/child primitives (histogram, run‑length decode)
For these, `.create(...)` is **just a constructor alias** and returns the same
parent object as calling the class directly.

Example:
```python
hist = coop.block.histogram.create(
    item_dtype=np.uint8, counter_dtype=np.uint32,
    dim=128, items_per_thread=4, bins=256,
)
# `hist` is *not* an Invocable; it is the parent instance.
```

In‑kernel, you call `hist()` to create the parent struct and then call
its child methods (`init`, `decode`, `composite`, etc.).

---

## 3) Accessing temp storage sizes without `.create()`

You do **not** need `.create()` to access temp‑storage sizes. Any host‑side
primitive instance exposes:

```python
prim.temp_storage_bytes
prim.temp_storage_alignment
```

This is often used with `gpu_dataclass` to compute aggregate temp storage:

```python
@dataclass
class Traits:
    load: Any
    scan: Any
    store: Any

traits = coop.gpu_dataclass(Traits(
    load=coop.block.load(np.int32, 128, 4),
    scan=coop.block.scan(np.int32, 128, 4),
    store=coop.block.store(np.int32, 128, 4),
))

bytes_ = traits.temp_storage_bytes_max
align = traits.temp_storage_alignment
```

**Important:** Reading these properties can trigger NVRTC to compute size /
alignment. It is cheap relative to a full kernel, but it is still compilation.

---

## 4) Explicit temp storage in kernels (current state)

The explicit `temp_storage=` kwarg is only supported in **single‑phase**
calls today, and only for a subset of primitives.

Working (single‑phase only):
- block.load / block.store
- block.scan / block.reduce / block.sum
- block.adjacent_difference
- block.discontinuity
- block.shuffle
- block.radix_rank

Not yet supported (explicit temp storage rejected or ignored):
- block.exchange
- block.merge_sort_keys
- block.radix_sort_keys / radix_sort_keys_descending
- block.run_length
- block.histogram (explicit temp storage raises NotImplementedError)
- all warp primitives

Example (single‑phase explicit temp storage):
```python
block_scan = coop.block.scan(np.int32, 128, 4)
bytes_ = block_scan.temp_storage_bytes
align = block_scan.temp_storage_alignment

@cuda.jit
def kernel(inp, out):
    temp = coop.TempStorage(bytes_, align)
    td = coop.ThreadData(4)
    coop.block.load(inp, td, temp_storage=temp)
    coop.block.scan(td, td, temp_storage=temp)
    coop.block.store(out, td, temp_storage=temp)
```

---

## 5) Why “explicit temp_storage in two‑phase invocables” is missing

The current `Algorithm` codegen picks **one** signature:
* If `primitive.temp_storage` is set, the signature includes the explicit
  temp storage parameter.
* Otherwise, it drops temp storage and uses implicit shared memory.

Two‑phase invocables (created via `.create`) have no kernel‑local temp storage
object to bind, so they always use the implicit signature. That’s why
`temp_storage=` is not currently available in the two‑phase invocable path.

---

## 6) Design goal (what we’re moving toward)

We want *all* primitives to accept `temp_storage=` inside kernels and to
allow kernels to explicitly carve shared memory for multiple primitives.
That will likely require:

1. Adding explicit temp storage parameters to **all** primitive signatures.
2. Generating **two overloads** for each primitive:
   - implicit temp storage
   - explicit temp storage
3. Extending warp primitives to use external temp storage buffers.
4. Providing an ergonomic “shared memory slice” helper so users can carve
   a single shared buffer into aligned slices.

This will make `gpu_dataclass` + explicit shared memory the idiomatic path
for kernels composed from multiple primitives.
