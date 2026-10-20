# Native Oxide integration fixture

This private unsafe fixture qualifies the compiler/provider boundary. It is not
an end-user collective API. The API crate embeds constant registrations from its
Cargo build script; the compiler invokes that same built executable for one
native request batch. A separate generic helper crate is reached from five
concrete kernel roots (seven with `--root-context`). No consumer supplies an LTO artifact list.

From the CCCL checkout, with the agreed Oxide source and backend already built:

```sh
python3 rust/cuda-coop-codegen/tests/run_oxide_fixture.py \
  --oxide-root /path/to/cuda-rust \
  --cargo-oxide /path/to/cargo-oxide \
  --backend /path/to/librustc_codegen_cuda.so \
  --cuda-root /path/to/cuda \
  --arch sm_120 \
  --output /path/to/new-evidence-directory
```

The runner copies the fixture into the new output directory and patches only
that copy to the supplied source checkout. It seeds dependency resolutions from
that checkout's `vecadd` lockfile. The selected nightly must match Oxide's pin.
The log and command receipt are retained beside the copied consumer.

The 480 cases cover i32 Direct/Transpose Load and Store, items 1/4, one/three
blocks, full/default-filled Load, full/partial Store, post-Store payload values,
and Store → explicit block synchronization → Load of the same region. Every
block uses uniform origins, count, default and mode; per-thread payload arrays
are initialized before use. There is no no-default partial-validity claim.

The fixture checks compiler-derived scratch sizes and exact block geometry,
rejects wrong shapes, thread counts and explicit byte counts, and runs with
`auto_shared`. Its helpers deliberately retain function boundaries to
exercise discovery; register/stack results are not an inlining baseline. The
materialized path uses `cargo oxide run --materialize-cubin` and the ordinary
embedded module loader. Convergent declarations require the coordinated O03a
compiler change.

Add `--checked` to exercise the internal checked full-block marker. This route
gets block geometry from each root contract, separates common controls from
lane-local payload pointers, selects either of two payload arrays per lane after
reconvergence, and repeats Load in a uniform loop with explicit synchronization.
The raw unsafe route remains the default. Both routes retain caller obligations
for bounds, initialized values, unique output ownership and scratch reuse.

After a checked run, exercise rejected source programs in the generated copy:

```sh
python3 rust/cuda-coop-codegen/tests/check_collective_rejections.py \
  /path/to/new-evidence-directory
```

The cases cover varying controls/carriers, different contexts of one helper,
projected references, atomic results, participation, and unsupported
bounds-elision modes. Positive contrasts retain reconvergent lane work in a
uniform loop and an exhaustive enum match. Sources are restored after
each run. Shared helper sites support different exact root shapes; the compiler
checks each root and argument context independently. Fixed external helper symbols
(`#[no_mangle]` or `#[export_name]`) are unsupported on checked paths, including
single-root paths. Remove those attributes to permit root-specific naming.

Add `--root-context` instead of `--checked` for 672 native cases. Two additional
64-thread roots call the same item-count 1/4 helpers and marker sites used by
128-thread roots. The outer generic helper uses `#[device]`; a nongeneric
dependency bridge also reaches it at both block sizes. Checked helpers defer
device emission until a kernel supplies its context. An intermediate
helper retains another call boundary, and
all copies read one device static. The fixture checks each root's exact shape
and scratch, including 272/1,040 bytes at 64 threads versus 528/2,064 at 128
threads. Two equal-shape roots also share the original helper Instance. This
route requires the coordinated contextual-helper compiler support.

The checked executable also accepts `--trap-before` and `--trap-after-load`.
Run each in a separate process: lane zero deliberately indexes past its payload,
before the first Load or between that Load and its scratch-reuse barrier.
Both runs must terminate with a CUDA execution error, without the success
message. These negative modes check the compiler's bounds-failure abort path;
the normal numerical run uses neither mode.

Add `--carrier` to run the separate private scratch-free carrier slice. Its safe
fixture wrappers use `DisjointSlice<i32, CollectiveTiles<4>>` and initialized
lane-local arrays. The compiler tracks the original kernel allocations through
direct helpers, joins and explicit `while` loops, and requires a full-block
barrier before conflicting reuse. The wrapper checks the supplied offset and
extent against a fixed per-block ownership tile; changing an operation's item
count never repartitions that allocation. The carrier exposes no safe lane
reference or conversion to an ordinary slice.

This bounded route covers only canonical full-block Direct Load/Store. Arbitrary
offsets retain their meaning and must fit the owning tile. The raw unsafe marker
remains available for other ownership mappings. Warp groups, reservations,
explicit scratch, public payload validity, descriptor mutation through references
and unresolved allocation identities are outside this fixture's supported slice.
The numerical checks include shifted offsets, zero/tail/full counts, item counts
1 and 4 in one kernel, Store/barrier/Load cycles and owned async completion.

```sh
python3 rust/cuda-coop-codegen/tests/check_carrier_rejections.py \
  /path/to/carrier-evidence-directory
```

This command checks missing and conditional barriers, loop backedges, divergent
controls, descriptor/reference mutation, ambiguous origins and mixed markers,
with positive cases for independent allocations and barriers in helpers. The
native carrier binary also accepts `overlap` or `bounds`; each invocation must
fail through a device trap, before an invalid native access.

Add `--validity` to exercise the five typed Direct recipes with the actual
`ThreadData` source, through a test-only facade. The private bridge passes
separate addresses for opaque cells and packed validity bits; it exposes no
initialized slice. This route uses the checked resource marker, canonical
full-block carrier and ordinary owned async completion. It requires compiler
support for the `MaybeUninit` intrinsics reached by safe payload access.

The cases cover all ten numeric types, items 1/4, exact typed defaults,
full and no-default masked Load, arbitrary initialized slots outside a partial
Store's write domain, preservation after Store, genuinely short allocations and
Store/barrier/Load. A full Load followed by masked Load checks that old
initialization is reset. A focused i32/items9 case crosses a bitmap-byte boundary
and preserves scattered initialized slots outside the Store prefix. The binary
accepts `hole`, `count`, `bounds`, `ownership` and `different-count`; each must
fail in a separate process. The facade is private test infrastructure, not a
published safe collective API.
