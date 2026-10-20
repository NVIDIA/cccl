# Native Oxide integration fixture

This private unsafe fixture qualifies the compiler/provider boundary. It is not
an end-user collective API. The API crate embeds constant registrations from its
Cargo build script; the compiler invokes that same built executable for one
native request batch. A separate generic helper crate is reached from five
concrete kernel roots. No consumer supplies an LTO artifact list.

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
participation and unsupported bounds-elision modes. Sources are restored after
each run. The initial checker diagnoses a shared helper site reached under two
different exact block shapes; separate sites remain supported.
