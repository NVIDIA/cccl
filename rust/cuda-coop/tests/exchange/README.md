# Public Exchange fixture

This standalone test uses the public `cuda_coop` API and the pinned Oxide
toolchain in `Cargo.toml`. It needs no generic-const-expression feature.

The positive mode runs 14 cases: both Exchange directions for Block33/i8/N1,
Block4×3×2/i16/N4, Block64/f64/N9, physical Warp32/f32/N4,
logical Warp8/i32/N8 and Warp1/u16/N9, plus both directions with selected W8
siblings and different group-uniform iteration counts. Every launch uses two
blocks. The host independently computes each permutation.

Every lane checks all output values and validity, extent and out-of-bounds
access, input preservation, inverse roundtrips, and a mutated fresh result
retained across later calls. A full-block Direct Store records one observation
per lane after Warp branches reconverge. No private validity bytes or raw
output pointers are used.

Build and run from this directory with the matching installed toolchain:

```sh
cargo oxide build --arch sm_120 --materialize-cubin -- --release --locked
cargo oxide run --arch sm_120 --materialize-cubin
```

The executable also accepts `invalid-input`. Run that mode in a separate
process: physical lane 31 leaves slot 3 unwritten before Exchange. An actual
CUDA launch failure is expected; successful return or an unrelated host error
does not qualify the trap. Positive sanitizer runs are separate from this
intentional failure.
