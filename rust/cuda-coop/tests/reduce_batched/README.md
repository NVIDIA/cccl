# Public batched Reduce fixture

This standalone test uses the public `cuda_coop` API and the pinned Oxide
toolchain in `Cargo.toml`. It enables nightly `generic_const_exprs` for the
operation-derived result extent; callers do not supply an output size.

The positive mode runs 14 cases and checks 1,792 per-lane reports. Six typed
shape rows exercise both Striped and Blocked results: physical W32/B1 and B4
i32 Sum, W8/B10 i8 Min and f64 Max, W1/B9 u16 Sum, and W8/B65 u32 BitXor.
Logical rows use XYZ blocks. Two additional i8 cases select alternating W8
siblings with differing group-uniform iteration counts.

Every result slot is checked against a host-computed numerical and validity
oracle, including absent slots, extent/OOB access, unchanged input, and fresh
results after mutating an earlier result. A full-block Direct Store records
one observation per lane after all Warp work reconverges. The test does not
inspect private bitmap bytes or read absent numeric values.

Build and run from this directory with the matching installed toolchain:

```sh
cargo oxide build --arch sm_120 --materialize-cubin -- --release --locked
cargo oxide run --arch sm_120 --materialize-cubin
```

The built executable also accepts `invalid-input`. Run that mode in a separate
process: it leaves B4 slot 3 unwritten in physical lane 31, which owns no output.
An actual CUDA launch failure is expected; successful return or an unrelated
host error does not qualify the trap. Run positive sanitizer checks separately
from this deliberate failure.
