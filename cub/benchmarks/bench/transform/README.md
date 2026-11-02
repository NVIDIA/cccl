# `cub::DeviceTransform` benchmarks and tuning

`cub::DeviceTransform` has several backend algorithms (prefetch, vectorized, ublkcp, ...), each with its own set of
tuning parameters. When a single benchmark file exposes all of those parameters at once via `%RANGE%` comments, the tuning
infrastructure has to search a far larger hypercubic parameter space than necessary,
since some parameters are ignored for some algorithms (e.g. varying the prefetch stride for the vectorized algorithm has no effect).

We could fail fast for such nonsensical combinations using `#if`/`#error` guards,
and fix the nonsensical parameter at a default value, but it turns out that the evolutionary tuner doesn't like that.
See also https://github.com/NVIDIA/cccl/issues/7262 for more information.

The workaround applied here is to give each backend algorithm its own benchmark file with the right tuning parameters,
so every parameter combination the tuner tries is valid and compiles.
For benchmarks that follow this pattern, the layout is:

* `<benchmark>.h` — the actual benchmark code, shared by all variants below.
* `<benchmark>.cu` — plain benchmark using the default tuning policy.
   Use this one for regular benchmarking. It cannot be tuned.
* `<benchmark>.prefetch.cu`, `<benchmark>.vectorized.cu`, `<benchmark>.ublkcp.cu`:
   Each file hardcodes a backend algorithm and exposes only the `%RANGE%` parameters relevant to that
   algorithm. Use these for tuning a specific backend algorithm.
   To select the best backend algorithm, just compare the results of the different tuning benchmarks.

Not all `cub::DeviceTransform` benchmarks have been split this way, only the ones that seem relevant for tuning.
