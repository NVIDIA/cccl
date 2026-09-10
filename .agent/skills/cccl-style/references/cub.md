# Extra CUB Style Guidance

## Arch/SM-Scoped Perf Changes

Applies to files in `cub/cub/device/dispatch/tuning/**`.
Any change that conditions a tuning constant, dispatch policy, or other perf-affecting
behavior like tuning encodings on compute capability or an SM arch macro.

- Never leave the comparison open-ended toward higher/future archs (e.g. `cc >= X` with no upper
  bound). An unbounded lower-only check silently applies to every arch above X, including ones
  that did not exist when the change was benchmarked.
- A closed range (`cc >= X && cc < Y`) is not automatically correct either: `Y` must sit exactly at
  the boundary of what was benchmarked, not at a rounder or more generous value picked out of
  convenience. A range wider than what was tested still leaks the change onto untested archs inside
  that range.
