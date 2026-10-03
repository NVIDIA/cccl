# Fix: RANGE low-bin regression (SM100 _4/_8 tuning leak)

## Symptom (data: run_2026-06-16{,_lowbin})
Single-channel RANGE regresses vs upstream `main` at the STATIC <=256-bin SMEM-privatized tier:
- range F64: bins 16/32/64 = 0.943 / 0.939 / 0.975 (geomean of default/main). bin 256 already a win (1.089).
- range I32: bin 32 = 0.973 (others ~1.0).
EVEN never regresses (1.14-3.86x). Dynamic-SMEM tier (bins 1024-49152) WINS 2.8-4.4x and must not move.

## Root cause
The SM100 single-channel non-byte tuning (tuning_histogram.cuh, `policy_selector_from_types::operator()`,
the `sample_size == 4 || sample_size == 8` arm) sets RANGE `threads = 768`. That ONE policy drives BOTH:
- the STATIC <=256-bin kernel (`PRIVATIZED_SMEM_BINS == max_privatized_smem_bins == 256`), and
- the DYNAMIC-SMEM kernel (`PRIVATIZED_SMEM_BINS == kDynamicSmemKernelTagBins == 16384`, bins 1024..cap).
768 threads is right for the dynamic mid-bin tiers (occupancy-bound) but too wide for the tiny static tier,
where RANGE's per-sample SearchTransform binary search is latency-bound and extra resident warps only add
issue/occupancy pressure. Launch tags confirm bins<=256 run `smem_privatized:static`, 1024-49152 run
`smem_privatized:dynamic`, >=57344 run `direct_single_probe`.

## Why the runtime-override approach (like direct_atomic_threads) is UNSOUND here
`direct_atomic_threads_per_block` works because the direct-atomic kernel is a pure grid-stride loop -- correct
at ANY launch block size. The SMEM-priv SWEEP kernel is NOT: `AgentHistogram` builds
`BlockLoad<SampleT, BLOCK_THREADS=threads_per_block, ITEMS_PER_THREAD>` (agent_histogram.cuh:264-268), a
tile-based load that requires the LAUNCH thread count to EQUAL the compile-time `BLOCK_THREADS`. Launching
384 threads of a kernel compiled for 768 => wrong/incomplete loads => incorrect histogram. So the static
thread count must be a COMPILE-TIME choice baked into the kernel's policy, AND the host launch must use the
SAME number.

## Sound fix (compile-time, kernel + host agree)
1. `histogram_policy`: add trailing defaulted `int static_smem_threads_per_block = 0` (0 = inherit
   threads_per_block) + accessor `static_smem_threads()`, mirroring `direct_atomic_threads`. Update ==/<<.
2. RANGE `_4`/`_8` policy brace: append the override value (start at 384 = main's width; tune empirically).
   EVEN leaves it 0 (its static tier never regressed).
3. `DeviceHistogramSmemPrivatizedKernel` (kernel_histogram.cuh:1537+) and its host-init twin (1707+):
   compute `kSweepThreads = (PrivatizedSmemBins == max_privatized_smem_bins && hp.static_smem_threads_per_block != 0)
   ? hp.static_smem_threads_per_block : hp.threads_per_block;` (all compile-time). Use kSweepThreads in BOTH
   the `AgentHistogramPolicy<...>` BLOCK_THREADS slot AND `__launch_bounds__`.
4. Host launch (dispatch_histogram.cuh, the privatized sweep around line 670): when
   `PRIVATIZED_SMEM_BINS == max_privatized_smem_bins`, set the sweep's `threads_per_block` to
   `active_policy.static_smem_threads()` (compile-time `if constexpr`) so the grid block dim, occupancy query,
   and `pixels_per_tile` all match the kernel's compile-time BLOCK_THREADS. The dynamic kernel
   (kDynamicSmemKernelTagBins) and GMEM path (0) are untouched.

## Invariant to preserve
Launch block dim == kernel compile-time BLOCK_THREADS (BlockLoad correctness). Verify the kernel-side
`kSweepThreads` and the host-side resolved threads use IDENTICAL logic.

## Validation
- catch2: cub.test.device.histogram lid_0 (53587/38) + lid_2 (54338/32).
- Re-sweep range I32+F64 bins 16/32/64/256 vs main -> regression closed (>= ~1.0), and confirm EVEN low-bin
  + range dynamic tier (1024-49152) UNCHANGED (byte-identical routing, no perf move).
- Try static override in {256, 384, 512}; pick the best low-bin range without hurting bin-256 (already a win).
