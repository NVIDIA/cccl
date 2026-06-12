#!/usr/bin/env bash
# Build 64-bit-counter / 64-bit-offset histogram bench variants for the large-input
# characterization sweep. Recompiles the even/range TUs with TUNE_CounterT=uint64_t and
# TUNE_OffsetT=int64_t (the 64-bit OffsetT is required for >2^31 elements) into *.u64
# binaries, relinking against the prebuilt nvbench objects -- same surgical recompile
# recipe as build_hitrate_variants.sh. DO NOT run while a benchmark is on the GPU.
#
# Caller must first write /tmp/hist_u64_cmds.json via:
#   (cd <build>; ninja -t compdb) | <extract even/range .cu compile commands>
set -euo pipefail
BUILD=$(cd "${HIST_BENCH_BUILD:-build/autocuda/cub-benchmark}" && pwd)
CMDS=/tmp/hist_u64_cmds.json
LABELS="even range"
# 64-bit counter MUST be `unsigned long long` (not uint64_t): CUDA atomicAdd /
# atomicAdd_block provide an `unsigned long long` overload but none for `unsigned long`
# (= uint64_t on this platform), so uint64_t fails to compile in the histogram kernels.
COUNTER="${TUNE_COUNTER:-unsigned long long}"
OFFSET="${TUNE_OFFSET:-long long}"
cd "$BUILD"
for L in $LABELS; do
  OUT="bin/cub.bench.histogram.${L}.base.u64"
  OBJ="/tmp/hist_${L}_u64.o"
  echo "  compile+link $L (CounterT=$COUNTER OffsetT=$OFFSET) -> $OUT"
  rm -f "$OBJ"
  L="$L" OBJ="$OBJ" OUT="$OUT" BUILD="$BUILD" CMDS="$CMDS" COUNTER="$COUNTER" OFFSET="$OFFSET" python3 - <<'PY'
import json, os, subprocess, shlex, sys
L=os.environ['L']; obj=os.environ['OBJ']; out=os.environ['OUT']; build=os.environ['BUILD']
counter=os.environ['COUNTER']; offset=os.environ['OFFSET']
m=json.load(open(os.environ['CMDS']))
toks=shlex.split(m[L]['cmd'])
i=toks.index('-o'); toks[i+1]=obj
nvcc_i=next(j for j,t in enumerate(toks) if t.endswith('nvcc'))
toks=toks[:nvcc_i+1] + [f'-DTUNE_CounterT={counter}', f'-DTUNE_OffsetT={offset}'] + toks[nvcc_i+1:]
r=subprocess.run(toks, cwd=build)
if r.returncode!=0: sys.exit(f'compile {L} u64 failed')
helper=f'{build}/nvbench_helper/CMakeFiles/cccl.nvbench_helper.dir/nvbench_helper/nvbench_helper.cu.o'
mainobj=f'{build}/_deps/nvbench-build/nvbench/CMakeFiles/nvbench.main.dir/main.cu.o'
libs=shlex.split(f'-Xlinker -rpath -Xlinker {build}/lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so '
  f'{build}/lib/libnvbench.so -lstdc++fs /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so '
  f'/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -ldl /usr/lib/x86_64-linux-gnu/librt.a '
  f'/usr/local/cuda/lib64/libcupti.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl')
link=['/usr/local/cuda/bin/nvcc','-ccbin=/usr/bin/c++','-arch=native',helper,mainobj,obj,'-o',out]+libs
r=subprocess.run(link, cwd=build)
if r.returncode!=0: sys.exit(f'link {L} u64 failed')
PY
done
echo "=== u64 variants: ==="; ls -la bin/*.u64 2>/dev/null | awk '{print $5,$9}'
