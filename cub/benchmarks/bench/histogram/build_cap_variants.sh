#!/usr/bin/env bash
# Sweep B builder: recompile ONLY the 4 histogram benchmark TUs with a larger
# privatized-SMEM cap (-DCUB_HISTO_MAX_DYN_SMEM_BINS=N) and relink against the
# already-built nvbench objects, producing one binary per (cap,label):
#   bin/cub.bench.histogram.<label>.base.cap<N>
# Surgical (no full project rebuild). DO NOT run while a sweep is benchmarking on
# the GPU -- nvcc CPU/mem load perturbs timings. Run only when the GPU is idle.
set -euo pipefail
# Benchmark build dir (override with HIST_BENCH_BUILD).
BUILD=${HIST_BENCH_BUILD:-build/autocuda/cub-benchmark}
CMDS=/tmp/hist_compile_cmds.json
CAPS="${1:-24576 32768 49152}"
LABELS="even range multi_even multi_range"
cd "$BUILD"
for CAP in $CAPS; do
  echo "=== cap variant $CAP ==="
  for L in $LABELS; do
    OUT="bin/cub.bench.histogram.${L}.base.cap${CAP}"
    OBJ="/tmp/hist_${L}_cap${CAP}.o"
    echo "  compile+link $L -> $OUT"
    CAP="$CAP" L="$L" OBJ="$OBJ" OUT="$OUT" BUILD="$BUILD" CMDS="$CMDS" python3 - <<'PY'
import json, os, subprocess, shlex, sys
cap=os.environ['CAP']; L=os.environ['L']; obj=os.environ['OBJ']; out=os.environ['OUT']; build=os.environ['BUILD']
m=json.load(open(os.environ['CMDS']))
toks=shlex.split(m[L]['cmd'])
# Replace the -o <obj> target with our object path; keep original -c <src>.
i=toks.index('-o'); toks[i+1]=obj
# Add our define right after nvcc.
toks=[toks[0], f'-DCUB_HISTO_MAX_DYN_SMEM_BINS={cap}'] + toks[1:]
if os.path.exists(obj):
    print(f'    (reuse cached object {obj})')
else:
    r=subprocess.run(toks, cwd=build)
    if r.returncode!=0: sys.exit(f'compile {L} cap{cap} failed')
# Link: nvbench objects + our object -> standalone binary (libs verbatim from build.ninja).
helper=f'{build}/nvbench_helper/CMakeFiles/cccl.nvbench_helper.dir/nvbench_helper/nvbench_helper.cu.o'
mainobj=f'{build}/_deps/nvbench-build/nvbench/CMakeFiles/nvbench.main.dir/main.cu.o'
libs=shlex.split(f'-Xlinker -rpath -Xlinker {build}/lib /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so '
  f'{build}/lib/libnvbench.so -lstdc++fs /usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so '
  f'/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -ldl /usr/lib/x86_64-linux-gnu/librt.a '
  f'/usr/local/cuda/lib64/libcupti.so -lcudadevrt -lcudart_static -lrt -lpthread -ldl')
link=['/usr/local/cuda/bin/nvcc','-ccbin=/usr/bin/c++','-arch=native',helper,mainobj,obj,'-o',out]+libs
r=subprocess.run(link, cwd=build)
if r.returncode!=0: sys.exit(f'link {L} cap{cap} failed')
PY
  done
done
echo "=== built variants: ==="
ls -la bin/*cap* 2>/dev/null | awk '{print $5, $9}'
