#!/usr/bin/env bash

# TEMPORARY: prints one CPUSAMPLE line of raw /proc counters every few seconds while the given PID is alive.
# Used to see where build time goes on different runner sizes. Remove before merge.

parent="${1:?usage: cpu_sampler.sh <pid> [interval seconds]}"
interval="${2:-10}"

sample() {
  local now stat load psi vm disk procs running scc_pid scc
  now=$(date +%s)
  stat=$(awk '
    $1 == "cpu" { printf "u=%s n=%s s=%s i=%s w=%s q=%s sq=%s st=%s", $2, $3, $4, $5, $6, $7, $8, $9 }
    $1 == "ctxt" { printf " ctxt=%s", $2 }
    $1 == "processes" { printf " forks=%s", $2 }
    $1 == "procs_running" { printf " run=%s", $2 }
    $1 == "procs_blocked" { printf " blk=%s", $2 }' /proc/stat)
  load=$(cut -d" " -f1 /proc/loadavg)
  psi=$(awk '
    FNR == 1 { n = split(FILENAME, p, "/"); name = p[n] }
    { for (k = 2; k <= NF; k++) if ($k ~ /^total=/) { sub(/^total=/, "", $k); printf " psi_%s_%s=%s", name, $1, $k } }' \
    /proc/pressure/cpu /proc/pressure/io /proc/pressure/memory 2>/dev/null)
  vm=$(awk '$1 == "pgfault" || $1 == "pgmajfault" || $1 == "nr_dirty" || $1 == "nr_writeback" { printf " %s=%s", $1, $2 }' /proc/vmstat)
  disk=$(awk '$3 ~ /^(nvme[0-9]+n[0-9]+|sd[a-z]+|xvd[a-z]+|vd[a-z]+)$/ { r += $6; w += $10; t += $13 }
    END { printf " rsec=%d wsec=%d iotick=%d", r, w, t }' /proc/diskstats)
  procs=$(ps -eo pid=,state=,etimes=,comm= 2>/dev/null)
  running=$(awk '
    { total++; if ($2 == "D") d++ }
    $2 == "R" { r[$4]++ }
    END {
      printf " nprocs=%d dstate=%d R=", total, d
      n = 0
      for (c in r) { printf "%s%s:%d", (n++ ? "," : ""), c, r[c] }
    }' <<< "$procs")
  # The sccache server is the longest-lived sccache process; clients are short-lived.
  scc_pid=$(awk '$4 == "sccache" && $3 >= age { age = $3; pid = $1 } END { print pid }' <<< "$procs")
  scc=""
  if [[ -n "$scc_pid" && -r "/proc/$scc_pid/stat" ]]; then
    scc=$(awk '{ printf " scc_u=%s scc_s=%s scc_thr=%s", $14, $15, $20 }' "/proc/$scc_pid/stat" 2>/dev/null)
  fi
  echo "CPUSAMPLE t=$now hz=$hz $stat load=$load$psi$vm$disk$running$scc"
}

hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
while kill -0 "$parent" 2>/dev/null; do
  sample
  sleep "$interval"
done
