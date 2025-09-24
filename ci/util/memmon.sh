#!/usr/bin/env bash
set -euo pipefail

pid_file="/tmp/.memmon.pid"

log_threshold="2"
print_threshold="5"
poll_interval="5"
log_file="$PWD/memmon.log"
mode=""

usage() {
  cat <<'USAGE'
Monitors running processes for high memory usage, logging and reporting peaks above specified thresholds.

Usage: memmon.sh (--start | --stop | --monitor | --help)
        [--log-threshold <GB>]   # Write to log file if process exceeds this memory (default 2 GB)
        [--print-threshold <GB>] # Print to stdout if process exceeds this memory (default 5 GB)
        [--poll <seconds>]       # Poll interval for checking processes (default 5 seconds)
        [--log-file <file>]      # Log file path (default ./memmon.log)

Modes:

  --start    Start monitoring in the background (writes pid to /tmp/.memmon.pid)
  --stop     Stop monitoring (kills pid in /tmp/.memmon.pid)
  --monitor  Run monitoring in the foreground (for testing/debugging)

Example Session:

  memmon.sh --start                 # Start monitoring
  launch_memory_intensive_processes # Do work
  memmon.sh --stop                  # Stop monitoring and write log
  cat memmon.log                    # View log
USAGE
}

error() {
  echo "memmon: $*" >&2
  exit 1
}

error_usage() {
  echo "memmon: $*" >&2
  usage
  exit 1
}

ensure_absolute_log() {
  case "$log_file" in
    /*) return ;;
    *)
      local dir="$(cd "$(dirname "$log_file")" && pwd)"
      local base="$(basename "$log_file")"
      log_file="$dir/$base"
      ;;
  esac
}

to_kib() {
  awk -v gib="$1" 'BEGIN {printf "%.0f", gib * 1024 * 1024}'
}

format_gib() {
  awk -v rss="$1" 'BEGIN {printf "%.3f", rss/1024/1024}'
}

format_threshold() {
  awk -v val="$1" 'BEGIN {printf "%.3f", val + 0}'
}

declare -A MEMMON_MAX_RSS
declare -A MEMMON_CMD
declare -A MEMMON_TARGET

get_cmdline() {
  local pid="$1"
  local cmdline_file="/proc/$pid/cmdline"
  local raw

  # Attempt to read from /proc first for accuracy
  if [[ -r "$cmdline_file" ]]; then
    raw="$(tr '\0' ' ' <"$cmdline_file" 2>/dev/null)"
    # Clean up whitespace
    raw="${raw//$'\n'/ }"
    raw="${raw//$'\r'/ }"
    raw="${raw//$'\t'/ }"
    while [[ "$raw" == *' ' ]]; do
      raw="${raw% }"
    done
    if [[ -n "$raw" ]]; then
      printf '%s' "$raw"
      return 0
    fi
  fi
  # Fallback to ps if /proc is unavailable
  raw="$(ps -wwp "$pid" -o command= 2>/dev/null | head -n1)"
  # Clean up whitespace
  raw="${raw//$'\n'/ }"
  raw="${raw//$'\r'/ }"
  raw="${raw//$'\t'/ }"
  if [[ -n "$raw" ]]; then
    printf '%s' "$raw"
    return 0
  fi
  printf '[command unavailable]'
}

extract_target() {
  local cmd="$1"
  # Try to locate the name of the cmake target:
  if [[ "$cmd" =~ CMakeFiles/([^[:space:]]*)\.dir ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '-'
  fi
}

start_memmon() {
  # Check if already running
  if [[ -f "$pid_file" ]]; then
    local existing_pid="$(<"$pid_file")"
    if kill -0 "$existing_pid" 2>/dev/null; then
      error "already running (pid $existing_pid)"
    fi
    rm -f "$pid_file"
  fi

  ensure_absolute_log

  # Start monitoring in the background
  "$0" --monitor \
       --log-threshold "$log_threshold" \
       --print-threshold "$print_threshold" \
       --poll "$poll_interval" \
       --log-file "$log_file" &
  local child_pid=$!
  echo "$child_pid" >"$pid_file"
  echo "memmon started (pid $child_pid, log-threshold ${log_threshold}GB, print-threshold ${print_threshold}GB, log $log_file)"
}

stop_memmon() {
  if [[ ! -f "$pid_file" ]]; then
    error "not running"
  fi
  local running_pid="$(<"$pid_file")"
  if ! kill -0 "$running_pid" 2>/dev/null; then
    rm -f "$pid_file"
    error "not running"
  fi

  # Attempt graceful shutdown
  kill "$running_pid" 2>/dev/null || true

  for _ in {1..20}; do
    if ! kill -0 "$running_pid" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done

  # Force kill if still running
  if kill -0 "$running_pid" 2>/dev/null; then
    kill -9 "$running_pid" 2>/dev/null || true
  fi

  # Wait for pid file to be removed
  for _ in {1..20}; do
    [[ ! -f "$pid_file" ]] && break
    sleep 0.25
  done
  [[ -f "$pid_file" ]] && rm -f "$pid_file"
  echo "memmon stopped"
}

monitor_mem() {
  MEMMON_MAX_RSS=()
  MEMMON_CMD=()
  MEMMON_TARGET=()

  ensure_absolute_log

  local log_threshold_kib="$(to_kib "$log_threshold")"
  local print_threshold_kib="$(to_kib "$print_threshold")"

  local running=true

  cleanup() {
    trap - INT TERM EXIT
    mkdir -p "$(dirname "$log_file")"
    {
      printf "peak-mem | PID | target | command-line\n"
      if [[ ${#MEMMON_MAX_RSS[@]} -eq 0 ]]; then
        printf "No processes exceeded %s GB\n" "$(format_threshold "$log_threshold")"
      else
        local tmp="$(mktemp)"
        for pid in "${!MEMMON_MAX_RSS[@]}"; do
          printf "%s\t%s\t%s\t%s\n" "${MEMMON_MAX_RSS[$pid]}" "$pid" "${MEMMON_TARGET[$pid]}" "${MEMMON_CMD[$pid]}" >>"$tmp"
        done
        sort -nr -k1,1 "$tmp" | while IFS=$'\t' read -r peak pid target cmd; do
          local mem_gib="$(format_gib "$peak")"
          printf "%s GB | %s | %s | %s\n" "$mem_gib" "$pid" "$target" "$cmd"
        done
        rm -f "$tmp"
      fi
    } >"$log_file"
    echo "memmon log written to $log_file"
    rm -f "$pid_file"
  }

  trap 'running=false' INT TERM
  trap cleanup EXIT

  while $running; do
    while read -r pid rss; do
      [[ -z "$pid" || -z "$rss" ]] && continue
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      [[ "$rss" =~ ^[0-9]+$ ]] || continue
      if (( rss >= log_threshold_kib )); then
        local current=${MEMMON_MAX_RSS[$pid]:-0}
        if (( rss > current )); then
          MEMMON_MAX_RSS[$pid]=$rss
          MEMMON_CMD[$pid]=$(get_cmdline "$pid")
          MEMMON_TARGET[$pid]=$(extract_target "${MEMMON_CMD[$pid]}")
          if (( rss >= print_threshold_kib )); then
            local mem_gib="$(format_gib "$rss")"
            printf 'memmon: %s GB | %s | %s | %s\n' "$mem_gib" "$pid" "${MEMMON_TARGET[$pid]}" "${MEMMON_CMD[$pid]}"
          fi
        fi
      fi
    done < <(ps -eo pid=,rss=)

    if ! sleep "$poll_interval"; then
      break
    fi
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-threshold)
      [[ $# -lt 2 ]] && error "--log-threshold requires a value"
      log_threshold="$2"
      shift 2
      ;;
    --print-threshold)
      [[ $# -lt 2 ]] && error "--print-threshold requires a value"
      print_threshold="$2"
      shift 2
      ;;
    --log-file)
      [[ $# -lt 2 ]] && error "--log-file requires a value"
      log_file="$2"
      shift 2
      ;;
    --poll)
      [[ $# -lt 2 ]] && error "--poll requires a value"
      poll_interval="$2"
      shift 2
      ;;
    --start)
      [[ -n "$mode" ]] && error "Specify only one of --start or --stop"
      mode="start"
      shift
      ;;
    --stop)
      [[ -n "$mode" ]] && error "Specify only one of --start or --stop"
      mode="stop"
      shift
      ;;
    --monitor)
      mode="monitor"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      error_usage "Unknown option: $1"
      ;;
  esac
done

if [[ -z "$mode" ]]; then
  error_usage "Must specify one of --start or --stop or --monitor"
fi

case "$mode" in
  start)
    start_memmon
    ;;
  stop)
    stop_memmon
    ;;
  monitor)
    monitor_mem
    ;;
  *)
    error_usage "Unhandled mode: $mode"
    ;;
esac
