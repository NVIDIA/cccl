#!/usr/bin/env bash

set -euo pipefail

# Print current dist status to verify we're connected
# Print current dist status to verify we're connected
sccache --dist-status | jq -r -f <(cat <<"EOF"
  def info_to_row: {
    time: now | floor,
    type: (.type // "server"),
    id: .id,
    servers: ((.servers | length) // 1),
    cpus: .info.occupancy,
    util: ((.info.cpu_usage // 0) * 100 | round | . / 100 | tostring | . + "%"),
    jobs: (.jobs.loading + .jobs.pending + .jobs.running),
    loading: .jobs.loading,
    pending: .jobs.pending,
    running: .jobs.running,
    accepted: .jobs.accepted,
    finished: .jobs.finished,
    u_time: ((.u_time // 0) | tostring | . + "s")
  };

  .SchedulerStatus as [$x, $y] | [
    ($y + { id: $x, type: "scheduler", u_time: ($y.servers // {} | map(.u_time) | min | . // 0 | tostring) }),
    ($y.servers // [] | sort_by(.id)[])
  ]
  | map(info_to_row) as $rows
  | ($rows[0] | keys_unsorted) as $cols
  | ($rows | map(. as $row | $cols | map($row[.]))) as $rows
  | $cols, $rows[] | @csv
EOF
)
