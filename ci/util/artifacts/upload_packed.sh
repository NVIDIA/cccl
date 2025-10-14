#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <name> <regex> [<regex> ...]

Create a compressed artifact, suitable for large, temporary files such as build products or test binaries
that need to be quickly uploaded and downloaded between CI jobs. The artifact will exist of a
zip file containing an <artifact_name>.tar.zst archive, packed with the parallel zstd.

Regexes are passed to the `find` command's -regex option in the current directory.
'./' is prepended to all regexes for convenience.
The artifact will contain all matching files relative to the current directory.

If no regexes are provided, the artifact will be created from a file in the current directory.
The file must have the same name as the artifact.

Example Usage:

  Create an artifact of the given file in the current directory using the filename as the artifact name:

    $0 some_resource.log

  Copy all files that match the regexes to a staging directory, and upload a artifact that zips this directory.

    $0 job-\$JOB_ID-products \
       'build\.ninja$' \
       '.*rules\.ninja$' \
       'CMakeCache\.txt$' \
       '.*VerifyGlobs\.cmake$' \
       '.*CTestTestfile\.cmake$' \
       'bin/.*' \
       'lib/.*'
EOF
)

if [ "$#" -lt 2 ]; then
  echo "Error: Invalid number of arguments." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"
shift

"$ci_dir/util/artifacts/stage.sh" "$artifact_name" "$@" > /dev/null
"$ci_dir/util/artifacts/upload_stage_packed.sh" "$artifact_name"
