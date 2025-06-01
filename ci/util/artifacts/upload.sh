#!/bin/bash

set -euo pipefail

readonly ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
source "$ci_dir/util/artifacts/common.sh"

readonly usage=$(cat <<EOF
Usage: $0 <name> [<regex> ...]

Regexes are passed to the `find` command's -regex option in the current directory.
'./' is prepended to all regexes for convenience.
The artifact will contain files with paths relative to the current directory.

If no regexes are provided, the artifact will be created from a file in the current directory.
The file must have the same name as the artifact.

Example Usage:

  Create an artifact of the given file using the filename as the artifact name:

    $0 source_archive.tar.gz

  Create a .tag.bz2 archive of all files matching the regexes and upload it as
  an packed artifact named 'job-\$ID-products':

    $0 job-\$ID-products \
       'build\.ninja$' \
       '.*rules\.ninja$' \
       'CMakeCache\.txt$' \
       '.*VerifyGlobs\.cmake$' \
       '.*CTestTestfile\.cmake$' \
       'bin/.*' \
       'lib/.*'
EOF
)

if [ "$#" -lt 1 ]; then
  echo "Error: Missing artifact name." >&2
  echo "$usage" >&2
  exit 1
fi

readonly artifact_name="$1"

# If no regexes are provided, use the artifact name as the path:
if [ "$#" -eq 1 ]; then
  "$ci_dir/util/artifacts/upload/register.sh" "$artifact_name" "$artifact_name"
  exit
fi

shift

readonly artifact_path="$ARTIFACT_UPLOAD_STAGE/$artifact_name"
readonly artifact_archive="$ARTIFACT_ARCHIVES/${artifact_name}.tar.bz2"

mkdir -p "$artifact_path"

"$ci_dir/util/artifacts/upload/stage.sh" "$artifact_path" "$@"
"$ci_dir/util/artifacts/upload/pack.sh" "$artifact_path" "$artifact_archive"
"$ci_dir/util/artifacts/upload/register.sh" "$artifact_name" "$artifact_archive"
