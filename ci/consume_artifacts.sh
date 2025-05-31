#!/bin/bash

set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

dump_artifact() {
  local artifact_name="$1"
  local artifact_path="/tmp/artifact/$artifact_name"

  util/artifacts/download.sh "$artifact_name" "$artifact_path"
  tree "$artifact_path"
}

dump_artifact "test_artifact1"
dump_artifact "test_artifact2"
dump_artifact "test_artifact3"
dump_artifact "test_artifact4"
