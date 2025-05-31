#!/bin/bash

set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

mkdir -p build/dir1/sub1
mkdir -p build/dir1/sub2
mkdir -p build/dir2
mkdir -p build/dir3

touch build/dir1/file1.txt
touch build/dir1/sub1/file2.txt
touch build/dir1/sub2/file3.txt
touch build/dir2/file4.txt
touch build/dir3/file5.txt

tree build

util/artifacts/upload.sh test_artifact1 \
  'build/dir1/.*' \
  '.*file5.txt$'

util/artifacts/upload.sh test_artifact2 \
  'build/dir1/sub1/.*' \
  'build/dir2/.*'

util/artifacts/upload.sh test_artifact3 \
  'build/dir1/sub2/.*' \
  'build/dir3/.*'

util/artifacts/upload.sh test_artifact4 \
  'build/dir1/.*' \
  'build/dir2/.*' \
  'build/dir3/.*'
