#!/bin/bash

# Ensure that the script is being executed from its directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Check if input JSON and field arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: ./get_groups.sh <input_json> <field>"
  exit 1
fi

# Assign command-line arguments to variables
input_json="$1"
field="$2"

output=$(echo $input_json | jq -L . -c --arg field "$field" 'include "group_by_field"; group_by_field($field)')
echo $output 

