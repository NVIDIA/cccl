#!/bin/bash

# Check if input JSON, field, and file arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: ./get_groups.sh <input_json> <field> <output_file>"
  exit 1
fi

# Assign command-line arguments to variables
input_json="$1"
field="$2"
output_file="$3"

output=$(echo $input_json | jq -L . -c --arg field "$field" 'include "group_by_field"; .include | group_by_field($field)')
echo $output | tee -a "$output_file"

