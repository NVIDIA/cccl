# Groups an array of objects by a field and returns an object with the field value as key and the array of objects as value
# Handles nested fields (e.g. "compiler.name")
def group_by_field($field):
    group_by(getpath($field | split("."))) | map( {(.[0] | getpath($field | split("."))): .}) | add;
