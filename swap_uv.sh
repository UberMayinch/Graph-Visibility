#!/bin/bash

# Directory to search; defaults to current directory if not given
DIR=${1:-.}

# Loop over all CSV files
for file in "$DIR"/*.csv; do
    [ -e "$file" ] || continue  # Skip if no csv files exist

    tmp_file="${file}.tmp"

    awk -F',' 'NR==1 {
        # Handle header: swap u and v
        for (i=1; i<=NF; i++) {
            if ($i == "u") ui = i;
            else if ($i == "v") vi = i;
            else if ($i == "time") ti = i;
        }
        print $ti "," $vi "," $ui
    }
    NR>1 {
        print $ti "," $vi "," $ui
    }' "$file" > "$tmp_file" && mv "$tmp_file" "$file"

    echo "Processed $file"
done
