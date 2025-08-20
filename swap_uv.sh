#!/bin/bash
# preprocessing script. finds all csv files and exchanges fields 1 and 3 to make reading faster. Also removes . from the end
# because that makes reading the file names impossible.

# Directory to search; defaults to current directory if not given
DIR=${1:-.}

# Loop over all CSV files that start with output_
for file in "$DIR"/output_*.csv; do
    [ -e "$file" ] || continue  # Skip if no csv files exist

    tmp_file="${file}.tmp"

    awk -F',' 'BEGIN{OFS=","} {
        # Swap column 1 and column 3
        temp = $1
        $1 = $3
        $3 = temp
        print
    }' "$file" > "$tmp_file" && mv "$tmp_file" "$file"

    echo "Processed $file"
done
