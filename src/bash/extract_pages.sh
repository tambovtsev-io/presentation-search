#!/bin/bash

# Get script directory and navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"  # Go up two levels from src/bash to project root

# Change to base directory
cd "$BASE_DIR"

# Default input and output directories
INPUT_DIR="$BASE_DIR/data/raw"
OUTPUT_DIR="$BASE_DIR/data/raw/weird-slides"

# Check and create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if at least 2 arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 filename_part page1 page2 ... pageN"
    exit 1
fi

# Get the filename part (first argument)
filename_part="$1"
shift  # Remove first argument, leaving only page numbers

# Find the input PDF file matching the filename part
input_pdf=$(find "$INPUT_DIR" -type f -iname "*$filename_part*.pdf" -print -quit)

# Check if the input PDF exists
if [ ! -f "$input_pdf" ]; then
    echo "Error: $input_pdf not found."
    exit 1
fi

# Get the base name without extension
base_name=$(basename "$input_pdf" .pdf)

# Join the remaining arguments (page numbers) with spaces
pages=$(echo "$@" | tr ' ' ' ')

# Create the output file name with specified slides
output_pdf="$OUTPUT_DIR/${base_name}_slides-${pages// /-}.pdf"

# Use pdftk to extract pages
pdftk "$input_pdf" cat $pages output "$output_pdf"

# Check if the operation was successful
if [ $? -eq 0 ]; then
    echo "Pages $pages from $input_pdf extracted to $output_pdf."
else
    echo "Failed to extract pages."
fi
