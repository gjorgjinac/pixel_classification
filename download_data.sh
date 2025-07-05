#!/bin/bash

# Output filename
OUTPUT_FILE="20170710_s2_manual_classification_data.h5"
# Raw download URL (correct GitLab format)
RAW_URL="https://git.gfz-potsdam.de/EnMAP/sentinel2_manual_classification_clouds/-/raw/master/20170710_s2_manual_classification_data.h5"
echo "Downloading dataset..."
wget -O "$OUTPUT_FILE" "$RAW_URL"

# Check if download succeeded
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "ERROR: Download failed or file not found!"
    exit 1
fi

echo "Download complete: $OUTPUT_FILE"