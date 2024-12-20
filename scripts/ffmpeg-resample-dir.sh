#!/bin/bash

# Check if the user provided a target directory
if [ -z "$1" ]; then
  echo "Usage: $0 <target_directory>"
  exit 1
fi

# Set the target directory and desired sample rate
TARGET_DIR=$1
TARGET_RATE=$2

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory $TARGET_DIR does not exist."
  exit 1
fi

# Iterate over all audio files in the target directory
for file in "$TARGET_DIR"/*.{wav,mp3,flac,aac}; do
  # Check if the file exists and is not a directory
  if [ -f "$file" ]; then
    # Get the file's extension and base name
    EXT="${file##*.}"
    BASENAME="${file%.*}"
    
    # Create a temporary file for resampling
    TEMP_FILE="${BASENAME}_temp.${EXT}"
    
    # Resample the file
    ffmpeg -i "$file" -ar $TARGET_RATE -y "$TEMP_FILE"
    
    # Replace the original file with the resampled file
    mv "$TEMP_FILE" "$file"
    echo "Resampled $file to $TARGET_RATE Hz"
  fi
done

echo "Resampling completed for all files in $TARGET_DIR."
