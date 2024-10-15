ffmpeg -i $1 -f segment -segment_time $2 -c copy -reset_timestamps 1 $3
