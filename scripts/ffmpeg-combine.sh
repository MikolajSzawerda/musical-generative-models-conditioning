ls $1 | sed "s/^/file '/;s/$/'/" > _filelist.txt
ffmpeg -f concat -safe 0 -i _filelist.txt -c copy $2
rm _filelist.txt
