default: prepare-audiocraft

# install dependencies for musicgen
prepare-audiocraft:
    cd src/audiocraft \
    && poetry run pip install "torch==2.1.0" \
    && poetry run pip install setuptools wheel torch\
    && poetry run pip install xformers==0.0.22.post7 --no-use-pep517 \
    && poetry install

# prepare dataset for experiments
prepare-dataset dataset_name audio_dataset:
    mkdir models/{{ dataset_name }} \
    && cd src/audiocraft \
    && poetry run python3 prepare_dataset.py --audio-dir {{ audio_dataset }} --dataset ../../data/input/{{ dataset_name }}

# uploads files to nextcloud
push src *dest:
    poetry run python3 src/tools/tools/nextcloud.py push {{ src }} {{ dest }}

# downloads files from nextcloud
pull src *dest:
    poetry run python3 src/tools/tools/nextcloud.py pull {{ src }} {{ dest }}

# combines files in provided dir into one audio file with ffmpeg
combine src output:
    sh scripts/ffmpeg-combine.sh {{ src }} {{ output }}

# splits audio file into multiple files with given length
split input length filename_format:
    sh scripts/ffmpeg-split.sh {{ input }} {{ length }} {{ filename_format }}

#	&& poetry run pip install setuptools wheel torch\
#	&& poetry run pip install xformers==0.0.22.post7 --no-use-pep517 \
