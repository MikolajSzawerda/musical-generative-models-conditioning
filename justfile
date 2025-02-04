default: prepare-audiocraft

# install dependencies for musicgen
prepare-audiocraft:
    cd src/audiocraft \
    && poetry run pip install "torch==2.1.0" \
    && poetry run pip install setuptools wheel torch\
    && poetry run pip install xformers==0.0.22.post7 --no-use-pep517 \
    && poetry install

# prepare dataset for app run
prepare-app-dataset dataset_name audio_dataset:
    mkdir models/{{ dataset_name }} \
    && cd src/audiocraft \
    && poetry run python3 prepare_dataset.py --audio-dir {{ audio_dataset }} --dataset ../../data/input/{{ dataset_name }}

# download example dataset for training
download-example-dataset:
    source .env \
    && wget --user=$STORAGE_LOGIN --password=$STORAGE_PASSWORD $STORAGE_URL/example-dataset.zip -O data/input/dataset.zip \
    && cd data/input
    && unzip dataset.zip

# install app dependencies
prepare-app: prepare-audiocraft
    cd src/app \
    && poetry run pip install "torch==2.1.0" \
    && poetry install

# run application
run-app dataset_path embeds_path:
    cd src/app \
    && poetry run python3 app.py --dataset {{ dataset_path }} --embeds-dir {{ embeds_path }}

# run sweep training
training-args:
    cd src/audiocraft \
    && poetry run python3 musicgen/trainer.py --help

# run sweep training
run-sweep dataset_name sweep_conifg:
    cd src/audiocraft \
    && poetry run python3 musicgen/trainer.py --sweep-cfg {{ sweep_conifg }} --dataset-name {{ dataset_name }}

# run sweep training
run-training dataset_name *args:
    cd src/audiocraft \
    && poetry run python3 musicgen/trainer.py --dataset-name {{ dataset_name }} {{ args }}

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
