default: compile-py-tools

compile-py-tools:
	poetry shell && poetry install
push src *dest:
	poetry run python3 src/tools/nextcloud.py push {{src}} {{dest}}
pull src *dest:
	poetry run python3 src/tools/nextcloud.py pull {{src}} {{dest}}

combine src output:
	sh scripts/ffmpeg-combine.sh {{src}} {{output}}
split input length filename_format:
	sh scripts/ffmpeg-split.sh {{input}} {{length}} {{filename_format}}

prepare-audiocraft:
	cd src/audiocraft \
	&& pyenv local 3.9 \
	&& poetry run pip install setuptools wheel torch\
	&& poetry run pip install xformers==0.0.22.post7 --no-use-pep517 \
	&& poetry install \
	&& poetry run python3 -c "from audiocraft.models import MusicGen;MusicGen.get_pretrained('facebook/musicgen-small')" \
	&& poetry run python -m ipykernel install --user --name audiocraft_lab --display-name "Python Audiocraft"


board exp:
	tensorboard --logdir=logs/{{exp}}
