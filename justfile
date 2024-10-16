default: compile-py-tools

#init handy project tools
compile-py-tools:
	poetry shell && poetry install

#uploads files to nextcloud
push src *dest:
	poetry run python3 src/tools/tools/nextcloud.py push {{src}} {{dest}}

#downloads files from nextcloud
pull src *dest:
	poetry run python3 src/tools/tools/nextcloud.py pull {{src}} {{dest}}

#combines files in provided dir into one audio file with ffmpeg
combine src output:
	sh scripts/ffmpeg-combine.sh {{src}} {{output}}

#splits audio file into multiple files with given length
split input length filename_format:
	sh scripts/ffmpeg-split.sh {{input}} {{length}} {{filename_format}}

#init env for audiocraft experiments
prepare-audiocraft:
	cd src/audiocraft \
	&& poetry install \
	&& poetry run python3 -c "from audiocraft.models import MusicGen;MusicGen.get_pretrained('facebook/musicgen-small')" \
	&& poetry run python -m ipykernel install --user --name audiocraft_lab --display-name "Python Audiocraft"

#	&& poetry run pip install setuptools wheel torch\
#	&& poetry run pip install xformers==0.0.22.post7 --no-use-pep517 \


#runs tensorboard
board exp:
	tensorboard --logdir=logs/{{exp}}

clear-board exp:
	rm -rf logs/{{exp}}
