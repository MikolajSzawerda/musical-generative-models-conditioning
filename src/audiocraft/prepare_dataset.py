import argparse
from pathlib import Path
import shutil
import tempfile
import subprocess
import uuid
from toolz import partition_all
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio_channels, convert_audio
import torch
import os
import json
import logging

logger = logging.getLogger(__name__)


def split_song(filepath: Path, file_length: int, dest: Path):
    command = [
        "ffmpeg",
        "-i",
        filepath,
        "-f",
        "segment",
        "-segment_time",
        str(file_length),
        "-c",
        "copy",
        "-reset_timestamps",
        "1",
        str(dest / "music_p%d.wav"),
    ]
    subprocess.run(command, check=True)


def clean_directory(directory: Path):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def already_prepared(path: Path) -> bool:
    return (path / "data").exists() and (path / "metadata_train.json").exists()


def extend_current(path: Path, new_data):
    with open(path, "r") as fh:
        fdata = json.load(fh)
    fdata.extend(new_data)
    with open(path, "w") as fh:
        json.dump(fdata, fh, indent=4, ensure_ascii=False)


@torch.no_grad()
def prepare(audio_dir: Path, ds_path: Path, f_length=5) -> None:
    concept_names = [d.name for d in audio_dir.iterdir() if d.is_dir()]
    logger.info(f"Will run prepare for {len(concept_names)} concepts.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    metadata_path = ds_path / "metadata_train.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps([], indent=4))
    for c in concept_names:
        logger.info(f"Preparing concept: {c}")
        new_rows = []
        audio_segment_dir = ds_path / "data" / "train" / c / "audio"
        encoded_segment_dir = ds_path / "data" / "train" / c / "encoded"
        clean_directory(audio_segment_dir)
        clean_directory(encoded_segment_dir)
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        logger.info(f"Splittng audio")

        for file in (audio_dir / c).rglob("*"):  # Recursively iterate through all files
            if file.suffix.lower() not in audio_extensions or not file.is_file():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                split_song(file, f_length, tmp_dir)
                segments = list(tmp_dir.glob("music_p*.wav"))
                for segment in segments:
                    seg_name = f"{str(uuid.uuid4())[:5]}_{segment.name}"
                    segment.rename(
                        audio_segment_dir / seg_name,
                    )
                    new_rows.append(
                        {
                            "audio_path": os.path.join(
                                "data", "train", c, "audio", seg_name
                            ),
                            "encoded_path": os.path.join(
                                "data",
                                "train",
                                c,
                                "encoded",
                                seg_name.replace(".wav", ".pt"),
                            ),
                            "concept": c,
                            "track_id": str(uuid.uuid4()),
                        }
                    )
        logger.info(f"Encoding audio")

        segmented_audio = audio_segment_dir.glob("*.wav")
        batches = partition_all(10, segmented_audio)
        for batch in batches:
            music_batch = []
            for f_audio in batch:
                music, sr = audio_read(str(f_audio), duration=f_length, pad=True)
                music = music[None]
                music_batch.append(
                    convert_audio(music, sr, 32000, 1).squeeze().unsqueeze(0)
                )
            encoded_music, _ = model.compression_model.encode(
                torch.stack(music_batch).to(DEVICE)
            )
            for f_audio, encoded in zip(batch, encoded_music):
                dest = encoded_segment_dir / f_audio.name.replace(".wav", ".pt")
                torch.save(encoded_music.cpu().type(torch.int64), dest)
        logger.info(f"Saving concept: {c}")

        extend_current(metadata_path, new_rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, required=True)

    args = parser.parse_args()
    ds_path = Path(args.dataset)
    if args.audio_dir is not None:
        prepare(Path(args.audio_dir), ds_path)
