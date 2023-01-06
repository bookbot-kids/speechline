# Copyright 2022 [PT BOOKBOT INDONESIA](https://bookbot.id/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import argparse
import sys
import subprocess
from p_tqdm import p_map
from glob import glob
from pathlib import Path
from functools import partial


def parse_args(args: List[str]) -> argparse.Namespace:
    """Utility argument parser function for batch audio conversion.

    Args:
        args (List[str]): List of arguments.

    Returns:
        argparse.Namespace: Objects with arguments values as attributes.
    """
    parser = argparse.ArgumentParser(
        prog="python speechline/utils/aac_to_wav.py",
        description="Batch-convert aac audios in a folder to wav format with ffmpeg.",
    )

    parser.add_argument(
        "input_dir", type=str, help="Directory of input audios to convert."
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        default=1,
        help="Number of audio channels in output.",
    )
    parser.add_argument(
        "-r", "--rate", type=int, default=16_000, help="Sample rate of audio output."
    )
    args = parser.parse_args(args)
    return args


def convert_to_wav(
    input_audio_path: str, num_channels: int = 1, sampling_rate: int = 16_000
) -> subprocess.CompletedProcess:
    """Convert aac audio file to wav at same directory.

    Args:
        input_audio_path (str): Path to aac file.
        num_channels (int, optional): Number of output audio channels. Defaults to 1.
        sampling_rate (int, optional): Sampling rate of output audio. Defaults to 16_000.

    Returns:
        subprocess.CompletedProcess: Finished subprocess.
    """
    # replace input file's extension to wav as output file path
    output_audio_path = Path(input_audio_path).with_suffix(".wav")

    # equivalent to:
    # `ffmpeg -i {input}.aac -acodec pcm_s16le -ac {num_channels} -ar {sampling_rate} {output}.wav`
    job = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-hide_banner",
            "-y",
            "-i",
            input_audio_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(num_channels),
            "-ar",
            str(sampling_rate),
            str(output_audio_path),
        ],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.PIPE,
    )

    return job


if __name__ == "__main__":
    parser = parse_args(sys.argv[1:])
    audios = glob(f"{parser.input_dir}/**/*.aac", recursive=True)
    fn = partial(convert_to_wav, num_channels=parser.channel, sampling_rate=parser.rate)
    _ = p_map(fn, audios)
