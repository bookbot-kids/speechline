# Copyright 2023 [PT BOOKBOT INDONESIA](https://bookbot.id/)
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

import re
from glob import glob
from pathlib import Path
import json

import pandas as pd
from datasets import Audio, Dataset, config, load_from_disk


def prepare_dataframe(path_to_files: str, audio_extension: str = "wav") -> pd.DataFrame:
    """
    Prepares audio and ground truth files as Pandas `DataFrame`.
    Recursively searches for audio files in all subdirectories.

    Args:
        path_to_files (str):
            Path to files.
        audio_extension (str, optional):
            Audio extension of files to include. Defaults to "wav".

    Raises:
        ValueError: No audio files found.

    Returns:
        pd.DataFrame:
            DataFrame consisting of:

        - `audio` (audio path)
        - `id`
        - `language`
        - `language_code`
        - `ground_truth`
    """
    audios = sorted(glob(f"{path_to_files}/**/*.{audio_extension}", recursive=True))
    audios = [a for a in audios if Path(a).stat().st_size > 0]
    if len(audios) == 0:
        raise ValueError("No audio files found!")

    df = pd.DataFrame({"audio": audios})
    # ID is filename stem (before extension)
    df["id"] = df["audio"].apply(lambda f: Path(f).stem)
    # language code is immediate parent directory
    df["language_code"] = df["audio"].apply(lambda f: Path(f).parent.name)
    df["language"] = df["language_code"].apply(lambda f: f.split("-")[0])
    # ground truth is same filename, except with .txt extension
    df["ground_truth"] = df["audio"].apply(lambda p: Path(p).with_suffix(".txt"))
    df["ground_truth"] = df["ground_truth"].apply(
        lambda p: open(p).read() if p.exists() else ""
    )

    df = df[df["ground_truth"] != ""]

    return df


def format_audio_dataset(df: pd.DataFrame, sampling_rate: int = 16000) -> Dataset:
    """
    Formats Pandas `DataFrame` as a datasets `Dataset`.
    Converts `audio` path column to audio arrays and resamples accordingly.

    Args:
        df (pd.DataFrame):
            Pandas DataFrame to convert to `Dataset`.

    Returns:
        Dataset:
            `datasets`' `Dataset` object usable for batch inference.
    """
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(str(config.HF_DATASETS_CACHE))
    saved_dataset = load_from_disk(str(config.HF_DATASETS_CACHE))
    saved_dataset = saved_dataset.cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )
    return saved_dataset


def preprocess_audio_transcript(text: str) -> str:
    """
    Preprocesses audio transcript.
    - Removes punctuation.
    - Converts to lowercase.
    - Removes special tags (e.g. GigaSpeech).
    """
    tags = [
        "<COMMA>",
        "<PERIOD>",
        "<QUESTIONMARK>",
        "<EXCLAMATIONPOINT>",
        "<SIL>",
        "<MUSIC>",
        "<NOISE>",
        "<OTHER>",
    ]
    chars_to_remove_regex = '[\,\?\.\!\-\;\:""'
    text = re.sub(chars_to_remove_regex, " ", text).lower().strip()
    text = re.sub(r"\s+", " ", text).strip()
    for tag in tags:
        text = text.replace(tag.lower(), "").strip()
    return text


def prepare_dataframe_from_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Prepares audio and ground truth files as Pandas `DataFrame` from a manifest file.

    Args:
        manifest_path (str):
            Path to the manifest JSON file.

    Raises:
        ValueError: No valid entries found in manifest file.

    Returns:
        pd.DataFrame:
            DataFrame consisting of:

        - `audio` (audio path)
        - `id`
        - `language_code`
        - `language`
        - `ground_truth`
    """
    entries = []
    try:
        # Load the JSON file as a complete array
        with open(manifest_path, "r") as f:
            json_data = json.load(f)

        # Process each entry in the array
        for entry in json_data:
            if "audio" in entry and "text" in entry:
                audio_path = entry["audio"]
                # Check if the audio file exists
                if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
                    # Use the provided fields directly when available
                    entries.append(
                        {
                            "audio": audio_path,
                            "id": entry.get("id", Path(audio_path).stem),
                            "language_code": entry.get(
                                "accent",
                                entry.get("language", Path(audio_path).parent.name),
                            ),
                            "language": entry.get(
                                "language", Path(audio_path).parent.name.split("-")[0]
                            ),
                            "ground_truth": entry.get("text", ""),
                        }
                    )
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse manifest file: {e}")

    if not entries:
        raise ValueError("No valid entries found in manifest file!")

    df = pd.DataFrame(entries)
    df = df[df["ground_truth"] != ""]

    return df
