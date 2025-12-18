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
import numpy as np
from datasets import Audio, Dataset, config, load_from_disk, Features, Value, Sequence


def prepare_dataframe(path_to_files: str, audio_extension: str = "wav", filter_empty: bool = True, max_files: int = None, folder_filter: str = None) -> pd.DataFrame:
    """
    Prepares audio and ground truth files as Pandas `DataFrame`.
    Recursively searches for audio files in all subdirectories.

    Args:
        path_to_files (str):
            Path to files.
        audio_extension (str, optional):
            Audio extension of files to include. Supports multiple formats separated by comma.
            Defaults to "wav". Common formats: wav, aac, mp3, flac, opus, m4a, ogg.
        filter_empty (bool, optional):
            Whether to filter out files with empty ground truth. Defaults to True.
        max_files (int, optional):
            Maximum number of files to include. Useful for testing or memory-limited processing.
            Defaults to None (no limit).
        folder_filter (str, optional):
            Prefix filter for folder names. Only folders starting with this prefix will be processed.
            Defaults to None (no filtering).

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
    # Support multiple extensions separated by comma
    extensions = [ext.strip() for ext in audio_extension.split(',')]
    
    print(f"🔍 Searching for audio files in: {path_to_files}")
    if folder_filter:
        print(f"   📁 Filtering folders starting with: {folder_filter}")
    
    audios = []
    for ext in extensions:
        found = sorted(glob(f"{path_to_files}/**/*.{ext}", recursive=True))
        
        # Apply folder filter if specified
        if folder_filter:
            filtered_found = []
            for audio_path in found:
                # Get the immediate subdirectory under path_to_files
                relative_path = Path(audio_path).relative_to(path_to_files)
                first_dir = str(relative_path.parts[0]) if relative_path.parts else ""
                
                # Check if the first directory starts with the filter prefix
                if first_dir.startswith(folder_filter):
                    filtered_found.append(audio_path)
            
            print(f"   Found {len(found)} .{ext} files, {len(filtered_found)} after folder filter")
            found = filtered_found
        else:
            print(f"   Found {len(found)} .{ext} files")
        
        audios.extend(found)
    
    # Remove duplicates and filter empty files
    audios = sorted(list(set(audios)))
    print(f"   Total unique files: {len(audios)}")
    audios = [a for a in audios if Path(a).stat().st_size > 0]
    print(f"   Non-empty files: {len(audios)}")
    
    if len(audios) == 0:
        raise ValueError(f"No audio files found with extensions: {', '.join(extensions)}")

    # Limit files if max_files is specified
    if max_files is not None and len(audios) > max_files:
        print(f"   ⚠️  Limiting to first {max_files} files (out of {len(audios)} total)")
        audios = audios[:max_files]
    
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

    if filter_empty:
        df = df[df["ground_truth"] != ""]

    return df


def format_audio_dataset(df: pd.DataFrame, sampling_rate: int = 16000, lazy_loading: bool = True) -> Dataset:
    """
    Formats Pandas `DataFrame` as a datasets `Dataset`.
    Converts `audio` path column to audio arrays and resamples accordingly.
    Supports WAV, AAC, MP3, FLAC, OPUS, M4A, OGG formats using librosa.

    Args:
        df (pd.DataFrame):
            Pandas DataFrame to convert to `Dataset`.
        sampling_rate (int):
            Target sampling rate for audio.
        lazy_loading (bool):
            If True, audio files are loaded on-demand rather than all at once.
            This significantly reduces memory usage for large datasets.
            Defaults to True.

    Returns:
        Dataset:
            `datasets`' `Dataset` object usable for batch inference.
    """
    # Non-WAV formats that need librosa for loading
    NON_WAV_FORMATS = ('.aac', '.mp3', '.flac', '.opus', '.m4a', '.ogg')
    
    first_audio = df['audio'].iloc[0] if len(df) > 0 else ""
    needs_librosa = first_audio.lower().endswith(NON_WAV_FORMATS)
    
    print(f"📊 Creating dataset with {len(df)} audio files (sampling_rate={sampling_rate}Hz)")
    
    if needs_librosa:
        # For AAC and other non-WAV formats, just store paths
        # Audio will be loaded on-demand by the transcriber
        print(f"   ✅ Using lazy loading (paths only, audio loaded on-demand)")
        print(f"   ℹ️  Audio arrays will be loaded during transcription to minimize memory usage")
        
        # Create a simple dataset with just file paths
        # Don't cast to Audio feature to avoid triggering soundfile
        dataset = Dataset.from_pandas(df)
        return dataset
    else:
        # For WAV files, use the standard approach with soundfile
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
    chars_to_remove_regex = r'[\,\?\.\!\-\;\:""'
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
