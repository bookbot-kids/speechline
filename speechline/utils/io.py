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

import json
import os
from typing import Dict, List, Union

from pydub import AudioSegment


def export_transcripts_json(
    output_json_path: str,
    phoneme_offsets: List[Dict[str, Union[str, float]]],
) -> None:
    """
    Exports phoneme transcript with offsets as JSON.

    ```json title="example_output_transcripts.json"
    [
      {
        "phoneme": {phoneme},
        "start_time": {start_time},
        "end_time": {end_time}
      },
      {
        "phoneme": {phoneme},
        "start_time": {start_time},
        "end_time": {end_time}
      },
      ...
    ]
    ```

    Args:
        output_json_path (str):
            Path to output JSON file.
        phoneme_offsets (List[Dict[str, Union[str, float]]]):
            List of phonemes with offsets.
    """
    with open(output_json_path, "w") as f:
        json.dump(phoneme_offsets, f, indent=2)


def export_segment_transcripts_tsv(
    output_tsv_path: str, segment: List[Dict[str, Union[str, float]]]
) -> None:
    """
    Export segment transcripts to TSV of structure:

    ```tsv title="example_output_segment_transcripts.tsv"
    start_time_in_secs\tend_time_in_secs\tlabel
    start_time_in_secs\tend_time_in_secs\tlabel
    ...
    ```

    Args:
        output_tsv_path (str):
            Path to TSV file.
        segment (List[Dict[str, Union[str, float]]]):
            List of phonemes in segment.
    """
    with open(output_tsv_path, "w") as f:
        for s in segment:
            f.write(f'{s["start_time"]}\t{s["end_time"]}\t{s["phoneme"]}\n')


def export_segment_audio_wav(output_wav_path: str, segment: AudioSegment) -> None:
    """
    Export segment audio to WAV.

    Equivalent to:

    ```sh title="example_export_segment_audio_wav.sh"
    ffmpeg -i {segment} -acodec pcm_s16le -ac 1 -ar 16000 {output_wav_path}
    ```

    Args:
        output_wav_path (str):
            Path to WAV file.
        segment (AudioSegment):
            Audio segment to export.
    """
    parameters = ["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"]
    segment.export(output_wav_path, format="wav", parameters=parameters)


def get_outdir_path(path: str, outdir: str) -> str:
    """
    Generate path at output directory.

    Assumes `path` as `{inputdir}/{lang}/*.wav`,
    and will return `{outdir}/{lang}/*.wav`

    Args:
        path (str):
            Path to file.
        outdir (str):
            Output directory where file will be saved.

    Returns:
        str:
            Path to output directory.
    """
    pathname, _ = os.path.splitext(path)  # remove extension
    components = os.path.normpath(pathname).split(os.sep)  # split into components
    # keep last 2 components: {outdir}/{lang}/
    output_path = f"{os.path.join(outdir, *components[-2:-1])}"
    return output_path


def get_chunk_path(path: str, outdir: str, idx: int, extension: str) -> str:
    """
    Generate path to chunk at output directory.

    Assumes `path` as `{inputdir}/{lang}/{utt_id}.{old_extension}`,
    and will return `{outdir}/{lang}/{utt_id}-{idx}.{extension}`

    Args:
        path (str):
            Path to file.
        outdir (str):
            Output directory where file will be saved.
        idx (int):
            Index of chunk.
        extension (str):
            New file extension.

    Returns:
        str:
            Path to chunk at output directory.
    """
    outdir_path = get_outdir_path(path, outdir)
    filename = os.path.splitext(os.path.basename(path))[0]
    output_path = f"{os.path.join(outdir_path, filename)}-{str(idx)}.{extension}"
    return output_path
