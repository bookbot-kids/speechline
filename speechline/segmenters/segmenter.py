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

import os
from typing import Dict, List, Union

from pydub import AudioSegment

from ..utils.io import (
    export_segment_audio_wav,
    export_segment_transcripts_tsv,
    get_chunk_path,
    get_outdir_path,
)


class Segmenter:
    def _shift_offsets(
        self, offset: List[Dict[str, Union[str, float]]]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Shift start and end time of offsets by index start time.
        Subtracts all start and end times by index start time.

        Args:
            offset (List[Dict[str, Union[str, float]]]):
                Offsets to shift.

        Returns:
            List[Dict[str, Union[str, float]]]:
                Shifted offsets.
        """
        index_start = offset[0]["start_time"]
        shifted_offset = [
            {
                "text": o["text"],
                "start_time": round(o["start_time"] - index_start, 3),
                "end_time": round(o["end_time"] - index_start, 3),
            }
            for o in offset
        ]
        return shifted_offset

    def chunk_audio_segments(
        self,
        audio_path: str,
        outdir: str,
        offsets: List[Dict[str, Union[str, float]]],
        minimum_chunk_duration: float = 1.0,
        **kwargs,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Chunks an audio file based on its phoneme offsets.
        Generates and exports WAV audio chunks and aligned TSV phoneme transcripts.

        Args:
            audio_path (str):
                Path to audio file to chunk.
            outdir (str):
                Output directory to save chunked audio.
                Per-region subfolders will be generated under this directory.
            offsets (List[Dict[str, Union[str, float]]]):
                List of phoneme offsets.
            minimum_chunk_duration (float, optional):
                Minimum chunk duration (in seconds) to be exported.
                Defaults to 1.0 second.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of phoneme offsets for every segment.
        """
        segments = self.chunk_offsets(offsets, **kwargs)
        # skip empty segments (undetected transcripts)
        if len(segments) == 0:
            return [[{}]]

        audio = AudioSegment.from_file(audio_path)
        audio_segments: List[AudioSegment] = [
            audio[s[0]["start_time"] * 1000 : s[-1]["end_time"] * 1000]
            for s in segments
        ]

        # shift segments based on their respective index start times
        shifted_segments = [self._shift_offsets(segment) for segment in segments]

        # create output directory folder and subfolders
        os.makedirs(get_outdir_path(audio_path, outdir), exist_ok=True)

        for idx, (segment, audio_segment) in enumerate(
            zip(shifted_segments, audio_segments)
        ):
            # skip export if audio segment does not meet minimum chunk duration
            if len(audio_segment) < minimum_chunk_duration * 1000:
                continue

            # export TSV transcripts and WAV audio segment
            output_tsv_path = get_chunk_path(audio_path, outdir, idx, "tsv")
            export_segment_transcripts_tsv(output_tsv_path, segment)

            output_audio_path = get_chunk_path(audio_path, outdir, idx, "wav")
            export_segment_audio_wav(output_audio_path, audio_segment)

        return shifted_segments
