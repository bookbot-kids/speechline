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

from datasets import Audio, Dataset
from pydub import AudioSegment

from ..modules import AudioModule
from ..utils.io import (
    export_segment_audio_wav,
    export_segment_transcripts_tsv,
    get_chunk_path,
    get_outdir_path,
    pydub_to_np,
)


class Segmenter:
    def chunk_audio_segments(
        self,
        audio_path: str,
        outdir: str,
        offsets: List[Dict[str, Union[str, float]]],
        do_noise_classify: bool = False,
        minimum_chunk_duration: float = 1.0,
        **kwargs,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Chunks an audio file based on its offsets.
        Generates and exports WAV audio chunks and aligned TSV phoneme transcripts.

        Args:
            audio_path (str):
                Path to audio file to chunk.
            outdir (str):
                Output directory to save chunked audio.
                Per-region subfolders will be generated under this directory.
            offsets (List[Dict[str, Union[str, float]]]):
                List of phoneme offsets.
            do_noise_classify (bool, optional):
                Whether to perform noise classification on empty chunks.
                Defaults to `False`.
            minimum_chunk_duration (float, optional):
                Minimum chunk duration (in seconds) to be exported.
                Defaults to 0.3 second.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                List of offsets for every segment.
        """
        segments = self.chunk_offsets(offsets, **kwargs)
        # skip empty segments (undetected transcripts)
        if len(segments) == 0:
            return [[{}]]

        if do_noise_classify:
            segments = self.insert_empty_tags(segments, **kwargs)
            segments = self.classify_noise(segments, audio_path, **kwargs)

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

    def classify_noise(
        self,
        segments: List[List[Dict[str, Union[str, float]]]],
        audio_path: str,
        noise_classifier: AudioModule,
        noise_classifier_threshold: float,
        empty_tag: str = "<EMPTY>",
        **kwargs,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Classify empty tags as noise.

        Args:
            segments (List[List[Dict[str, Union[str, float]]]]):
                List of chunked segments with empty tag.
            audio_path (str):
                Path to audio file to chunk.
            noise_classifier (AudioModule):
                Audio Module to perform noise classification.
            noise_classifier_threshold (float):
                Minimum probability threshold for multi label classification.
            empty_tag (str, optional):
                Special empty tag.
                Defaults to `"<EMPTY>"`.

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                Chunk segments with classified noise tags.
        """
        pos, empty_tag_pos = 0, {}
        for i, segment in enumerate(segments):
            for j, offset in enumerate(segment):
                if offset["text"] == empty_tag:
                    empty_tag_pos[pos] = (i, j)
                    pos += 1

        # return original segments if no empty tags
        if len(empty_tag_pos) == 0:
            return segments

        audio = AudioSegment.from_file(audio_path)
        audio_arrays = [
            {
                "path": None,
                "array": pydub_to_np(
                    audio[offset["start_time"] * 1000 : offset["end_time"] * 1000]
                ),
                "sampling_rate": audio.frame_rate,
            }
            for segment in segments
            for offset in segment
            if offset["text"] == empty_tag
        ]

        dataset = Dataset.from_dict({"audio": audio_arrays})
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=noise_classifier.sampling_rate)
        )

        outputs = noise_classifier.predict(
            dataset, threshold=noise_classifier_threshold
        )

        for idx, predictions in enumerate(outputs):
            if len(predictions) > 0:
                i, j = empty_tag_pos[idx]
                offset = segments[i][j]
                label = max(predictions, key=lambda item: item["score"])["label"]
                offset["text"] = f"<{label}>"

        return segments

    def insert_empty_tags(
        self,
        segments: List[List[Dict[str, Union[str, float]]]],
        minimum_empty_duration: float,
        empty_tag: str = "<EMPTY>",
        **kwargs,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Inserts special `<EMPTY>` tag to mark for noise classification.
        Inserts tags at indices in segments where empty duration
        is at least `minimum_empty_duration`.

        Args:
            segments (List[List[Dict[str, Union[str, float]]]]):
                List of chunked segments to insert into.
            minimum_empty_duration (float):
                Minimum silence duration in seconds.
            empty_tag (str, optional):
                Special empty tag.
                Defaults to "<EMPTY>".

        Returns:
            List[List[Dict[str, Union[str, float]]]]:
                Updated segments where empty tags have been inserted.
        """
        for segment in segments:
            gaps = [
                round(next["start_time"] - curr["end_time"], 3)
                for curr, next in zip(segment, segment[1:])
            ]

            for idx, gap in reversed(list(enumerate(gaps))):
                if gap >= minimum_empty_duration:
                    start_time = segment[idx]["end_time"]
                    end_time = segment[idx + 1]["start_time"]
                    empty_offset = {
                        "text": empty_tag,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                    segment.insert(idx + 1, empty_offset)
        return segments

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
