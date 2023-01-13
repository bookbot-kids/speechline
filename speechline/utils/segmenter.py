from typing import List, Dict, Any
from pydub import AudioSegment
import os

from speechline.utils.io import (
    get_outdir_path,
    get_chunk_path,
    export_segment_transcripts_tsv,
    export_segment_audio_wav,
)


class AudioSegmenter:
    def chunk_offsets(
        self, phoneme_offsets: List[Dict[str, Any]], silence_duration: float
    ) -> List[List[Dict[str, Any]]]:
        """Chunk transcript offsets based on in-between silence duration.

        Example:
        ```py
        >> offsets = [
        >>     {"start_time": 0.0, "end_time": 0.4},
        >>     {"start_time": 0.4, "end_time": 0.5},
        >>     {"start_time": 0.6, "end_time": 0.8},
        >>     {"start_time": 1.1, "end_time": 1.3},
        >>     {"start_time": 1.5, "end_time": 1.7},
        >>     {"start_time": 1.7, "end_time": 2.0},
        >> ]
        >> segments = self.chunk_offsets(offsets, silence_duration=0.2)
        [
            [
                {"start_time": 0.0, "end_time": 0.4},
                {"start_time": 0.4, "end_time": 0.5},
                {"start_time": 0.6, "end_time": 0.8},
            ],
            [
                {"start_time": 1.1, "end_time": 1.3}
            ],
            [
                {"start_time": 1.5, "end_time": 1.7},
                {"start_time": 1.7, "end_time": 2.0}
            ],
        ]
        ```

        Args:
            phoneme_offsets (List[Dict[str, Any]]):
                Offsets to chunk.
            silence_duration (float):
                Minimum in-between silence duration (in seconds) to consider as gaps.

        Returns:
            List[List[Dict[str, Any]]]: List of chunked/segmented offsets.
        """
        # calculate gaps in between offsets
        gaps = [
            round(next["start_time"] - curr["end_time"], 3)
            for curr, next in zip(phoneme_offsets, phoneme_offsets[1:])
        ]
        # generate segment slices (start and end indices) based on silence
        slices = (
            [0]
            + [idx + 1 for idx, gap in enumerate(gaps) if gap >= silence_duration]
            + [len(phoneme_offsets)]
        )
        # group consecutive offsets (segments) based on slices
        segments = [phoneme_offsets[i:j] for i, j in zip(slices, slices[1:])]
        return segments

    def _shift_offsets(self, offset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Shift start and end time of offsets by index start time.
        Subtracts all start and end times by index start time.

        Args:
            offset (List[Dict[str, Any]]): Offsets to shift.

        Returns:
            List[Dict[str, Any]]: Shifted offsets.
        """
        index_start = offset[0]["start_time"]
        shifted_offset = [
            {
                "phoneme": o["phoneme"],
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
        phoneme_offsets: List[Dict[str, Any]],
        silence_duration: float = 0.1,
    ) -> List[List[Dict[str, Any]]]:
        """Chunks an audio file based on its phoneme offsets.
        Generates and exports WAV audio chunks and aligned TSV phoneme transcripts.

        Args:
            audio_path (str):
                Path to audio file to chunk.
            outdir (str):
                Output directory to save chunked audio.
                Per-region subfolders will be generated under this directory.
            phoneme_offsets (List[Dict[str, Any]]):
                List of phoneme offsets.
            silence_duration (float, optional):
                Minimum in-between silence duration (in seconds) to consider as gaps.
                Defaults to 0.1 seconds.

        Returns:
            List[List[Dict[str, Any]]]: List of phoneme offsets for every segment.
        """
        segments = self.chunk_offsets(phoneme_offsets, silence_duration)
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
            # export TSV transcripts and WAV audio segment
            output_tsv_path = get_chunk_path(audio_path, outdir, idx, "tsv")
            export_segment_transcripts_tsv(output_tsv_path, segment)

            output_audio_path = get_chunk_path(audio_path, outdir, idx, "wav")
            export_segment_audio_wav(output_audio_path, audio_segment)

        return shifted_segments
