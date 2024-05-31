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

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
from tqdm.contrib.concurrent import thread_map


from datasets import Audio, load_dataset
from lexikos import Lexicon

from speechline.config import Config
from speechline.segmenters import (
    PhonemeOverlapSegmenter,
    SilenceSegmenter,
    WordOverlapSegmenter,
)
from speechline.transcribers import Wav2Vec2Transcriber, WhisperTranscriber
from speechline.utils.dataset import preprocess_audio_transcript
from speechline.utils.io import export_transcripts_json
from speechline.utils.tokenizer import WordTokenizer


@dataclass
class Runner:
    @staticmethod
    def parse_args(args: List[str]) -> argparse.Namespace:
        """
        Utility argument parser function for SpeechLine.

        Args:
            args (List[str]):
                List of arguments.

        Returns:
            argparse.Namespace:
                Objects with arguments values as attributes.
        """
        parser = argparse.ArgumentParser(
            prog="python speechline/run.py",
            description="Perform end-to-end speech labeling pipeline.",
        )

        parser.add_argument(
            "--dataset_name",
            type=str,
            required=True,
            help="HuggingFace dataset name.",
        )
        parser.add_argument(
            "--dataset_config",
            type=str,
            default="default",
            help="HuggingFace dataset config.",
        )
        parser.add_argument(
            "--dataset_split",
            type=str,
            default="train",
            help="HuggingFace dataset split name.",
        )
        parser.add_argument(
            "--audio_column_name",
            type=str,
            default="audio",
            help="HuggingFace dataset audio column name.",
        )
        parser.add_argument(
            "--text_column_name",
            type=str,
            default="text",
            help="HuggingFace dataset text column name.",
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            required=True,
            help="Directory to save pipeline results.",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="examples/config.json",
            help="SpeechLine configuration file.",
        )
        return parser.parse_args(args)

    @staticmethod
    def run(
        config: Config,
        output_dir: str,
        dataset_name: str,
        dataset_config: str = "default",
        dataset_split: str = "train",
        audio_column_name: str = "audio",
        text_column_name: str = "text",
    ) -> None:
        """
        Runs end-to-end SpeechLine pipeline.

        ### Pipeline Overview
        - Transcribes audio.
        - Segments audio into chunks based on silences.

        Args:
            config (Config):
                SpeechLine Config object.
            output_dir (str):
                Path to output directory.
            dataset_name (str):
                HuggingFace dataset name.
            dataset_config (str, optional):
                HuggingFace dataset config. Defaults to "default".
            dataset_split (str, optional):
                HuggingFace dataset split name. Defaults to "train".
            audio_column_name (str, optional):
                HuggingFace dataset audio column name. Defaults to "audio".
            text_column_name (str, optional):
                HuggingFace dataset text column name. Defaults to "text".
        """
        num_proc = os.cpu_count()
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, trust_remote_code=True, num_proc=num_proc)
        # dataset = dataset.select(range(100))

        # load transcriber model
        if config.transcriber.type == "wav2vec2":
            transcriber = Wav2Vec2Transcriber(config.transcriber.model)
        elif config.transcriber.type == "whisper":
            transcriber = WhisperTranscriber(config.transcriber.model)

        # perform audio transcription
        dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=transcriber.sampling_rate))
        dataset = dataset.map(
            lambda example: {text_column_name: preprocess_audio_transcript(example[text_column_name])},
            num_proc=num_proc,
            remove_columns=list(set(dataset.column_names) - set([audio_column_name, text_column_name])),
        )

        if config.filter_empty_transcript:
            dataset = dataset.filter(lambda example: example != "", num_proc=num_proc, input_columns=[text_column_name])

        output_offsets = transcriber.predict(
            dataset,
            chunk_length_s=config.transcriber.chunk_length_s,
            output_offsets=True,
            return_timestamps=config.transcriber.return_timestamps,
            keep_whitespace=config.segmenter.keep_whitespace,
        )

        def export_offsets(idx: int):
            datum = dataset[idx]
            json_path = Path(datum[audio_column_name]["path"]).with_suffix(".json")
            json_path = os.path.normpath(json_path).split(os.sep)[-2:]
            export_path = os.path.join(output_dir, *json_path)
            export_transcripts_json(export_path, output_offsets[idx])

        thread_map(export_offsets, range(len(dataset)), desc="Exporting offsets to JSON", total=len(dataset))

        # segment audios based on offsets
        if config.segmenter.type == "silence":
            segmenter = SilenceSegmenter()
        elif config.segmenter.type == "word_overlap":
            segmenter = WordOverlapSegmenter()
        elif config.segmenter.type == "phoneme_overlap":
            lexicon = Lexicon()
            if config.segmenter.lexicon_path:
                with open(config.segmenter.lexicon_path) as json_file:
                    lex = json.load(json_file)
                # merge dict with lexicon
                for k, v in lex.items():
                    lexicon[k] = lexicon[k].union(set(v)) if k in lexicon else set(v)
            segmenter = PhonemeOverlapSegmenter(lexicon)

        tokenizer = WordTokenizer()

        def segment_audio(idx):
            example = dataset[idx]
            offset = output_offsets[idx]
            # chunk audio into segments
            segmenter.chunk_audio_segments(
                example[audio_column_name],
                output_dir,
                offset,
                minimum_chunk_duration=config.segmenter.minimum_chunk_duration,
                silence_duration=config.segmenter.silence_duration,
                ground_truth=tokenizer(example[text_column_name]),
            )

        thread_map(segment_audio, range(len(dataset)), desc="Segmenting Audio into Chunks", total=len(dataset))


if __name__ == "__main__":
    args = Runner.parse_args(sys.argv[1:])
    config = Config(args.config)
    Runner.run(
        config,
        args.output_dir,
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.audio_column_name,
        args.text_column_name,
    )
