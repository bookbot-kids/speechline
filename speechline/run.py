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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm

from speechline.classifiers import Wav2Vec2Classifier
from speechline.config import Config
from speechline.segmenters import SilenceSegmenter, WordOverlapSegmenter
from speechline.transcribers import Wav2Vec2Transcriber, WhisperTranscriber
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe
from speechline.utils.io import export_transcripts_json
from speechline.utils.logger import logger
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
            "-i",
            "--input_dir",
            type=str,
            required=True,
            help="Directory of input audios.",
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
    def run(config: Config, input_dir: str, output_dir: str) -> None:
        """
        Runs end-to-end SpeechLine pipeline.

        ### Pipeline Overview
        - Classifies for children's speech audio (optional).
        - Transcribes audio.
        - Segments audio into chunks based on silences.

        Args:
            config (Config):
                SpeechLine Config object.
            input_dir (str):
                Path to input directory.
            output_dir (str):
                Path to output directory.
        """
        logger.info("Preparing DataFrame..")
        df = prepare_dataframe(input_dir, audio_extension="wav")

        if config.filter_empty_transcript:
            df = df[df["ground_truth"] != ""]

        if config.do_classify:
            # load classifier model
            classifier = Wav2Vec2Classifier(
                config.classifier.model,
                max_duration_s=config.classifier.max_duration_s,
            )

            # perform audio classification
            dataset = format_audio_dataset(df, sampling_rate=classifier.sampling_rate)
            df["category"] = classifier.predict(dataset)

            # filter audio by category
            df = df[df["category"] == "child"]

        # load transcriber model
        if config.transcriber.type == "wav2vec2":
            transcriber = Wav2Vec2Transcriber(config.transcriber.model)
        elif config.transcriber.type == "whisper":
            transcriber = WhisperTranscriber(config.transcriber.model)

        # perform audio transcription
        dataset = format_audio_dataset(df, sampling_rate=transcriber.sampling_rate)

        output_offsets = transcriber.predict(
            dataset,
            chunk_length_s=config.transcriber.chunk_length_s,
            output_offsets=True,
            return_timestamps=config.transcriber.return_timestamps,
        )

        # segment audios based on offsets
        if config.segmenter.type == "silence":
            segmenter = SilenceSegmenter()
        elif config.segmenter.type == "word_overlap":
            segmenter = WordOverlapSegmenter()

        tokenizer = WordTokenizer()

        for audio_path, ground_truth, offsets in tqdm(
            zip(df["audio"], df["ground_truth"], output_offsets),
            desc="Segmenting Audio into Chunks",
            total=len(df),
        ):
            json_path = Path(audio_path).with_suffix(".json")
            # export JSON transcripts
            export_transcripts_json(str(json_path), offsets)
            # chunk audio into segments
            segmenter.chunk_audio_segments(
                audio_path,
                output_dir,
                offsets,
                minimum_chunk_duration=config.segmenter.minimum_chunk_duration,
                silence_duration=config.segmenter.silence_duration,
                ground_truth=tokenizer(ground_truth),
            )


if __name__ == "__main__":
    args = Runner.parse_args(sys.argv[1:])
    config = Config(args.config)
    Runner.run(config, args.input_dir, args.output_dir)
