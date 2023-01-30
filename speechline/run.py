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
from pathlib import Path
from typing import List

from tqdm import tqdm

from speechline.classifiers import Wav2Vec2Classifier
from speechline.transcribers import Wav2Vec2Transcriber
from speechline.utils.config import Config
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe
from speechline.utils.io import export_transcripts_json
from speechline.utils.logger import logger
from speechline.utils.segmenter import AudioSegmenter


class Runner:
    """
    SpeechLine Runnner.

    Args:
        config (Config):
            SpeechLine Config object.
        input_dir (str):
            Path to input directory.
        output_dir (str):
            Path to output directory.
    """

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

    def __init__(self, config: Config, input_dir: str, output_dir: str) -> None:
        self.config = config
        self.languages = self.config.languages
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        """
        Runs end-to-end SpeechLine pipeline.

        ### Pipeline Overview
        - Prepare DataFrame of audio data.
        - For every specified language:
            - Filters dataset based on language.
            - Classifies for children's speech audio.
            - Transcribes children's speech audio.
            - Segments audio into chunks based on silences.
        """
        logger.info("Preparing DataFrame..")
        raw_df = prepare_dataframe(self.input_dir, audio_extension="wav")

        for language in self.languages:
            # filter dataframe by language
            df = raw_df[raw_df["language"] == language]
            if len(df) == 0:
                logger.info(f"DataFrame for language {language} is empty. Skipping..")
                continue

            # load classifier model
            classifier_checkpoint = self.config.classifier["models"][language]
            classifier = Wav2Vec2Classifier(
                classifier_checkpoint,
                max_duration_s=self.config.classifier["max_duration_s"],
            )

            # perform audio classification
            # TODO: add minimum length filter for super-short audio?
            dataset = format_audio_dataset(df, sampling_rate=classifier.sampling_rate)
            df["category"] = classifier.predict(
                dataset, batch_size=self.config.classifier["batch_size"]
            )

            # filter audio by category
            child_speech_df = df[df["category"] == "child"]

            # load transcriber model
            transcriber_checkpoint = self.config.transcriber["models"][language]
            transcriber = Wav2Vec2Transcriber(transcriber_checkpoint)

            # perform audio transcription
            dataset = format_audio_dataset(
                child_speech_df, sampling_rate=transcriber.sampling_rate
            )
            phoneme_offsets = transcriber.predict(
                dataset,
                output_offsets=True,
            )

            # segment audios based on offsets
            segmenter = AudioSegmenter()
            for audio_path, offsets in tqdm(
                zip(child_speech_df["audio"], phoneme_offsets),
                desc="Segmenting Audio into Chunks",
                total=len(child_speech_df),
            ):
                json_path = Path(audio_path).with_suffix(".json")
                # export JSON transcripts
                export_transcripts_json(str(json_path), offsets)
                # chunk audio into segments
                segmenter.chunk_audio_segments(
                    audio_path,
                    self.output_dir,
                    offsets,
                    silence_duration=self.config.segmenter["silence_duration"],
                    minimum_chunk_duration=self.config.segmenter[
                        "minimum_chunk_duration"
                    ],
                )


if __name__ == "__main__":
    args = Runner.parse_args(sys.argv[1:])
    config = Config(args.config)
    runner = Runner(config, args.input_dir, args.output_dir)
    runner.run()
