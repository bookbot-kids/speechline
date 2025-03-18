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
import pandas as pd
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union
from datasets import Dataset, Audio
from lexikos import Lexicon
from tqdm.contrib.concurrent import thread_map

from speechline.classifiers import Wav2Vec2Classifier
from speechline.config import Config
from speechline.segmenters import (
    PhonemeOverlapSegmenter,
    SilenceSegmenter,
    WordOverlapSegmenter,
)
from speechline.transcribers import (
    Wav2Vec2Transcriber,
    WhisperTranscriber,
    ParakeetTranscriber,
)
from speechline.utils.dataset import (
    format_audio_dataset,
    prepare_dataframe,
    prepare_dataframe_from_manifest,
)
from speechline.utils.io import export_transcripts_json
from speechline.utils.logger import Logger
from speechline.utils.tokenizer import WordTokenizer
from speechline.utils.manifest import write_manifest


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
            help="Directory of input audios or path to manifest JSON file.",
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
        parser.add_argument(
            "--script_name",
            type=str,
            help="Name of the shell script being executed",
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
            help="Directory to save log files.",
        )
        parser.add_argument(
            "--resume_from_manifest",
            type=str,
            help="Path to manifest file to resume from.",
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
                Path to input directory or manifest file if input_type is 'manifest'.
            output_dir (str):
                Path to output directory.
        """
        Logger.setup(script_name=args.script_name, log_dir=args.log_dir)
        logger = Logger.get_logger()

        # load transcriber model
        if config.transcriber.type == "wav2vec2":
            transcriber = Wav2Vec2Transcriber(config.transcriber.model)
        elif config.transcriber.type == "whisper":
            transcriber = WhisperTranscriber(config.transcriber.model)
        elif config.transcriber.type == "parakeet":
            transcriber = ParakeetTranscriber(
                config.transcriber.model, config.transcriber.transcriber_device
            )

        logger.info("Preparing DataFrame..")
        # Auto-detect input type based on path
        if os.path.isfile(input_dir) and input_dir.endswith(".json"):
            # Input is a JSON manifest file
            df = prepare_dataframe_from_manifest(input_dir)
        elif os.path.isdir(input_dir):
            # Input is a directory of audio files
            df = prepare_dataframe(input_dir, audio_extension="wav")
        else:
            logger.error(
                f"Input path {input_dir} is neither a directory nor a JSON file."
            )
            return

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

        dataset = format_audio_dataset(df, sampling_rate=transcriber.sampling_rate)

        os.makedirs(output_dir, exist_ok=True)

        # Common parameters for all transcribers
        predict_params = {
            "dataset": dataset,
            "chunk_length_s": config.transcriber.chunk_length_s,
            "output_offsets": True,
            "return_timestamps": config.transcriber.return_timestamps,
            "keep_whitespace": config.segmenter.keep_whitespace,
        }

        # Add output_dir only if the transcriber is ParakeetTranscriber
        if isinstance(transcriber, ParakeetTranscriber):
            predict_params["output_dir"] = output_dir

        output_offsets = transcriber.predict(**predict_params)

        def export_offsets(
            audio_path: str,
            offsets: List[Dict[str, Union[str, float]]],
        ):
            json_path = Path(audio_path).with_suffix(".json")
            # export JSON transcripts
            export_transcripts_json(str(json_path), offsets)

        # Create a list of (audio_path, offsets) pairs for export
        export_pairs = list(zip(df["audio"], output_offsets))

        # Filter out pairs with empty offsets
        export_pairs = [(audio, offsets) for audio, offsets in export_pairs if offsets]

        if export_pairs:
            thread_map(
                lambda pair: export_offsets(pair[0], pair[1]),
                export_pairs,
                desc="Exporting offsets to JSON",
                total=len(export_pairs),
            )
        else:
            logger.warning("No offsets to export. Skipping export step.")

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

        if config.do_noise_classify:
            noise_classifier = config.noise_classifier.model
            minimum_empty_duration = config.noise_classifier.minimum_empty_duration
            noise_classifier_threshold = config.noise_classifier.threshold
        else:
            noise_classifier = None
            minimum_empty_duration = None
            noise_classifier_threshold = None

        def segment_audio(
            audio_path: str,
            ground_truth: str,
            # We're removing the in-memory offsets parameter, but we need to keep the function signature compatible with thread_map
            _: List[Dict[str, Union[str, float]]],  # This parameter will be ignored
        ):
            # Load offsets from the JSON file instead of using in-memory offsets
            json_path = Path(audio_path).with_suffix(".json")

            # Check if the JSON file exists
            if not json_path.exists():
                logger.warning(
                    f"JSON file not found for {audio_path}. Skipping segmentation."
                )
                return [{}]

            # Load offsets from JSON file
            try:
                with open(json_path, "r") as f:
                    loaded_offsets = json.load(f)

                # Validate loaded offsets
                if not loaded_offsets:
                    logger.warning(
                        f"Empty offsets in JSON file for {audio_path}. Skipping segmentation."
                    )
                    return [{}]

                # Ensure loaded_offsets has the expected structure
                for offset in loaded_offsets:
                    if not all(
                        key in offset for key in ["text", "start_time", "end_time"]
                    ):
                        logger.warning(
                            f"Invalid offset format in JSON file for {audio_path}. Skipping segmentation."
                        )
                        return [{}]

            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON file for {audio_path}. Skipping segmentation."
                )
                return [{}]
            except Exception as e:
                logger.error(
                    f"Error loading JSON file for {audio_path}: {str(e)}. Skipping segmentation."
                )
                return [{}]

            # chunk audio into segments using loaded offsets
            segmented_manifest = segmenter.chunk_audio_segments(
                audio_path,
                output_dir,
                loaded_offsets,  # Use loaded offsets instead of in-memory offsets
                do_noise_classify=config.do_noise_classify,
                noise_classifier=noise_classifier,
                minimum_empty_duration=minimum_empty_duration,
                minimum_chunk_duration=config.segmenter.minimum_chunk_duration,
                noise_classifier_threshold=noise_classifier_threshold,
                silence_duration=config.segmenter.silence_duration,
                ground_truth=tokenizer(ground_truth),
            )
            return segmented_manifest

        # Keep the thread_map call the same to maintain compatibility
        # We're still passing output_offsets, but our segment_audio function will ignore it
        all_manifest = thread_map(
            segment_audio,
            df["audio"],
            df["ground_truth"],
            output_offsets,  # Keep this parameter to maintain compatibility with thread_map
            desc="Segmenting Audio into Chunks",
            total=len(df),
        )

        # Skip writing to manifest file if all_manifest is empty or contains only empty items
        if all_manifest and any(manifest for manifest in all_manifest):
            logger.info("Processing segmentation results for manifest creation")
            manifest_path = os.path.join(output_dir, "audio_segment_manifest.json")

            # Check if file exists before overwriting
            if os.path.exists(manifest_path):
                logger.warning(f"Overwriting existing manifest file: {manifest_path}")

            write_manifest(
                all_manifest,
                manifest_path,
                force_overwrite=True,  # Explicitly force overwrite
            )

            # Verify the file was written correctly
            if os.path.exists(manifest_path):
                logger.info(f"Manifest file exists after writing: {manifest_path}")
                try:
                    with open(manifest_path, "r") as f:
                        content = json.load(f)
                        logger.info(f"Manifest contains {len(content)} entries")
                except Exception as e:
                    logger.error(f"Error verifying manifest content: {str(e)}")
            else:
                logger.error(f"Failed to create manifest file: {manifest_path}")
        else:
            logger.warning(
                "No valid segmentation results found, skipping manifest creation"
            )


if __name__ == "__main__":
    args = Runner.parse_args(sys.argv[1:])
    config = Config(args.config)
    Runner.run(config, args.input_dir, args.output_dir)
