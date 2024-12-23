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
from speechline.transcribers import Wav2Vec2Transcriber, WhisperTranscriber, ParakeetTranscriber
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe
from speechline.utils.io import export_transcripts_json
from speechline.utils.logger import Logger
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
        parser.add_argument(
            "--script_name",
            type=str,
            help="Name of the shell script being executed",
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
                Path to input directory.
            output_dir (str):
                Path to output directory.
        """
        Logger.setup(script_name=args.script_name)
        logger = Logger.get_logger()
        
        
        # load transcriber model
        if config.transcriber.type == "wav2vec2":
            transcriber = Wav2Vec2Transcriber(config.transcriber.model)
        elif config.transcriber.type == "whisper":
            transcriber = WhisperTranscriber(config.transcriber.model)
        elif config.transcriber.type == "parakeet":
            transcriber = ParakeetTranscriber(config.transcriber.model, config.transcriber.transcriber_device)
        
        if not args.resume_from_manifest:
            logger.info("Preparing DataFrame..")
            df = prepare_dataframe(input_dir, audio_extension="wav")

            if config.filter_empty_transcript:
                df = df[df["ground_truth"] != ""]
                
        else:
            all_new_rows = json.load(open(args.resume_from_manifest, "r"))
            
            # Create DataFrame directly from the list of dictionaries
            df = pd.DataFrame(all_new_rows)
            
            df.rename(columns={"text": "ground_truth"}, inplace=True)
            df["audio"] = df["audio"].apply(lambda f: os.path.abspath(f))
            df["language_code"] = df["language"]
            df["language"] = df["language"].apply(lambda x: x.split("-")[0])
            df["id"] = df["audio"].apply(lambda f: Path(f).stem)

            # Ensure the DataFrame has the same columns
            df = df[["audio", "id", "language", "language_code", "ground_truth"]]
            
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
        output_offsets = transcriber.predict(
            dataset,
            chunk_length_s=config.transcriber.chunk_length_s,
            output_offsets=True,
            return_timestamps=config.transcriber.return_timestamps,
            keep_whitespace=config.segmenter.keep_whitespace,
            segment_with_ground_truth=config.segmenter.segment_with_ground_truth,
            output_dir=output_dir,
        )

        def export_offsets(
            audio_path: str,
            offsets: List[Dict[str, Union[str, float]]],
        ):
            json_path = Path(audio_path).with_suffix(".json")
            # export JSON transcripts
            export_transcripts_json(str(json_path), offsets)

        thread_map(
            export_offsets,
            df["audio"],
            output_offsets,
            desc="Exporting offsets to JSON",
            total=len(df),
        )

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
            offsets: List[Dict[str, Union[str, float]]],
        ):
            # chunk audio into segments
            segmented_manifest = segmenter.chunk_audio_segments(
                audio_path,
                output_dir,
                offsets,
                do_noise_classify=config.do_noise_classify,
                noise_classifier=noise_classifier,
                minimum_empty_duration=minimum_empty_duration,
                minimum_chunk_duration=config.segmenter.minimum_chunk_duration,
                noise_classifier_threshold=noise_classifier_threshold,
                silence_duration=config.segmenter.silence_duration,
                ground_truth=tokenizer(ground_truth),
            )
                 
        thread_map(
            segment_audio,
            df["audio"],
            df["ground_truth"],
            output_offsets,
            desc="Segmenting Audio into Chunks",
            total=len(df),
        )



if __name__ == "__main__":
    args = Runner.parse_args(sys.argv[1:])
    config = Config(args.config)
    Runner.run(config, args.input_dir, args.output_dir)

    Runner.run(config, args.input_dir, args.output_dir)
