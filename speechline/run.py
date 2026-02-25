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
    ParakeetTDTTranscriber,
    CanaryTranscriber,
    GentleTranscriber,
)
from speechline.utils.dataset import (
    format_audio_dataset,
    prepare_dataframe,
    prepare_dataframe_from_manifest,
)
from speechline.utils.io import export_transcripts_json, export_transcripts_txt
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
            required=False,
            default=None,
            help="Directory to save pipeline results. Defaults to input_dir.",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="examples/config.json",
            help="SpeechLine configuration file.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Number of files to process in each batch (for memory efficiency with large directories).",
        )
        parser.add_argument(
            "--max_files",
            type=int,
            default=None,
            help="Maximum number of files to process (useful for testing on large directories).",
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
    def run(config: Config, input_dir: str, output_dir: str = None) -> None:
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
        # Default output_dir to input_dir if not specified
        if output_dir is None:
            output_dir = input_dir
            
        args = getattr(Runner, '_args', None)
        Logger.setup(script_name=getattr(args, 'script_name', None) if args else None,
                     log_dir=getattr(args, 'log_dir', 'logs') if args else 'logs')
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
        elif config.transcriber.type == "parakeet_tdt":
            # Get device and torch_dtype from config if available
            transcriber_device = getattr(config.transcriber, 'transcriber_device', None)
            torch_dtype = getattr(config.transcriber, 'torch_dtype', None)
            transcriber = ParakeetTDTTranscriber(
                model_checkpoint=config.transcriber.model,
                transcriber_device=transcriber_device,
                torch_dtype=torch_dtype
            )
        elif config.transcriber.type == "canary":
            # Get torch_dtype from config if available, default to "float16"
            torch_dtype = getattr(config.transcriber, 'torch_dtype', 'float16')
            transcriber = CanaryTranscriber(
                model_checkpoint=config.transcriber.model,
                torch_dtype=torch_dtype
            )
        elif config.transcriber.type == "gentle":
            # Get Gentle-specific parameters from config
            gentle_path = getattr(config.transcriber, 'gentle_path', '/mnt/Projects/Projects/AudioProcessing/gentle')
            output_phonemes = getattr(config.transcriber, 'output_phonemes', True)
            output_word_boundaries = getattr(config.transcriber, 'output_word_boundaries', True)
            transcriber = GentleTranscriber(
                gentle_path=gentle_path,
                output_phonemes=output_phonemes,
                output_word_boundaries=output_word_boundaries
            )

        logger.info("Preparing DataFrame..")
        
        # Check if validation mode is enabled - if so, don't filter empty ground truth
        # We'll transcribe first and use that as ground truth for validation
        is_validation_mode = (
            config.transcriber.type == "parakeet_tdt" and
            hasattr(config.transcriber, 'validate_alignment') and
            config.transcriber.validate_alignment
        )
        
        # Auto-detect input type based on path
        if os.path.isfile(input_dir) and input_dir.endswith(".json"):
            # Input is a JSON manifest file
            df = prepare_dataframe_from_manifest(input_dir)
        elif os.path.isdir(input_dir):
            # Input is a directory of audio files
            # In validation mode, don't filter empty transcripts - we'll generate them
            # For Gentle transcriber, always require ground truth (.txt files)
            if config.transcriber.type == "gentle":
                filter_empty = True  # Gentle requires existing .txt files
            else:
                filter_empty = False if is_validation_mode else config.filter_empty_transcript
            
            df = prepare_dataframe(
                input_dir,
                audio_extension=config.audio_extension,
                filter_empty=filter_empty,
                max_files=getattr(args, 'max_files', None) if args else None,
                folder_filter=getattr(config, 'folder_filter', None)
            )
        else:
            logger.error(
                f"Input path {input_dir} is neither a directory nor a JSON file."
            )
            return
        
        logger.info(f"📊 DataFrame prepared: {len(df)} files to process")
        
        # Apply batch_size if specified
        if args and args.batch_size and len(df) > args.batch_size:
            logger.warning(
                f"⚠️  Large dataset detected: {len(df)} files. "
                f"Processing in batches of {args.batch_size} for memory efficiency."
            )
            logger.warning(
                f"⚠️  Note: Batch processing only works with validation mode disabled. "
                f"For validation mode, use --max_files to limit dataset size."
            )

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

        logger.info(f"🔄 Creating dataset (this may take time for large directories)...")
        # Gentle uses fixed 8kHz sampling rate
        sampling_rate = transcriber.sampling_rate if hasattr(transcriber, 'sampling_rate') else 16000
        dataset = format_audio_dataset(df, sampling_rate=sampling_rate)
        logger.info(f"✅ Dataset created successfully")

        os.makedirs(output_dir, exist_ok=True)

        # Check if validation mode is enabled for Parakeet TDT
        if (config.transcriber.type == "parakeet_tdt" and
            hasattr(config.transcriber, 'validate_alignment') and
            config.transcriber.validate_alignment):
            
            # Validation mode: use ground truth for alignment validation
            logger.info("Running in alignment validation mode...")
            
            # MEMORY OPTIMIZATION: Process in batches to avoid OOM
            validation_batch_size = getattr(args, 'batch_size', None) if args else None
            validation_batch_size = validation_batch_size or 100
            logger.info(f"Processing validation in batches of {validation_batch_size} files to manage memory")
            
            # Extract ground truth texts from dataframe
            ground_truth_texts = df["ground_truth"].tolist()
            
            # Process in batches
            all_validation_results = []
            total_files = len(df)
            
            for batch_start in range(0, total_files, validation_batch_size):
                batch_end = min(batch_start + validation_batch_size, total_files)
                batch_indices = list(range(batch_start, batch_end))
                
                logger.info(f"Processing batch {batch_start//validation_batch_size + 1}/{(total_files + validation_batch_size - 1)//validation_batch_size} (files {batch_start}-{batch_end-1})")
                
                # Get batch data
                batch_df = df.iloc[batch_indices]
                batch_dataset = dataset.select(batch_indices)
                batch_ground_truth = ground_truth_texts[batch_start:batch_end]
                
                # Identify files without ground truth in this batch
                files_without_gt = [i for i, gt in enumerate(batch_ground_truth) if not gt.strip()]
                
                if files_without_gt:
                    logger.info(f"  Found {len(files_without_gt)} files without ground truth in this batch. Transcribing first...")
                    
                    # Transcribe files without ground truth
                    predict_params = {
                        "dataset": batch_dataset.select(files_without_gt),
                        "chunk_length_s": config.transcriber.chunk_length_s,
                        "output_offsets": True,
                        "return_timestamps": config.transcriber.return_timestamps,
                        "keep_whitespace": config.segmenter.keep_whitespace,
                    }
                    
                    transcribed_offsets = transcriber.predict(**predict_params)
                    
                    # Extract transcribed text and update batch_ground_truth
                    for idx, offsets in zip(files_without_gt, transcribed_offsets):
                        if offsets:
                            # Concatenate all tokens to form the transcription
                            transcription = " ".join(token["text"] for token in offsets)
                            batch_ground_truth[idx] = transcription
                            
                            # Also save the transcription as .txt file
                            audio_path = batch_df["audio"].iloc[idx]
                            txt_path = Path(audio_path).with_suffix(".txt")
                            with open(txt_path, 'w') as f:
                                f.write(transcription)
                            logger.info(f"  Generated ground truth for {Path(audio_path).name}: {transcription}")
                        else:
                            logger.warning(f"  Failed to transcribe {batch_df['audio'].iloc[idx]}")
                            batch_ground_truth[idx] = ""
                
                # Run validation on this batch
                logger.info(f"  Running validation on batch...")
                batch_validation_results = transcriber.predict_with_validation(
                    dataset=batch_dataset,
                    ground_truth_texts=batch_ground_truth,
                    nfa_model=getattr(config.transcriber, 'nfa_model', 'nvidia/parakeet-ctc-1.1b'),
                    token_confidence_threshold=getattr(config.transcriber, 'token_confidence_threshold', 0.7),
                    min_alignment_ratio=getattr(config.transcriber, 'min_alignment_ratio', 0.8),
                    output_dir=input_dir,  # Use input_dir for temporary files
                    chunk_length_s=config.transcriber.chunk_length_s
                )
                
                all_validation_results.extend(batch_validation_results)
                
                logger.info(f"  Batch complete. {len([r for r in batch_validation_results if r['is_valid']])} accepted, {len([r for r in batch_validation_results if not r['is_valid']])} rejected")
                
                # Export validation results for this batch immediately
                logger.info(f"  Saving validation results for batch...")
                for idx, result in enumerate(batch_validation_results):
                    # Get the original audio path from batch_df
                    audio_path = batch_df["audio"].iloc[idx]
                    
                    # Create simplified validation JSON (only keep essential fields)
                    simplified_result = {
                        "ground_truth": result["ground_truth"],
                        "transcription": result["transcription"],
                        "alignment_ratio": result["alignment_ratio"],
                        "avg_confidence": result["avg_confidence"],
                        "token_alignment": result["token_alignment"],
                        "word_confidence": result.get("word_confidence", [])  # Include word-level confidence scores
                    }
                    
                    # Write JSON file next to audio file (same directory, same name)
                    validation_json_path = Path(audio_path).with_suffix(".json")
                    
                    with open(validation_json_path, 'w') as f:
                        json.dump(simplified_result, f, indent=2)
                
                logger.info(f"  ✅ Batch validation results saved ({len(batch_validation_results)} files)")
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
            
            # Use combined results
            validation_results = all_validation_results
            
            # Filter to only valid audio files
            valid_indices = [i for i, r in enumerate(validation_results) if r["is_valid"]]
            rejected_indices = [i for i, r in enumerate(validation_results) if not r["is_valid"]]
            
            logger.info(f"Validation results: {len(valid_indices)} accepted, {len(rejected_indices)} rejected")
            logger.info(f"All validation JSON files have been saved next to their respective audio files")
            
            logger.info("Validation mode complete. All results saved next to audio files.")
            
            # Skip the rest of the pipeline in validation mode
            return
        else:
            # Normal mode: standard transcription without validation
            
            # Gentle transcriber has different interface
            if isinstance(transcriber, GentleTranscriber):
                # Gentle only needs dataset and output_dir
                output_offsets = transcriber.predict(
                    dataset=dataset,
                    output_dir=output_dir
                )
            else:
                # Common parameters for all other transcribers
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
            # Write TXT file next to the audio file (same directory)
            txt_path = Path(audio_path).with_suffix(".txt")
            export_transcripts_txt(str(txt_path), offsets)

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
            offsets: List[Dict[str, Union[str, float]]],
        ):
            # Use in-memory offsets directly instead of loading from file
            if not offsets:
                logger.warning(
                    f"Empty offsets for {audio_path}. Skipping segmentation."
                )
                return [{}]

            # chunk audio into segments using offsets
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
    
    # Store args in a module-level variable so Runner.run can access them
    Runner._args = args
    
    Runner.run(config, args.input_dir, args.output_dir)
