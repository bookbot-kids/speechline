import json
import os

import subprocess
import tempfile
from typing import Dict, List, Union

from datasets import Dataset, Audio

from ..utils.io import export_transcripts_json
from tqdm import tqdm
from ..utils.logger import Logger

logger = Logger.get_logger()


class ParakeetTranscriber:
    def __init__(self, model_name: str, transcribe_device: str = "cuda") -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.model_name = model_name
        self.transcribe_device = transcribe_device
        self.sampling_rate = 16000
        self.batch_size = 1

    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 0,
        output_offsets: bool = True,
        return_timestamps: bool = True,
        keep_whitespace: bool = False,
        segment_with_ground_truth: bool = False,
        output_dir: str = None,
        **kwargs,
    ) -> Union[List[List[Dict[str, Union[str, float]]]], List[str]]:
        manifest_path = os.path.join(output_dir, "manifest.json")
        output_dir = os.path.join(output_dir, "alignment_output")

        # Get all audio paths from the dataset and extract filenames
        dataset_filenames = set()
        filename_to_path = {}

        for item in dataset:
            path = item["audio"]["path"]
            filename = os.path.basename(path)
            dataset_filenames.add(filename)
            filename_to_path[filename] = path

        logger.info(
            f"Received dataset with {len(dataset_filenames)} audio files to process"
        )

        # Create manifest file with only the requested audio files
        logger.info("Creating manifest file for current dataset...")

        with open(manifest_path, "w") as f:
            if segment_with_ground_truth:
                for item in tqdm(dataset, desc="Writing ground truth manifest"):
                    json.dump(
                        {
                            "audio_filepath": item["audio"]["path"],
                            "text": item["ground_truth"],
                        },
                        f,
                    )
                    f.write("\n")
            else:
                for item in tqdm(dataset, desc="Writing audio manifest"):
                    json.dump({"audio_filepath": item["audio"]["path"]}, f)
                    f.write("\n")

        # Run Parakeet aligner
        cmd = [
            "python",
            "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/NeMo/tools/nemo_forced_aligner/align.py",
            f"pretrained_name={self.model_name}",
            f"manifest_filepath={manifest_path}",
            f"output_dir={output_dir}",
            f"transcribe_device={self.transcribe_device}",
            f"batch_size={self.batch_size}",
        ]

        if not segment_with_ground_truth:
            cmd.append("align_using_pred_text=true")

        # Only add chunk length parameter if it's greater than 0
        if chunk_length_s is not None and chunk_length_s > 0:
            cmd.append(f"chunk_len_in_secs={chunk_length_s}")

        logger.info("\nRunning Parakeet aligner...")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            logger.info(f"Error: {e}")
            error_msg = (
                str(e.output)
                if isinstance(e, subprocess.CalledProcessError)
                else str(e)
            )
            if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                logger.warning(
                    "CUDA out of memory error encountered. Falling back to CPU processing..."
                )
                # Modify command to use CPU instead
                cmd = [
                    arg.replace(
                        f"transcribe_device={self.transcribe_device}",
                        "transcribe_device=cpu",
                    )
                    for arg in cmd
                ]
                # remove batch size arg
                cmd = [arg for arg in cmd if not arg.startswith("batch_size=")]
                logger.info("Retrying alignment with CPU...")
                subprocess.run(cmd, check=True)
            else:
                # Re-raise other types of errors
                raise
        # Read manifest with output paths
        manifest_with_paths = os.path.join(
            output_dir,
            os.path.basename(manifest_path).replace(
                ".json", "_with_output_file_paths.json"
            ),
        )

        # Read alignment results and format them as a dictionary mapping audio paths to offsets
        audio_to_offsets = {}

        # Skip processing if manifest doesn't exist
        if not os.path.exists(manifest_with_paths):
            logger.warning(f"Output manifest file {manifest_with_paths} not found.")
            # Return empty offsets for all files
            return [[] for _ in dataset]

        with open(manifest_with_paths) as f:
            lines = list(f)
            for line in tqdm(lines, desc="Processing alignments"):
                entry = json.loads(line)

                audio_path = entry["audio_filepath"]
                audio_filename = os.path.basename(audio_path)

                # Check if this file is in our dataset using filename
                if audio_filename not in dataset_filenames:
                    logger.debug(
                        f"Skipping {audio_filename} as it's not in the current dataset"
                    )
                    continue

                # Use the full path from our dataset for consistency
                original_path = filename_to_path[audio_filename]

                if "words_level_ctm_filepath" not in entry:
                    audio_to_offsets[original_path] = []
                    continue

                # Read word-level CTM file
                offsets = []
                with open(entry["words_level_ctm_filepath"]) as ctm_file:
                    for line in ctm_file:
                        # Format: <utt_id> 1 <start_time> <duration> <word>
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Ensure we have all required field
                            start_time = float(parts[2])
                            duration = float(parts[3])
                            word = parts[4]

                            if not word.strip() and not keep_whitespace:
                                continue

                            offsets.append(
                                {
                                    "text": word,
                                    "start_time": start_time,
                                    "end_time": start_time + duration,
                                }
                            )

                audio_to_offsets[original_path] = offsets

        logger.info(
            f"Total processed audio files for this dataset: {len(audio_to_offsets)}"
        )

        # Return offsets in the same order as the input dataset
        all_offsets = []
        offsets_dict = {}
        for item in dataset:
            audio_path = item["audio"]["path"]
            if audio_path in audio_to_offsets:
                all_offsets.append(audio_to_offsets[audio_path])
                offsets_dict[audio_path] = audio_to_offsets[audio_path]
            else:
                logger.warning(f"No alignment found for {audio_path}")
                all_offsets.append([])
        logger.info(f"Offsets dict: {offsets_dict}")

        return all_offsets


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test ParakeetTranscriber")
    parser.add_argument(
        "--audio_path", type=str, required=True, help="Path to audio file"
    )
    parser.add_argument(
        "--model", type=str, default="nvidia/parakeet-ctc-1.1b", help="Model name"
    )
    args = parser.parse_args()

    # Create a simple dataset with one audio file
    dataset = Dataset.from_dict(
        {
            "audio": [args.audio_path],
        }
    ).cast_column("audio", Audio())

    # Initialize transcriber and run prediction
    transcriber = ParakeetTranscriber(args.model, transcribe_device="cpu")
    results = transcriber.predict(
        dataset=dataset, chunk_length_s=30, output_offsets=True, return_timestamps=True
    )

    # Print results
    print("\nTranscription Results:")
    print("-" * 50)
    for word_data in results[0]:  # First (and only) audio file 
        print(
            f"Word: {word_data['text']:<20} "
            f"Start: {word_data['start_time']:.2f}s "
            f"End: {word_data['end_time']:.2f}s"
        )
