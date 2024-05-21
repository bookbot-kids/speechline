# Copyright 2024 [PT BOOKBOT INDONESIA](https://bookbot.id/)
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
import os
import sys
from typing import List

from datasets import Dataset, load_dataset
from speechline.transcribers import Wav2Vec2Transcriber


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Utility argument parser function for dataset creation.

    Args:
        args (List[str]):
            List of arguments.

    Returns:
        argparse.Namespace:
            Objects with arguments values as attributes.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/filter_dataset.py",
        description="Filter HuggingFace speech dataset by comparing with transcripts.",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset repository name.")
    parser.add_argument("--dataset_config", type=str, default="default", help="HuggingFace dataset config.")
    parser.add_argument("--dataset_split", type=str, default="train", help="HuggingFace dataset split name.")
    parser.add_argument("--label_column_name", type=str, default="phonemes_ipa", help="Label column name in dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Labeling model name from HuggingFace Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Filtered dataset HuggingFace Hub repo ID .")
    parser.add_argument("--torch_dtype", type=str, default=None, help="PyTorch data type for model.")
    parser.add_argument("--private", type=bool, default=True, help="Set HuggingFace dataset to private.")
    return parser.parse_args(args)


def filter_dataset(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    label_column_name: str,
    model_name: str,
    torch_dtype: str = None,
) -> Dataset:
    transcriber = Wav2Vec2Transcriber(model_name, torch_dtype)

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, num_proc=os.cpu_count())

    # perform transcription
    transcripts = transcriber.predict(dataset)
    assert len(transcripts) == len(dataset)
    transcript_column_name = "transcript"
    dataset = dataset.add_column(transcript_column_name, transcripts)

    # NOTE: this changes according to each dataset's schema
    # currently, all phoneme labels are e.g. ["ɡ ɛ t ɪ ŋ", "ð ə m", "f ɔ ɹ", "t w ɛ l v", "d ɑ l ɝ z", "ə", "n aɪ t"]
    # while our transcripts are e.g.
    normalize_list_labels = lambda sentence: " ".join("".join(word.split()) for word in sentence)
    label_preprocess_fn = normalize_list_labels if isinstance(dataset[0][label_column_name], list) else lambda x: x

    # filter dataset
    filtered_dataset = dataset.filter(
        lambda label, transcript: label_preprocess_fn(label) == transcript,
        input_columns=[label_column_name, transcript_column_name],
        num_proc=os.cpu_count(),
    )
    filtered_dataset = filtered_dataset.remove_columns(transcript_column_name)

    return filtered_dataset


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    filtered_dataset: Dataset = filter_dataset(
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.label_column_name,
        args.model_name,
        args.torch_dtype,
    )
    print(filtered_dataset)
    filtered_dataset.push_to_hub(args.repo_id, args.dataset_config, split=args.dataset_split, private=args.private)
