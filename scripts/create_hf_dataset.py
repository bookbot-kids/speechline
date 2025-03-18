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
import csv
import sys
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from speechline.utils.g2p import get_g2p
from tqdm.auto import tqdm


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
        prog="python scripts/create_hf_datasets.py",
        description="Create HuggingFace dataset from SpeechLine outputs.",
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory of input audios.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset repository name.",
    )
    parser.add_argument("--phonemize", type=bool, default=False, help="Phonemize text.")
    parser.add_argument(
        "--private", type=bool, default=True, help="Set HuggingFace dataset to private."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.0,
        help="Proportion of data for test set (0.0 to 1.0)",
    )
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.0,
        help="Proportion of data for validation set (0.0 to 1.0)",
    )
    return parser.parse_args(args)


def parse_tsv(path: str) -> str:
    """
    Join text transcripts of TSV annotation.

    Args:
        path (str):
            Path to TSV file.

    Returns:
        str:
            Joined text transcript.
    """
    with open(path) as fd:
        rows = csv.reader(fd, delimiter="\t", quotechar='"')
        return " ".join(row[2] for row in rows)


def create_dataset(
    input_dir: str,
    dataset_name: str,
    private: bool = True,
    phonemize: bool = False,
    test_size: float = 0.0,
    valid_size: float = 0.0,
) -> DatasetDict:
    """
    Creates HuggingFace dataset from SpeechLine outputs.
    Ensures unique utterance and speaker IDs in each subset.

    Args:
        input_dir (str):
            Path to input audio directory.
        dataset_name (str):
            HuggingFace dataset name.
        private (bool, optional):
            Set HuggingFace dataset as private. Defaults to `True`.
        phonemize (bool, optional):
            Phonemize text to phoneme strings. Defaults to `False`.
        test_size (float, optional):
            Proportion of data for test set. Defaults to 0.0.
        valid_size (float, optional):
            Proportion of data for validation set. Defaults to 0.0.

    Returns:
        DatasetDict:
            Created HuggingFace dataset.
    """
    audios = glob(f"{input_dir}/**/*.wav")
    df = pd.DataFrame({"audio": audios})
    # `audio` =  `"{dir}/{language}/{speaker}_{utt_id}.wav"`
    df["id"] = df["audio"].apply(lambda x: x.split("/")[-1].replace(".wav", ""))
    df["language"] = df["audio"].apply(lambda x: x.split("/")[-2])
    df["speaker"] = df["audio"].apply(lambda x: x.split("/")[-1].split("_")[0])
    df["text"] = df["audio"].apply(lambda x: parse_tsv(Path(x).with_suffix(".tsv")))

    tqdm.pandas(desc="Phonemization")

    if phonemize:
        df["phonemes"] = df.progress_apply(
            lambda row: get_g2p(row["language"].split("-")[0])(row["text"]), axis=1
        )

    speaker, counts = np.unique(df["speaker"], return_counts=True)
    speaker2count = {s: c for s, c in zip(speaker, counts)}

    total_samples = len(df)
    train_num = int((1 - test_size - valid_size) * total_samples)
    test_num = int((1 - valid_size) * total_samples)

    train_speakers, test_speakers, valid_speakers = [], [], []
    total = 0

    for speaker, count in sorted(
        speaker2count.items(), key=lambda item: item[1], reverse=True
    ):
        if total < train_num:
            train_speakers.append(speaker)
        elif total < test_num:
            test_speakers.append(speaker)
        else:
            valid_speakers.append(speaker)
        total += count

    train_df = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
    test_df = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)
    valid_df = df[df["speaker"].isin(valid_speakers)].reset_index(drop=True)

    dataset_dict = {
        "train": Dataset.from_pandas(train_df).cast_column("audio", Audio())
    }

    if test_size > 0:
        dataset_dict["test"] = Dataset.from_pandas(test_df).cast_column(
            "audio", Audio()
        )
    if valid_size > 0:
        dataset_dict["validation"] = Dataset.from_pandas(valid_df).cast_column(
            "audio", Audio()
        )

    dataset = DatasetDict(dataset_dict)
    dataset.push_to_hub(dataset_name, private=private)
    return dataset


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    dataset = create_dataset(
        args.input_dir,
        args.dataset_name,
        args.private,
        args.phonemize,
        args.test_size,
        args.valid_size,
    )
