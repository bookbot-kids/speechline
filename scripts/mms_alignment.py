from datasets import load_dataset

from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np


from scipy.io.wavfile import write
from tqdm.auto import tqdm
from utils import preprocess_text, compute_alignments, compute_alignment_scores

import torch
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Huggingface dataset name",
)
parser.add_argument(
    "--dataset_config",
    type=str,
    required=False,
    default=None,
    help="Huggingface dataset config",
)
parser.add_argument(
    "--dataset_split",
    type=str,
    default="train",
    help="HuggingFace dataset split name.",
)
parser.add_argument(
    "--text_column",
    type=str,
    default="sentence",
    help="Target Column",
)

parser.add_argument(
    "--chunk_size_s", type=int, default=15, help="Chunk size in seconds"
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of audio and transcript files to process",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(device)


DICTIONARY = bundle.get_dict()
#########################################################
# MMS feature extractor minimum input frame size (25ms)
# also the same value as `ratio`
# `ratio = input_waveform.size(1) / num_frames`
#########################################################
MMS_SUBSAMPLING_RATIO = 400


def get_word_alignment(datum, chunk_size_s, text_column: str):
    transcript = datum[args.text_column]
    audio = datum["audio"]

    transcript = preprocess_text(transcript)
    words = transcript.split()

    sampling_rate = audio["sampling_rate"]
    audio_array = torch.from_numpy(audio["array"])
    audio_id = Path(audio["path"]).stem

    resampler = T.Resample(sampling_rate, bundle.sample_rate, dtype=audio_array.dtype)
    resampled_waveform = resampler(audio_array)

    # split audio into chunks to avoid OOM and faster inference
    chunk_size_frames = chunk_size_s * bundle.sample_rate
    resampled_waveform = torch.unsqueeze(resampled_waveform, 0).float()
    chunks = [
        resampled_waveform[:, i : i + chunk_size_frames]
        for i in range(0, resampled_waveform.shape[1], chunk_size_frames)
    ]

    # collect per-chunk emissions, rejoin
    emissions = []
    with torch.inference_mode():
        for chunk in chunks:
            # NOTE: we could pad here, but it'll need to be removed later
            # skipping for simplicity, since it's at most 25ms
            if chunk.size(1) >= MMS_SUBSAMPLING_RATIO:
                emission, _ = model(chunk.to(device))
                emissions.append(emission)

    emission = torch.cat(emissions, dim=1)
    num_frames = emission.size(1)

    # compute greedy search score
    probs = torch.softmax(emission, dim=-1)  # (1, frame_length, num_labels)
    greedy_probs = torch.max(probs, dim=-1).values.squeeze()  # (1, frame_length)
    greedy_log_probs = torch.sum(torch.log(greedy_probs)).cpu().numpy().item()  # (1)

    aligned_probs = compute_alignment_scores(
        emission, words, DICTIONARY, device
    )  # (1, frame_length)
    aligned_log_probs = torch.sum(torch.log(aligned_probs)).cpu().numpy().item()  # (1)

    if aligned_log_probs == -np.inf:
        return False

    probability_diff = (aligned_log_probs - greedy_log_probs) / num_frames

    return probability_diff > -0.2


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset_config:
        dataset = load_dataset(
            args.dataset_name, args.dataset_config, num_proc=os.cpu_count()
        )
    else:
        dataset = load_dataset(args.dataset_name, num_proc=os.cpu_count())

    if args.limit:
        dataset = dataset.select(range(args.limit))

    # Calculate initial statistics
    initial_size = len(dataset[args.dataset_split])
    initial_hours = (
        sum(
            x["audio"]["array"].shape[0] / x["audio"]["sampling_rate"]
            for x in dataset[args.dataset_split]
        )
        / 3600
    )

    # Apply filters
    dataset = dataset.filter(
        lambda text: not text.startswith("k") and not text.startswith("m"),
        input_columns="text",
        num_proc=os.cpu_count(),
    )
    dataset = dataset.filter(
        lambda datum: get_word_alignment(datum, args.chunk_size_s, args.text_column)
    )

    # Calculate final statistics
    final_size = len(dataset[args.dataset_split])
    final_hours = (
        sum(
            x["audio"]["array"].shape[0] / x["audio"]["sampling_rate"]
            for x in dataset[args.dataset_split]
        )
        / 3600
    )

    print(f"\nDataset Statistics:")
    print(f"Initial number of rows: {initial_size:,}")
    print(f"Final number of rows: {final_size:,}")
    print(f"Rows removed: {initial_size - final_size:,}")
    print(f"Initial hours of audio: {initial_hours:.2f}")
    print(f"Final hours of audio: {final_hours:.2f}")
    print(f"Hours removed: {initial_hours - final_hours:.2f}")

    print(dataset)
    print(dataset["train"][0])
    dataset.push_to_hub(f"{args.dataset_name}-filtered", private=True)
