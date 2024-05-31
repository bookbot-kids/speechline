from datasets import load_dataset

from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T


from scipy.io.wavfile import write
from tqdm.auto import tqdm
from utils import preprocess_text, compute_alignments

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
    required=True,
    help="Huggingface dataset name",
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

parser.add_argument("--output_dir", default="./tmp", help="Path to the output directory")
parser.add_argument("--chunk_size_s", type=int, default=15, help="Chunk size in seconds")
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


def get_word_alignment(datum, output_dir, chunk_size_s, text_column: str):
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

    # perform forced-alignment
    try:
        word_spans = compute_alignments(emission, words, DICTIONARY, device)
    except:
        print(f"Failed on audio: {audio_id}")
        return
        
    assert len(word_spans) == len(words)

    # collect verse-level segments
    segments, labels, start = [], [], 0
    for word, span in zip(words, word_spans):
        ratio = resampled_waveform.size(1) / num_frames
        x0 = int(ratio * span[0].start)
        x1 = int(ratio * span[-1].end)
        segment = resampled_waveform[:, x0:x1]
        segments.append(segment)
        labels.append(word)

    for segment, label in zip(segments, labels):
        audio_name = audio_id + "-" + label
        # write audio
        audio_path = (output_dir / audio_name).with_suffix(".wav")
        write(audio_path, bundle.sample_rate, segment.squeeze().numpy())

        # write transcript
        transcript_path = (output_dir / audio_name).with_suffix(".txt")
        with open(transcript_path, "w") as f:
            f.write(label)


if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, num_proc=os.cpu_count())

    if args.limit:
        dataset = dataset.select(range(args.limit))

    for datum in tqdm(dataset):
        get_word_alignment(datum, output_dir, args.chunk_size_s, args.text_column)
