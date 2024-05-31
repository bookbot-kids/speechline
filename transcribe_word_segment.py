from speechline.transcribers import Wav2Vec2Transcriber
from datasets import Dataset, Audio
from pathlib import Path
from itertools import islice
import json
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    required=True,
    help="Path to audio and word alignment directory",
)
parser.add_argument(
    "--transcriber_model",
    type=str,
    required=True,
    help="Name of transcriber model",
)
parser.add_argument(
    "--target_path",
    type=str,
    default="./word_segment.json",
    help="Target directory to store the word segment.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of audio and transcript files to process",
)
    

if __name__ == "__main__":
    args = parser.parse_args()
    transcriber = Wav2Vec2Transcriber(args.transcriber_model)
    source_dir = Path(args.dataset_dir)
    
    transcript, audio = [], []
    if args.limit:
        files = islice(source_dir.rglob("*.mp3"), args.limit)
    else:
        files = source_dir.rglob("*.mp3")
    # Iterate over all files in source_dir
    for file in files:
        # Check if both transcript and audio files exist
        transcript_path = source_dir / f"{Path(file).stem}.txt"
        audio_path = source_dir / f"{Path(file).stem}.mp3"  # replace .wav with your audio file extension

        if transcript_path.exists() and audio_path.exists():
            transcript.append(transcript_path.read_text())
            audio.append(audio_path)
            
    assert len(transcript) == len(audio)
        
    dataset = Dataset.from_dict({"audio": [str(a) for a in audio], "transcript": transcript}).cast_column(
        "audio", Audio(sampling_rate=transcriber.sampling_rate)
    )
    dataset = dataset.filter(lambda x: "array" in x["audio"] and len(x["audio"]["array"]) > 1600, num_proc=10)
    print(f"Dataset length after filtering: {len(dataset)}")
    offsets = transcriber.predict(dataset, output_offsets=True, return_timestamps="char")
    print(f"Offsets length: {len(offsets)}")
    phoneme_transcript = [" ".join([o["text"] for o in offset]) if offset else "" for offset in offsets]

    word_dictionary = {}
    for word, phoneme in zip(dataset["transcript"], phoneme_transcript):
        word_dictionary[word] = word_dictionary.get(word, {})
        word_dictionary[word][phoneme] = word_dictionary[word].get(phoneme, 0) + 1

    with open(args.target_path, "w") as f:
        json.dump(word_dictionary, f, ensure_ascii=False)