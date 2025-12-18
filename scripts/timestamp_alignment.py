import json
import re
import os
import string

from unidecode import unidecode
import unicodedata

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from datasets import load_dataset
from transformers import pipeline
from tqdm.auto import tqdm

bundle = torchaudio.pipelines.MMS_FA
device = "cuda"
model = bundle.get_model().to(device)
DICTIONARY = bundle.get_dict()
chunk_size_s = 15
MMS_SUBSAMPLING_RATIO = 400
THRESHOLD = 0.06 # 50ms

pipe = pipeline(
    "automatic-speech-recognition",
    model="bookbot/w2v-bert-2.0-libriphone",
    device="cuda",
    return_timestamps="char",
    chunk_length_s=chunk_size_s,
)

def preprocess_verse(text: str) -> str:
    text = unidecode(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # text = re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang="sw"), text)
    text = re.sub("\s+", " ", text)
    return text

def align(emission, tokens, device):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

def compute_alignments(emission, transcript, dictionary, device):
    tokens = [dictionary[char] for word in transcript for char in word]
    alignment, scores = align(emission, tokens, device)
    token_spans = F.merge_tokens(alignment, scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    return word_spans

def extract_lexicon(datum):
    audio = datum["audio"]
    words = preprocess_verse(datum["sentence"]).split()
    input_waveform, input_sample_rate = audio["array"], audio["sampling_rate"]
    input_waveform = torch.from_numpy(input_waveform).to(torch.float32).unsqueeze(0)
    resampler = T.Resample(input_sample_rate, bundle.sample_rate, dtype=input_waveform.dtype)
    resampled_waveform = resampler(input_waveform)
    # split audio into chunks to avoid OOM and faster inference
    chunk_size_frames = chunk_size_s * bundle.sample_rate
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
    ratio = resampled_waveform.size(1) / num_frames
    assert len(DICTIONARY) == emission.shape[2]

    word_spans = compute_alignments(emission, words, DICTIONARY, device)

    segments = []
    for word_span, word in zip(word_spans, words):
        segments.append({
            "word": word,
            "start": round(ratio * word_span[0].start / bundle.sample_rate, 2),
            "end": round(ratio * word_span[-1].start / bundle.sample_rate, 2),
        })

    offsets = pipe(audio)
    offsets = [{"phoneme": o["text"], "start": o["timestamp"][0], "end": o["timestamp"][1]} for o in offsets["chunks"] if o["text"] != " "]

    lex = []
    for word in segments:
        word_phonemes = [o["phoneme"] for o in offsets if o['start'] >= word['start'] - THRESHOLD and o['end'] <= word['end'] + THRESHOLD]
        lex.append((word["word"], " ".join(word_phonemes)))
    
    return lex

dataset = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", num_proc=os.cpu_count())

lexicon = []
word_dictionary = {}

for datum in tqdm(dataset):
    lexicon += extract_lexicon(datum)

for (word, phoneme) in lexicon:
    word_dictionary[word] = word_dictionary.get(word, {})
    word_dictionary[word][phoneme] = word_dictionary[word].get(phoneme, 0) + 1

with open("./common_voice.json", "w") as f:
    json.dump(word_dictionary, f, ensure_ascii=False)