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

from glob import glob
from pathlib import Path
import json

from speechline.ml.dataset import prepare_dataframe
from speechline.ml.classifier import Wav2Vec2Classifier
from speechline.ml.transcriber import Wav2Vec2Transcriber, WhisperTranscriber
from speechline.utils.aac_to_wav import parse_args, convert_to_wav
from speechline.utils.io import export_transcripts_json
from speechline.utils.segmenter import AudioSegmenter


def test_convert_to_wav(datadir):
    datadir = str(datadir)
    parser = parse_args([datadir, "-c", "2", "-r", "24_000"])
    assert parser.input_dir == datadir
    assert parser.channel == 2
    assert parser.rate == 24_000
    audios = glob(f"{datadir}/**/*.aac", recursive=True)
    # get first audio file as sample audio
    audio_path = audios[0]
    convert_to_wav(audio_path, num_channels=parser.channel, sampling_rate=parser.rate)
    # assert that new wav file exists
    assert Path(audio_path).with_suffix(".wav").exists()


def test_prepare_dataframe(datadir):
    df = prepare_dataframe(datadir)
    assert df.shape[1] == 5


def test_audio_classifier(datadir):
    model_checkpoint = "bookbot/distil-wav2vec2-adult-child-cls-52m"
    classifier = Wav2Vec2Classifier(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = classifier.format_audio_dataset(df)
    predictions = classifier.predict(dataset)
    assert predictions == ["child", "child", "child"]


def test_wav2vec2_transcriber(datadir, tmpdir):
    model_checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
    transcriber = Wav2Vec2Transcriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = transcriber.format_audio_dataset(df)
    transcriptions = transcriber.predict(dataset)
    assert transcriptions == [
        "h h ɚ i d ʌ m b ɹ ɛ l ə ɪ z d͡ʒ ʌ s t ð ə b ɛ s t",
        "ɪ t ɪ z n oʊ t ʌ p",
        "s ə b l ɛ n s ɪ p z ə f i t p l i s æ æ p l æ p ə",
    ]

    phoneme_offsets = transcriber.predict(dataset, output_phoneme_offsets=True)
    assert phoneme_offsets == [
        [
            {"phoneme": "h", "start_time": 0.01, "end_time": 0.05},
            {"phoneme": "h", "start_time": 0.05, "end_time": 0.17},
            {"phoneme": "ɚ", "start_time": 0.17, "end_time": 0.25},
            {"phoneme": "i", "start_time": 0.25, "end_time": 0.43},
            {"phoneme": "d", "start_time": 0.43, "end_time": 0.52},
            {"phoneme": "ʌ", "start_time": 0.52, "end_time": 0.65},
            {"phoneme": "m", "start_time": 0.65, "end_time": 0.72},
            {"phoneme": "b", "start_time": 0.72, "end_time": 0.8},
            {"phoneme": "ɹ", "start_time": 0.8, "end_time": 0.87},
            {"phoneme": "ɛ", "start_time": 0.87, "end_time": 0.93},
            {"phoneme": "l", "start_time": 0.93, "end_time": 1.02},
            {"phoneme": "ə", "start_time": 1.02, "end_time": 1.1},
            {"phoneme": "ɪ", "start_time": 1.1, "end_time": 1.37},
            {"phoneme": "z", "start_time": 1.37, "end_time": 1.55},
            {"phoneme": "d͡ʒ", "start_time": 1.55, "end_time": 1.61},
            {"phoneme": "ʌ", "start_time": 1.61, "end_time": 1.65},
            {"phoneme": "s", "start_time": 1.65, "end_time": 1.74},
            {"phoneme": "t", "start_time": 1.74, "end_time": 1.8},
            {"phoneme": "ð", "start_time": 1.8, "end_time": 1.87},
            {"phoneme": "ə", "start_time": 1.87, "end_time": 1.93},
            {"phoneme": "b", "start_time": 1.93, "end_time": 1.99},
            {"phoneme": "ɛ", "start_time": 1.99, "end_time": 2.06},
            {"phoneme": "s", "start_time": 2.06, "end_time": 2.24},
            {"phoneme": "t", "start_time": 2.24, "end_time": 2.39},
        ],
        [
            {"phoneme": "ɪ", "start_time": 0.01, "end_time": 0.05},
            {"phoneme": "t", "start_time": 0.05, "end_time": 0.28},
            {"phoneme": "ɪ", "start_time": 0.28, "end_time": 0.35},
            {"phoneme": "z", "start_time": 0.35, "end_time": 0.43},
            {"phoneme": "n", "start_time": 0.43, "end_time": 0.52},
            {"phoneme": "oʊ", "start_time": 0.52, "end_time": 0.57},
            {"phoneme": "t", "start_time": 0.57, "end_time": 0.62},
            {"phoneme": "ʌ", "start_time": 0.62, "end_time": 0.75},
            {"phoneme": "p", "start_time": 0.75, "end_time": 0.93},
        ],
        [
            {"phoneme": "s", "start_time": 0.01, "end_time": 0.21},
            {"phoneme": "ə", "start_time": 0.21, "end_time": 0.28},
            {"phoneme": "b", "start_time": 0.28, "end_time": 0.38},
            {"phoneme": "l", "start_time": 0.38, "end_time": 0.51},
            {"phoneme": "ɛ", "start_time": 0.51, "end_time": 0.55},
            {"phoneme": "n", "start_time": 0.55, "end_time": 0.61},
            {"phoneme": "s", "start_time": 0.61, "end_time": 0.69},
            {"phoneme": "ɪ", "start_time": 0.69, "end_time": 0.75},
            {"phoneme": "p", "start_time": 0.75, "end_time": 0.83},
            {"phoneme": "z", "start_time": 0.83, "end_time": 0.89},
            {"phoneme": "ə", "start_time": 0.89, "end_time": 1.06},
            {"phoneme": "f", "start_time": 1.12, "end_time": 1.64},
            {"phoneme": "i", "start_time": 1.64, "end_time": 1.71},
            {"phoneme": "t", "start_time": 1.71, "end_time": 1.79},
            {"phoneme": "p", "start_time": 1.79, "end_time": 1.83},
            {"phoneme": "l", "start_time": 1.83, "end_time": 1.87},
            {"phoneme": "i", "start_time": 1.87, "end_time": 1.93},
            {"phoneme": "s", "start_time": 1.93, "end_time": 2.06},
            {"phoneme": "æ", "start_time": 2.06, "end_time": 2.13},
            {"phoneme": "æ", "start_time": 2.13, "end_time": 2.29},
            {"phoneme": "p", "start_time": 2.29, "end_time": 2.41},
            {"phoneme": "l", "start_time": 2.41, "end_time": 2.47},
            {"phoneme": "æ", "start_time": 2.47, "end_time": 2.53},
            {"phoneme": "p", "start_time": 2.53, "end_time": 2.63},
            {"phoneme": "ə", "start_time": 2.63, "end_time": 2.79},
        ],
    ]

    segmenter = AudioSegmenter()
    segments = []
    for audio_path, offsets in zip(df["audio"], phoneme_offsets):
        json_path = Path(audio_path).with_suffix(".json")
        export_transcripts_json(json_path, offsets)
        assert json_path.exists()
        assert json.load(open(json_path)) == offsets

        segment = segmenter.chunk_audio_segments(
            audio_path, tmpdir, offsets, silence_duration=0.05
        )
        segments.append(segment)

    assert sum([len(s) for s in segments]) + len(df) == len(glob(f"{tmpdir}/*/*.wav"))
    assert segments == [
        [
            [
                {"phoneme": "h", "start_time": 0.0, "end_time": 0.04},
                {"phoneme": "h", "start_time": 0.04, "end_time": 0.16},
                {"phoneme": "ɚ", "start_time": 0.16, "end_time": 0.24},
                {"phoneme": "i", "start_time": 0.24, "end_time": 0.42},
                {"phoneme": "d", "start_time": 0.42, "end_time": 0.51},
                {"phoneme": "ʌ", "start_time": 0.51, "end_time": 0.64},
                {"phoneme": "m", "start_time": 0.64, "end_time": 0.71},
                {"phoneme": "b", "start_time": 0.71, "end_time": 0.79},
                {"phoneme": "ɹ", "start_time": 0.79, "end_time": 0.86},
                {"phoneme": "ɛ", "start_time": 0.86, "end_time": 0.92},
                {"phoneme": "l", "start_time": 0.92, "end_time": 1.01},
                {"phoneme": "ə", "start_time": 1.01, "end_time": 1.09},
                {"phoneme": "ɪ", "start_time": 1.09, "end_time": 1.36},
                {"phoneme": "z", "start_time": 1.36, "end_time": 1.54},
                {"phoneme": "d͡ʒ", "start_time": 1.54, "end_time": 1.6},
                {"phoneme": "ʌ", "start_time": 1.6, "end_time": 1.64},
                {"phoneme": "s", "start_time": 1.64, "end_time": 1.73},
                {"phoneme": "t", "start_time": 1.73, "end_time": 1.79},
                {"phoneme": "ð", "start_time": 1.79, "end_time": 1.86},
                {"phoneme": "ə", "start_time": 1.86, "end_time": 1.92},
                {"phoneme": "b", "start_time": 1.92, "end_time": 1.98},
                {"phoneme": "ɛ", "start_time": 1.98, "end_time": 2.05},
                {"phoneme": "s", "start_time": 2.05, "end_time": 2.23},
                {"phoneme": "t", "start_time": 2.23, "end_time": 2.38},
            ]
        ],
        [
            [
                {"phoneme": "ɪ", "start_time": 0.0, "end_time": 0.04},
                {"phoneme": "t", "start_time": 0.04, "end_time": 0.27},
                {"phoneme": "ɪ", "start_time": 0.27, "end_time": 0.34},
                {"phoneme": "z", "start_time": 0.34, "end_time": 0.42},
                {"phoneme": "n", "start_time": 0.42, "end_time": 0.51},
                {"phoneme": "oʊ", "start_time": 0.51, "end_time": 0.56},
                {"phoneme": "t", "start_time": 0.56, "end_time": 0.61},
                {"phoneme": "ʌ", "start_time": 0.61, "end_time": 0.74},
                {"phoneme": "p", "start_time": 0.74, "end_time": 0.92},
            ]
        ],
        [
            [
                {"phoneme": "s", "start_time": 0.0, "end_time": 0.2},
                {"phoneme": "ə", "start_time": 0.2, "end_time": 0.27},
                {"phoneme": "b", "start_time": 0.27, "end_time": 0.37},
                {"phoneme": "l", "start_time": 0.37, "end_time": 0.5},
                {"phoneme": "ɛ", "start_time": 0.5, "end_time": 0.54},
                {"phoneme": "n", "start_time": 0.54, "end_time": 0.6},
                {"phoneme": "s", "start_time": 0.6, "end_time": 0.68},
                {"phoneme": "ɪ", "start_time": 0.68, "end_time": 0.74},
                {"phoneme": "p", "start_time": 0.74, "end_time": 0.82},
                {"phoneme": "z", "start_time": 0.82, "end_time": 0.88},
                {"phoneme": "ə", "start_time": 0.88, "end_time": 1.05},
            ],
            [
                {"phoneme": "f", "start_time": 0.0, "end_time": 0.52},
                {"phoneme": "i", "start_time": 0.52, "end_time": 0.59},
                {"phoneme": "t", "start_time": 0.59, "end_time": 0.67},
                {"phoneme": "p", "start_time": 0.67, "end_time": 0.71},
                {"phoneme": "l", "start_time": 0.71, "end_time": 0.75},
                {"phoneme": "i", "start_time": 0.75, "end_time": 0.81},
                {"phoneme": "s", "start_time": 0.81, "end_time": 0.94},
                {"phoneme": "æ", "start_time": 0.94, "end_time": 1.01},
                {"phoneme": "æ", "start_time": 1.01, "end_time": 1.17},
                {"phoneme": "p", "start_time": 1.17, "end_time": 1.29},
                {"phoneme": "l", "start_time": 1.29, "end_time": 1.35},
                {"phoneme": "æ", "start_time": 1.35, "end_time": 1.41},
                {"phoneme": "p", "start_time": 1.41, "end_time": 1.51},
                {"phoneme": "ə", "start_time": 1.51, "end_time": 1.67},
            ],
        ],
    ]


def test_whisper_transcriber(datadir):
    model_checkpoint = "openai/whisper-tiny"
    transcriber = WhisperTranscriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = transcriber.format_audio_dataset(df)
    transcriptions = transcriber.predict(dataset)
    assert transcriptions == [
        " Her red umbrella is just the best.",
        " It is not up.",
        " Suppulant Secola, Fitri Sangat Lappar",
    ]
