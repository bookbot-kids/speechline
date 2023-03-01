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

import json
from glob import glob
from pathlib import Path

import pytest

from scripts.aac_to_wav import convert_to_wav, parse_args
from speechline.classifiers import Wav2Vec2Classifier
from speechline.config import Config, SegmenterConfig, TranscriberConfig
from speechline.run import Runner
from speechline.segmenters import SilenceSegmenter
from speechline.transcribers import Wav2Vec2Transcriber, WhisperTranscriber
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe
from speechline.utils.io import export_transcripts_json


def test_convert_to_wav(datadir):
    datadir = str(datadir)
    parser = parse_args(["--input_dir", datadir, "-c", "2", "-r", "24_000"])
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


def test_empty_dataframe():
    with pytest.raises(ValueError):
        _ = prepare_dataframe("foo")


def test_audio_classifier(datadir):
    model_checkpoint = "bookbot/distil-wav2vec2-adult-child-cls-52m"
    classifier = Wav2Vec2Classifier(model_checkpoint, max_duration_s=3.0)
    df = prepare_dataframe(datadir)
    dataset = format_audio_dataset(df, sampling_rate=classifier.sampling_rate)
    predictions = classifier.predict(dataset)
    assert predictions == ["child", "child", "child"]


def test_wav2vec2_transcriber(datadir, tmpdir):
    model_checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
    transcriber = Wav2Vec2Transcriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = format_audio_dataset(df, sampling_rate=transcriber.sampling_rate)
    transcriptions = transcriber.predict(dataset, return_timestamps="char")
    assert transcriptions == [
        "h h ɚ i d ʌ m b ɹ ɛ l ə ɪ z d͡ʒ ʌ s t ð ə b ɛ s t",
        "ɪ t ɪ z n oʊ t ʌ p",
        "s ə b l ɛ n s ɪ p z ə f i t p l i s æ æ p l æ p ə",
    ]

    output_offsets = transcriber.predict(
        dataset, return_timestamps="char", output_offsets=True
    )
    assert output_offsets == [
        [
            {"text": "h", "start_time": 0.0, "end_time": 0.04},
            {"text": "h", "start_time": 0.14, "end_time": 0.2},
            {"text": "ɚ", "start_time": 0.24, "end_time": 0.28},
            {"text": "i", "start_time": 0.42, "end_time": 0.44},
            {"text": "d", "start_time": 0.5, "end_time": 0.54},
            {"text": "ʌ", "start_time": 0.64, "end_time": 0.66},
            {"text": "m", "start_time": 0.7, "end_time": 0.74},
            {"text": "b", "start_time": 0.78, "end_time": 0.82},
            {"text": "ɹ", "start_time": 0.84, "end_time": 0.9},
            {"text": "ɛ", "start_time": 0.92, "end_time": 0.94},
            {"text": "l", "start_time": 1.0, "end_time": 1.04},
            {"text": "ə", "start_time": 1.08, "end_time": 1.12},
            {"text": "ɪ", "start_time": 1.36, "end_time": 1.38},
            {"text": "z", "start_time": 1.54, "end_time": 1.58},
            {"text": "d͡ʒ", "start_time": 1.58, "end_time": 1.62},
            {"text": "ʌ", "start_time": 1.62, "end_time": 1.66},
            {"text": "s", "start_time": 1.72, "end_time": 1.76},
            {"text": "t", "start_time": 1.78, "end_time": 1.82},
            {"text": "ð", "start_time": 1.86, "end_time": 1.88},
            {"text": "ə", "start_time": 1.92, "end_time": 1.94},
            {"text": "b", "start_time": 1.98, "end_time": 2.0},
            {"text": "ɛ", "start_time": 2.04, "end_time": 2.06},
            {"text": "s", "start_time": 2.22, "end_time": 2.26},
            {"text": "t", "start_time": 2.38, "end_time": 2.4},
        ],
        [
            {"text": "ɪ", "start_time": 0.0, "end_time": 0.02},
            {"text": "t", "start_time": 0.26, "end_time": 0.3},
            {"text": "ɪ", "start_time": 0.34, "end_time": 0.36},
            {"text": "z", "start_time": 0.42, "end_time": 0.44},
            {"text": "n", "start_time": 0.5, "end_time": 0.54},
            {"text": "oʊ", "start_time": 0.54, "end_time": 0.58},
            {"text": "t", "start_time": 0.58, "end_time": 0.62},
            {"text": "ʌ", "start_time": 0.76, "end_time": 0.78},
            {"text": "p", "start_time": 0.92, "end_time": 0.94},
        ],
        [
            {"text": "s", "start_time": 0.0, "end_time": 0.02},
            {"text": "ə", "start_time": 0.26, "end_time": 0.3},
            {"text": "b", "start_time": 0.36, "end_time": 0.4},
            {"text": "l", "start_time": 0.5, "end_time": 0.52},
            {"text": "ɛ", "start_time": 0.54, "end_time": 0.56},
            {"text": "n", "start_time": 0.6, "end_time": 0.62},
            {"text": "s", "start_time": 0.68, "end_time": 0.7},
            {"text": "ɪ", "start_time": 0.74, "end_time": 0.76},
            {"text": "p", "start_time": 0.82, "end_time": 0.84},
            {"text": "z", "start_time": 0.88, "end_time": 0.9},
            {"text": "ə", "start_time": 1.04, "end_time": 1.08},
            {"text": "f", "start_time": 1.62, "end_time": 1.66},
            {"text": "i", "start_time": 1.7, "end_time": 1.72},
            {"text": "t", "start_time": 1.78, "end_time": 1.8},
            {"text": "p", "start_time": 1.8, "end_time": 1.82},
            {"text": "l", "start_time": 1.86, "end_time": 1.88},
            {"text": "i", "start_time": 1.92, "end_time": 1.94},
            {"text": "s", "start_time": 2.04, "end_time": 2.08},
            {"text": "æ", "start_time": 2.12, "end_time": 2.14},
            {"text": "æ", "start_time": 2.28, "end_time": 2.3},
            {"text": "p", "start_time": 2.4, "end_time": 2.42},
            {"text": "l", "start_time": 2.46, "end_time": 2.48},
            {"text": "æ", "start_time": 2.52, "end_time": 2.54},
            {"text": "p", "start_time": 2.62, "end_time": 2.64},
            {"text": "ə", "start_time": 2.78, "end_time": 2.8},
        ],
    ]

    segmenter = SilenceSegmenter()
    segments = []
    for audio_path, offsets in zip(df["audio"], output_offsets):
        json_path = Path(audio_path).with_suffix(".json")
        export_transcripts_json(json_path, offsets)
        assert json_path.exists()
        assert json.load(open(json_path)) == offsets

        segment = segmenter.chunk_audio_segments(
            audio_path,
            tmpdir,
            offsets,
            minimum_chunk_duration=0.7,
            silence_duration=0.3,
        )
        segments.append(segment)

    assert sum([len(s) for s in segments]) + len(df) == len(glob(f"{tmpdir}/*/*.wav"))
    assert segments == [
        [
            [
                {"text": "h", "start_time": 0.0, "end_time": 0.04},
                {"text": "h", "start_time": 0.14, "end_time": 0.2},
                {"text": "ɚ", "start_time": 0.24, "end_time": 0.28},
                {"text": "i", "start_time": 0.42, "end_time": 0.44},
                {"text": "d", "start_time": 0.5, "end_time": 0.54},
                {"text": "ʌ", "start_time": 0.64, "end_time": 0.66},
                {"text": "m", "start_time": 0.7, "end_time": 0.74},
                {"text": "b", "start_time": 0.78, "end_time": 0.82},
                {"text": "ɹ", "start_time": 0.84, "end_time": 0.9},
                {"text": "ɛ", "start_time": 0.92, "end_time": 0.94},
                {"text": "l", "start_time": 1.0, "end_time": 1.04},
                {"text": "ə", "start_time": 1.08, "end_time": 1.12},
                {"text": "ɪ", "start_time": 1.36, "end_time": 1.38},
                {"text": "z", "start_time": 1.54, "end_time": 1.58},
                {"text": "d͡ʒ", "start_time": 1.58, "end_time": 1.62},
                {"text": "ʌ", "start_time": 1.62, "end_time": 1.66},
                {"text": "s", "start_time": 1.72, "end_time": 1.76},
                {"text": "t", "start_time": 1.78, "end_time": 1.82},
                {"text": "ð", "start_time": 1.86, "end_time": 1.88},
                {"text": "ə", "start_time": 1.92, "end_time": 1.94},
                {"text": "b", "start_time": 1.98, "end_time": 2.0},
                {"text": "ɛ", "start_time": 2.04, "end_time": 2.06},
                {"text": "s", "start_time": 2.22, "end_time": 2.26},
                {"text": "t", "start_time": 2.38, "end_time": 2.4},
            ]
        ],
        [
            [
                {"text": "ɪ", "start_time": 0.0, "end_time": 0.02},
                {"text": "t", "start_time": 0.26, "end_time": 0.3},
                {"text": "ɪ", "start_time": 0.34, "end_time": 0.36},
                {"text": "z", "start_time": 0.42, "end_time": 0.44},
                {"text": "n", "start_time": 0.5, "end_time": 0.54},
                {"text": "oʊ", "start_time": 0.54, "end_time": 0.58},
                {"text": "t", "start_time": 0.58, "end_time": 0.62},
                {"text": "ʌ", "start_time": 0.76, "end_time": 0.78},
                {"text": "p", "start_time": 0.92, "end_time": 0.94},
            ]
        ],
        [
            [
                {"text": "s", "start_time": 0.0, "end_time": 0.02},
                {"text": "ə", "start_time": 0.26, "end_time": 0.3},
                {"text": "b", "start_time": 0.36, "end_time": 0.4},
                {"text": "l", "start_time": 0.5, "end_time": 0.52},
                {"text": "ɛ", "start_time": 0.54, "end_time": 0.56},
                {"text": "n", "start_time": 0.6, "end_time": 0.62},
                {"text": "s", "start_time": 0.68, "end_time": 0.7},
                {"text": "ɪ", "start_time": 0.74, "end_time": 0.76},
                {"text": "p", "start_time": 0.82, "end_time": 0.84},
                {"text": "z", "start_time": 0.88, "end_time": 0.9},
                {"text": "ə", "start_time": 1.04, "end_time": 1.08},
            ],
            [
                {"text": "f", "start_time": 0.0, "end_time": 0.04},
                {"text": "i", "start_time": 0.08, "end_time": 0.1},
                {"text": "t", "start_time": 0.16, "end_time": 0.18},
                {"text": "p", "start_time": 0.18, "end_time": 0.2},
                {"text": "l", "start_time": 0.24, "end_time": 0.26},
                {"text": "i", "start_time": 0.3, "end_time": 0.32},
                {"text": "s", "start_time": 0.42, "end_time": 0.46},
                {"text": "æ", "start_time": 0.5, "end_time": 0.52},
                {"text": "æ", "start_time": 0.66, "end_time": 0.68},
                {"text": "p", "start_time": 0.78, "end_time": 0.8},
                {"text": "l", "start_time": 0.84, "end_time": 0.86},
                {"text": "æ", "start_time": 0.9, "end_time": 0.92},
                {"text": "p", "start_time": 1.0, "end_time": 1.02},
                {"text": "ə", "start_time": 1.16, "end_time": 1.18},
            ],
        ],
    ]


def test_whisper_transcriber(datadir):
    model_checkpoint = "openai/whisper-tiny"
    transcriber = WhisperTranscriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = format_audio_dataset(df, sampling_rate=transcriber.sampling_rate)
    transcriptions = transcriber.predict(dataset)
    assert transcriptions == [
        "Her red umbrella is just the best.",
        "It is not up.",
        "Sepulang sekolah, fitri sangat lapar.",
    ]

    offsets = transcriber.predict(dataset, output_offsets=True)
    assert offsets == [
        [
            {
                "text": "Her red umbrella is just the best.",
                "start_time": 0.0,
                "end_time": 3.0,
            }
        ],
        [{"text": "It is not up.", "start_time": 0.0, "end_time": 2.0}],
        [
            {
                "text": "Sepulang sekolah, fitri sangat lapar.",
                "start_time": 0.0,
                "end_time": 3.0,
            }
        ],
    ]


def test_runner_wav2vec2(datadir, tmpdir):
    args = Runner.parse_args(
        [
            "--input_dir",
            str(datadir),
            "--output_dir",
            str(tmpdir),
            "--config",
            f"{datadir}/config_wav2vec2.json",
        ]
    )
    config = Config(args.config)
    Runner.run(config, args.input_dir, args.output_dir)
    assert len(glob(f"{tmpdir}/*/*.wav")) == 7


def test_runner_wav2vec2_word(datadir, tmpdir):
    args = Runner.parse_args(
        [
            "--input_dir",
            str(datadir),
            "--output_dir",
            str(tmpdir),
            "--config",
            f"{datadir}/config_wav2vec2_word.json",
        ]
    )
    config = Config(args.config)
    Runner.run(config, args.input_dir, args.output_dir)
    assert len(glob(f"{tmpdir}/*/*.wav")) == 4


def test_runner_whisper(datadir, tmpdir):
    args = Runner.parse_args(
        [
            "--input_dir",
            str(datadir),
            "--output_dir",
            str(tmpdir),
            "--config",
            f"{datadir}/config_whisper.json",
        ]
    )
    config = Config(args.config)
    Runner.run(config, args.input_dir, args.output_dir)
    assert len(glob(f"{tmpdir}/*/*.wav")) == 6


def test_invalid_transcriber_config():
    with pytest.raises(ValueError):
        _ = TranscriberConfig("seq2seq", "model", "word", 0)

    with pytest.raises(ValueError):
        _ = TranscriberConfig("wav2vec2", "model", "phoneme", 0)

    with pytest.raises(ValueError):
        _ = TranscriberConfig("whisper", "model", "word", 0)


def test_invalid_segmenter_config():
    with pytest.raises(ValueError):
        _ = SegmenterConfig("foo")
