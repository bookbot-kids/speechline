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
from speechline.run import Runner
from speechline.transcribers import Wav2Vec2Transcriber, WhisperTranscriber
from speechline.utils.config import Config
from speechline.utils.dataset import format_audio_dataset, prepare_dataframe
from speechline.utils.io import export_transcripts_json
from speechline.utils.segmenter import AudioSegmenter


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
    assert df.shape[1] == 4


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
    transcriptions = transcriber.predict(dataset)
    assert transcriptions == [
        "h h ɚ i d ʌ m b ɹ ɛ l ə ɪ z d͡ʒ ʌ s t ð ə b ɛ s t",
        "ɪ t ɪ z n oʊ t ʌ p",
        "s ə b l ɛ n s ɪ p z ə f i t p l i s æ æ p l æ p ə",
    ]

    phoneme_offsets = transcriber.predict(dataset, output_offsets=True)
    assert phoneme_offsets == [
        [
            {"phoneme": "h", "start_time": 0.0, "end_time": 0.04},
            {"phoneme": "h", "start_time": 0.14, "end_time": 0.2},
            {"phoneme": "ɚ", "start_time": 0.24, "end_time": 0.28},
            {"phoneme": "i", "start_time": 0.42, "end_time": 0.44},
            {"phoneme": "d", "start_time": 0.5, "end_time": 0.54},
            {"phoneme": "ʌ", "start_time": 0.64, "end_time": 0.66},
            {"phoneme": "m", "start_time": 0.7, "end_time": 0.74},
            {"phoneme": "b", "start_time": 0.78, "end_time": 0.82},
            {"phoneme": "ɹ", "start_time": 0.84, "end_time": 0.9},
            {"phoneme": "ɛ", "start_time": 0.92, "end_time": 0.94},
            {"phoneme": "l", "start_time": 1.0, "end_time": 1.04},
            {"phoneme": "ə", "start_time": 1.08, "end_time": 1.12},
            {"phoneme": "ɪ", "start_time": 1.36, "end_time": 1.38},
            {"phoneme": "z", "start_time": 1.54, "end_time": 1.58},
            {"phoneme": "d͡ʒ", "start_time": 1.58, "end_time": 1.62},
            {"phoneme": "ʌ", "start_time": 1.62, "end_time": 1.66},
            {"phoneme": "s", "start_time": 1.72, "end_time": 1.76},
            {"phoneme": "t", "start_time": 1.78, "end_time": 1.82},
            {"phoneme": "ð", "start_time": 1.86, "end_time": 1.88},
            {"phoneme": "ə", "start_time": 1.92, "end_time": 1.94},
            {"phoneme": "b", "start_time": 1.98, "end_time": 2.0},
            {"phoneme": "ɛ", "start_time": 2.04, "end_time": 2.06},
            {"phoneme": "s", "start_time": 2.22, "end_time": 2.26},
            {"phoneme": "t", "start_time": 2.38, "end_time": 2.4},
        ],
        [
            {"phoneme": "ɪ", "start_time": 0.0, "end_time": 0.02},
            {"phoneme": "t", "start_time": 0.26, "end_time": 0.3},
            {"phoneme": "ɪ", "start_time": 0.34, "end_time": 0.36},
            {"phoneme": "z", "start_time": 0.42, "end_time": 0.44},
            {"phoneme": "n", "start_time": 0.5, "end_time": 0.54},
            {"phoneme": "oʊ", "start_time": 0.54, "end_time": 0.58},
            {"phoneme": "t", "start_time": 0.58, "end_time": 0.62},
            {"phoneme": "ʌ", "start_time": 0.76, "end_time": 0.78},
            {"phoneme": "p", "start_time": 0.92, "end_time": 0.94},
        ],
        [
            {"phoneme": "s", "start_time": 0.0, "end_time": 0.02},
            {"phoneme": "ə", "start_time": 0.26, "end_time": 0.3},
            {"phoneme": "b", "start_time": 0.36, "end_time": 0.4},
            {"phoneme": "l", "start_time": 0.5, "end_time": 0.52},
            {"phoneme": "ɛ", "start_time": 0.54, "end_time": 0.56},
            {"phoneme": "n", "start_time": 0.6, "end_time": 0.62},
            {"phoneme": "s", "start_time": 0.68, "end_time": 0.7},
            {"phoneme": "ɪ", "start_time": 0.74, "end_time": 0.76},
            {"phoneme": "p", "start_time": 0.82, "end_time": 0.84},
            {"phoneme": "z", "start_time": 0.88, "end_time": 0.9},
            {"phoneme": "ə", "start_time": 1.04, "end_time": 1.08},
            {"phoneme": "f", "start_time": 1.62, "end_time": 1.66},
            {"phoneme": "i", "start_time": 1.7, "end_time": 1.72},
            {"phoneme": "t", "start_time": 1.78, "end_time": 1.8},
            {"phoneme": "p", "start_time": 1.8, "end_time": 1.82},
            {"phoneme": "l", "start_time": 1.86, "end_time": 1.88},
            {"phoneme": "i", "start_time": 1.92, "end_time": 1.94},
            {"phoneme": "s", "start_time": 2.04, "end_time": 2.08},
            {"phoneme": "æ", "start_time": 2.12, "end_time": 2.14},
            {"phoneme": "æ", "start_time": 2.28, "end_time": 2.3},
            {"phoneme": "p", "start_time": 2.4, "end_time": 2.42},
            {"phoneme": "l", "start_time": 2.46, "end_time": 2.48},
            {"phoneme": "æ", "start_time": 2.52, "end_time": 2.54},
            {"phoneme": "p", "start_time": 2.62, "end_time": 2.64},
            {"phoneme": "ə", "start_time": 2.78, "end_time": 2.8},
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
            audio_path,
            tmpdir,
            offsets,
            silence_duration=0.3,
            minimum_chunk_duration=0.7,
        )
        segments.append(segment)

    assert sum([len(s) for s in segments]) + len(df) == len(glob(f"{tmpdir}/*/*.wav"))
    assert segments == [
        [
            [
                {"phoneme": "h", "start_time": 0.0, "end_time": 0.04},
                {"phoneme": "h", "start_time": 0.14, "end_time": 0.2},
                {"phoneme": "ɚ", "start_time": 0.24, "end_time": 0.28},
                {"phoneme": "i", "start_time": 0.42, "end_time": 0.44},
                {"phoneme": "d", "start_time": 0.5, "end_time": 0.54},
                {"phoneme": "ʌ", "start_time": 0.64, "end_time": 0.66},
                {"phoneme": "m", "start_time": 0.7, "end_time": 0.74},
                {"phoneme": "b", "start_time": 0.78, "end_time": 0.82},
                {"phoneme": "ɹ", "start_time": 0.84, "end_time": 0.9},
                {"phoneme": "ɛ", "start_time": 0.92, "end_time": 0.94},
                {"phoneme": "l", "start_time": 1.0, "end_time": 1.04},
                {"phoneme": "ə", "start_time": 1.08, "end_time": 1.12},
                {"phoneme": "ɪ", "start_time": 1.36, "end_time": 1.38},
                {"phoneme": "z", "start_time": 1.54, "end_time": 1.58},
                {"phoneme": "d͡ʒ", "start_time": 1.58, "end_time": 1.62},
                {"phoneme": "ʌ", "start_time": 1.62, "end_time": 1.66},
                {"phoneme": "s", "start_time": 1.72, "end_time": 1.76},
                {"phoneme": "t", "start_time": 1.78, "end_time": 1.82},
                {"phoneme": "ð", "start_time": 1.86, "end_time": 1.88},
                {"phoneme": "ə", "start_time": 1.92, "end_time": 1.94},
                {"phoneme": "b", "start_time": 1.98, "end_time": 2.0},
                {"phoneme": "ɛ", "start_time": 2.04, "end_time": 2.06},
                {"phoneme": "s", "start_time": 2.22, "end_time": 2.26},
                {"phoneme": "t", "start_time": 2.38, "end_time": 2.4},
            ]
        ],
        [
            [
                {"phoneme": "ɪ", "start_time": 0.0, "end_time": 0.02},
                {"phoneme": "t", "start_time": 0.26, "end_time": 0.3},
                {"phoneme": "ɪ", "start_time": 0.34, "end_time": 0.36},
                {"phoneme": "z", "start_time": 0.42, "end_time": 0.44},
                {"phoneme": "n", "start_time": 0.5, "end_time": 0.54},
                {"phoneme": "oʊ", "start_time": 0.54, "end_time": 0.58},
                {"phoneme": "t", "start_time": 0.58, "end_time": 0.62},
                {"phoneme": "ʌ", "start_time": 0.76, "end_time": 0.78},
                {"phoneme": "p", "start_time": 0.92, "end_time": 0.94},
            ]
        ],
        [
            [
                {"phoneme": "s", "start_time": 0.0, "end_time": 0.02},
                {"phoneme": "ə", "start_time": 0.26, "end_time": 0.3},
                {"phoneme": "b", "start_time": 0.36, "end_time": 0.4},
                {"phoneme": "l", "start_time": 0.5, "end_time": 0.52},
                {"phoneme": "ɛ", "start_time": 0.54, "end_time": 0.56},
                {"phoneme": "n", "start_time": 0.6, "end_time": 0.62},
                {"phoneme": "s", "start_time": 0.68, "end_time": 0.7},
                {"phoneme": "ɪ", "start_time": 0.74, "end_time": 0.76},
                {"phoneme": "p", "start_time": 0.82, "end_time": 0.84},
                {"phoneme": "z", "start_time": 0.88, "end_time": 0.9},
                {"phoneme": "ə", "start_time": 1.04, "end_time": 1.08},
            ],
            [
                {"phoneme": "f", "start_time": 0.0, "end_time": 0.04},
                {"phoneme": "i", "start_time": 0.08, "end_time": 0.1},
                {"phoneme": "t", "start_time": 0.16, "end_time": 0.18},
                {"phoneme": "p", "start_time": 0.18, "end_time": 0.2},
                {"phoneme": "l", "start_time": 0.24, "end_time": 0.26},
                {"phoneme": "i", "start_time": 0.3, "end_time": 0.32},
                {"phoneme": "s", "start_time": 0.42, "end_time": 0.46},
                {"phoneme": "æ", "start_time": 0.5, "end_time": 0.52},
                {"phoneme": "æ", "start_time": 0.66, "end_time": 0.68},
                {"phoneme": "p", "start_time": 0.78, "end_time": 0.8},
                {"phoneme": "l", "start_time": 0.84, "end_time": 0.86},
                {"phoneme": "æ", "start_time": 0.9, "end_time": 0.92},
                {"phoneme": "p", "start_time": 1.0, "end_time": 1.02},
                {"phoneme": "ə", "start_time": 1.16, "end_time": 1.18},
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
        " Her red umbrella is just the best.",
        " It is not up.",
        " Sepulang sekolah, fitri sangat lapar.",
    ]

    offsets = transcriber.predict(dataset, output_offsets=True)
    assert offsets == [
        [
            {
                "text": " Her red umbrella is just the best.",
                "start_time": 0.0,
                "end_time": 3.0,
            }
        ],
        [{"text": " It is not up.", "start_time": 0.0, "end_time": 2.0}],
        [
            {
                "text": " Sepulang sekolah, fitri sangat lapar.",
                "start_time": 0.0,
                "end_time": 3.0,
            }
        ],
    ]


def test_runner(datadir, tmpdir):
    args = Runner.parse_args(
        [
            "--input_dir",
            str(datadir),
            "--output_dir",
            str(tmpdir),
            "--config",
            f"{datadir}/config.json",
        ]
    )
    config = Config(args.config)
    runner = Runner(config, args.input_dir, args.output_dir)
    runner.run()
    assert len(glob(f"{tmpdir}/*/*.wav")) == 7


def test_failed_run(datadir, tmpdir):
    args = Runner.parse_args(
        [
            "--input_dir",
            str(datadir),
            "--output_dir",
            str(tmpdir),
            "--config",
            f"{datadir}/config.json",
        ]
    )
    config = Config(args.config)
    # inject unsupported language
    config.languages = ["zh"]
    with pytest.raises(AttributeError):
        config.validate_config()

    runner = Runner(config, args.input_dir, args.output_dir)
    runner.run()
