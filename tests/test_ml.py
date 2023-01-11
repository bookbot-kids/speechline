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

from speechline.ml.dataset import prepare_dataframe
from speechline.ml.classifier import Wav2Vec2Classifier
from speechline.ml.transcriber import Wav2Vec2Transcriber, WhisperTranscriber
from speechline.utils.aac_to_wav import parse_args, convert_to_wav


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


def test_wav2vec2_transcriber(datadir):
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
