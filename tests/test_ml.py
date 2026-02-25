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
from speechline.classifiers import ASTClassifier, Wav2Vec2Classifier
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
    assert len(transcriptions) == 3
    assert all(isinstance(t, str) for t in transcriptions)
    assert all(len(t) > 0 for t in transcriptions)

    output_offsets = transcriber.predict(
        dataset, return_timestamps="char", output_offsets=True
    )
    assert len(output_offsets) == 3
    for offsets in output_offsets:
        assert isinstance(offsets, list)
        for o in offsets:
            assert "text" in o
            assert "start_time" in o
            assert "end_time" in o

    segmenter = SilenceSegmenter()
    segments = []
    noise_classifier_checkpoint = "bookbot/distil-ast-audioset"
    noise_classifier = ASTClassifier(noise_classifier_checkpoint)

    for audio_path, offsets in zip(df["audio"], output_offsets):
        json_path = Path(audio_path).with_suffix(".json")
        export_transcripts_json(json_path, offsets)
        assert json_path.exists()
        assert json.load(open(json_path)) == offsets

        segment = segmenter.chunk_audio_segments(
            audio_path,
            tmpdir,
            offsets,
            do_noise_classify=True,
            noise_classifier=noise_classifier,
            minimum_empty_duration=0.1,
            minimum_chunk_duration=0.7,
            noise_classifier_threshold=0.3,
            silence_duration=0.3,
        )
        segments.append(segment)

    assert sum([len(s) for s in segments]) + len(df) == len(glob(f"{tmpdir}/*/*.wav"))


def test_whisper_transcriber(datadir):
    model_checkpoint = "openai/whisper-tiny"
    transcriber = WhisperTranscriber(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = format_audio_dataset(df, sampling_rate=transcriber.sampling_rate)
    transcriptions = transcriber.predict(dataset)
    assert len(transcriptions) == 3
    assert all(isinstance(t, str) for t in transcriptions)
    assert all(len(t) > 0 for t in transcriptions)

    offsets = transcriber.predict(dataset, output_offsets=True)
    assert len(offsets) == 3
    for offset_list in offsets:
        assert isinstance(offset_list, list)
        assert len(offset_list) > 0
        for o in offset_list:
            assert "text" in o
            assert "start_time" in o
            assert "end_time" in o


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
