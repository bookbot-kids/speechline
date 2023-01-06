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
from speechline.ml.classifier import AudioClassifier
from speechline.utils.aac_to_wav import parse_args, convert_to_wav


def test_convert_to_wav(datadir):
    datadir = str(datadir)
    parser = parse_args([datadir, "-c", "2", "-r", "24_000"])
    assert parser.input_dir == datadir
    assert parser.channel == 2
    assert parser.rate == 24_000
    audios = glob(f"{datadir}/**/*.aac", recursive=True)
    assert len(audios) == 3
    # get first audio file as sample audio
    audio_path = audios[0]
    convert_to_wav(audio_path, num_channels=parser.channel, sampling_rate=parser.rate)
    # assert that new wav file exists
    assert Path(audio_path).with_suffix(".wav").exists()


def test_prepare_dataframe(datadir):
    df = prepare_dataframe(datadir)
    assert df.shape == (3, 5)


def test_audio_classifier(datadir):
    model_checkpoint = "bookbot/distil-wav2vec2-adult-child-cls-52m"
    classifier = AudioClassifier(model_checkpoint)
    df = prepare_dataframe(datadir)
    dataset = classifier.format_audio_dataset(df)
    predictions = classifier.predict(dataset)
    assert predictions == ["child", "child", "child"]
