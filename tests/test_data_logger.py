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

import pytest
import requests

from scripts.data_logger import DataLogger
from speechline.utils.airtable import AirTable


class MockSuccessfulResponse:
    ok = True

    @staticmethod
    def json():
        return {
            "records": [
                {
                    "id": "rect3opE5TEpQDZGh",
                    "createdTime": "2022-09-12T05:00:17.000Z",
                    "fields": {
                        "language": "en",
                        "date": "2022-09-11",
                        "language-code": "en-ae",
                        "label": "archive",
                        "duration": 773,
                    },
                },
                {
                    "id": "recIxaYjVIIf7xPzg",
                    "createdTime": "2022-09-12T05:00:17.000Z",
                    "fields": {
                        "language": "en",
                        "date": "2022-09-11",
                        "language-code": "en-aq",
                        "label": "archive",
                        "duration": 28,
                    },
                },
                {
                    "id": "recfjYb9Z0PKaF9cP",
                    "createdTime": "2022-09-12T05:00:17.000Z",
                    "fields": {
                        "language": "en",
                        "date": "2022-09-11",
                        "language-code": "en-au",
                        "label": "archive",
                        "duration": 188083,
                    },
                },
            ]
        }


@pytest.fixture
def mock_response(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockSuccessfulResponse()

    monkeypatch.setattr(requests, "post", mock_post)


def test_successful_data_logger(monkeypatch, mock_response, datadir):
    url, api_key, label = "AIRTABLE_URL", "DUMMY_API_KEY", "training"
    monkeypatch.setenv("AIRTABLE_API_KEY", api_key)

    datadir = str(datadir)
    logger = DataLogger()
    args = logger.parse_args(["--url", url, "--input_dir", datadir, "--label", label])
    # TODO: get_audio_duration not tested due to using multithreading
    # even though `.log` method calls it
    assert logger.get_audio_duration(sorted(glob(f"{datadir}/**/*.wav"))[0]) == 2.38
    # test successful response via mock_response
    success = logger.log(args.url, args.input_dir, args.label)
    assert success is True


def test_failed_data_logger(monkeypatch, datadir):
    url, api_key, label = "AIRTABLE_URL", "DUMMY_API_KEY", "training"
    monkeypatch.setenv("AIRTABLE_API_KEY", api_key)

    datadir = str(datadir)
    logger = DataLogger()
    args = logger.parse_args(["--url", url, "--input_dir", datadir, "--label", label])
    # test failed response
    success = logger.log(args.url, args.input_dir, args.label)
    assert success is False


def test_no_api_key(monkeypatch):
    monkeypatch.delenv("AIRTABLE_API_KEY", raising=False)
    with pytest.raises(OSError):
        _ = AirTable("AIRTABLE_URL")
