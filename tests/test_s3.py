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

import pytest

from speechline.utils import s3


@pytest.fixture
def bucket_name():
    return "my-test-bucket"


@pytest.fixture
def s3_test(s3_client, bucket_name):
    s3_client.create_bucket(Bucket=bucket_name)
    yield


def test_s3_utils(tmp_path, s3_client, s3_test, bucket_name):
    my_client = s3.S3Client()
    testbox = tmp_path / "testbox"
    foo = testbox / "foo.txt"
    bar = testbox / "subbox/bar.txt"
    my_client.put_object(bucket_name, key=str(testbox / "baz/"), value="sb")
    my_client.put_object(bucket_name, key=str(foo), value="foo")
    my_client.put_object(bucket_name, key=str(bar), value="bar")
    my_client.download_s3_folder(bucket_name, s3_folder=str(testbox))
    my_client.download_s3_folder(
        bucket_name, s3_folder=str(testbox), local_dir=tmp_path
    )
    my_client.upload_folder(bucket_name, "uploads", tmp_path)
    assert foo.read_text() == "foo"
    assert bar.read_text() == "bar"
