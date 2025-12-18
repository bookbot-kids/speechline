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

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import boto3
from tqdm import tqdm


class S3Client:
    """
    AWS S3 Client Interface.

    Args:
        region_name (str, optional):
            AWS region name. Defaults to `"us-east-1"`.
    """

    def __init__(self, region_name: str = "us-east-1") -> None:
        self.client = boto3.client("s3", region_name=region_name)
        self.resource = boto3.resource("s3", region_name=region_name)

    def download_s3_folder(
        self, bucket_name: str, s3_folder: str, local_dir: Optional[str] = None
    ) -> None:
        """
        Download the contents of a folder directory in an S3 bucket.
        Source: [StackOverflow](https://stackoverflow.com/a/62945526).

        Args:
            bucket_name (str):
                Name of the s3 bucket
            s3_folder (str):
                Folder path in the s3 bucket
            local_dir (Optional[str], optional):
                Relative or absolute directory path in the local file system.
                Defaults to `None`.
        """
        bucket = self.resource.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            # use key as save path if local_dir not specified, otherwise use local_dir
            target = (
                obj.key
                if local_dir is None
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            )
            # create dir if target does't exist
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            # skip subfolders
            if obj.key[-1] == "/":
                continue
            bucket.download_file(obj.key, target)

    def upload_folder(self, bucket_name: str, prefix: str, local_dir: str) -> None:
        """
        Uploads all files under `local_dir` to S3 bucket with `prefix`.
        Utilizes parallelism to speed up upload process.

        ### Example
        ```title="Sample Directory"
        tmp/
        └── en-us
            ├── utt_0.tsv
            ├── utt_0.wav
        └── id-id
            ├── utt_1.tsv
            └── utt_1.wav
        ```
        ```pycon title="example_upload_folder.py"
        >>> bucket_name, prefix, local_dir = "my-bucket", "train/", "tmp/"
        >>> my_client = S3Client()
        >>> my_client.upload_folder(bucket_name, prefix, local_dir)
        ```
        ```title="Result"
        Uploaded tmp/en-us/utt_0.tsv to s3://my-bucket/train/en-us/utt_0.tsv
        Uploaded tmp/en-us/utt_0.wav to s3://my-bucket/train/en-us/utt_0.wav
        Uploaded tmp/id-id/utt_1.tsv to s3://my-bucket/train/id-id/utt_1.tsv
        Uploaded tmp/id-id/utt_1.wav to s3://my-bucket/train/id-id/utt_1.wav
        ```

        Args:
            bucket_name (str):
                S3 bucket name.
            prefix (str):
                Object key's prefix.
            local_dir (str):
                Path to local directory.
        """
        paths, keys = [], []
        # recursively walk through local directory
        # setup file paths and object keys
        for root, dirs, files in os.walk(local_dir):
            # skip hidden folders
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                # skip hidden files
                if not file.startswith("."):
                    # path to file in local dir
                    path = os.path.join(root, file)
                    # relative path from local dir to file
                    relpath = os.path.relpath(path, local_dir)
                    # object key from prefix to relative path
                    key = os.path.join(prefix, relpath)
                    paths.append(path)
                    keys.append(key)

        fn = partial(self.upload_file, bucket_name=bucket_name)
        with ThreadPoolExecutor() as executor:
            _ = list(tqdm(executor.map(fn, keys, paths), total=len(keys)))

    def put_object(self, bucket_name: str, key: str, value: str) -> None:
        """
        Puts `value` (in str) to S3 bucket.

        Args:
            bucket_name (str):
                S3 bucket name.
            key (str):
                Key to file in bucket.
            value (str):
                String representation of object to put in S3.
        """
        self.client.put_object(Bucket=bucket_name, Key=key, Body=value)

    def upload_file(self, key: str, path: str, bucket_name: str) -> None:
        """
        Uploads file at `path` to S3 bucket with `key` as object key.

        Args:
            key (str):
                Key to file in bucket.
            path (str):
                Path to local file to upload.
            bucket_name (str):
                S3 bucket name.
        """
        self.client.upload_file(Bucket=bucket_name, Key=key, Filename=path)
