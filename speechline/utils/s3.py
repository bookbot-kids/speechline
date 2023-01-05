# Copyright 2022 [PT BOOKBOT INDONESIA](https://bookbot.id/)
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
import boto3


class S3Client:
    def __init__(self, region_name="us-east-1") -> None:
        self.client = boto3.client("s3", region_name=region_name)
        self.resource = boto3.resource("s3", region_name=region_name)

    def download_s3_folder(
        self, bucket: str, s3_folder: str, local_dir: str = None
    ) -> None:
        """
        Download the contents of a folder directory in an S3 bucket.
        Source: https://stackoverflow.com/a/62945526

        Args:
            bucket (str): Name of the s3 bucket
            s3_folder (str): Folder path in the s3 bucket
            local_dir (str, optional): Relative or absolute directory path in the local file system. Defaults to None.
        """
        bucket = self.resource.Bucket(bucket)
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

    def put_object(self, bucket: str, key: str, value: str):
        """Puts `value` (in str) to S3 bucket.

        Args:
            bucket (str): S3 bucket name.
            key (str): Key to file in bucket.
            value (str): String representation of object to put in S3.
        """
        self.client.put_object(Bucket=bucket, Key=key, Body=value)
