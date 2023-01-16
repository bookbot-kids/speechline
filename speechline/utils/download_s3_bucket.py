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

from typing import List
import argparse
import sys

from speechline.utils.s3 import S3Client


def parse_args(args: List[str]) -> argparse.Namespace:
    """Utility argument parser function for S3 bucket downloader.

    Args:
        args (List[str]): List of arguments.

    Returns:
        argparse.Namespace: Objects with arguments values as attributes.
    """
    parser = argparse.ArgumentParser(
        prog="python speechline/utils/download_s3_bucket.py",
        description="Download an S3 bucket with folder prefix.",
    )

    parser.add_argument(
        "-b", "--bucket", type=str, required=True, help="S3 bucket name."
    )
    parser.add_argument(
        "-p", "--prefix", type=str, required=True, help="S3 folder prefix."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to local output directory.",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=str,
        default="ap-southeast-1",
        help="AWS region name.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    s3_client = S3Client(region_name=args.region)
    s3_client.download_s3_folder(
        args.bucket, s3_folder=args.prefix, local_dir=args.output_dir
    )
