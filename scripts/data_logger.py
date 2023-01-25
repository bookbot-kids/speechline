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

from typing import List, Dict, Any
from datetime import date
from p_tqdm import p_umap
from glob import glob
import subprocess
import argparse
import sys
import os

from speechline.utils.airtable import AirTable


class DataLogger:
    def parse_args(self, args: List[str]) -> argparse.Namespace:
        """Utility argument parser function for data logging to AirTable.

        Args:
            args (List[str]): List of arguments.

        Returns:
            argparse.Namespace: Objects with arguments values as attributes.
        """
        parser = argparse.ArgumentParser(
            prog="python scripts/data_logger.py",
            description="Log region-grouped total audio duration to AirTable.",
        )

        parser.add_argument(
            "-u", "--url", type=str, required=True, help="AirTable URL."
        )
        parser.add_argument(
            "-i",
            "--input_dir",
            type=str,
            required=True,
            help="Directory of input audios to log.",
        )
        parser.add_argument(
            "-l",
            "--label",
            type=str,
            required=True,
            help="Log record label. E.g. training/archive.",
        )
        return parser.parse_args(args)

    def get_audio_duration(self, audio_path: str) -> float:
        """Calculate audio duration via ffprobe.
        Equivalent to:
        ```sh title="example_get_audio_duration.sh"
        ffprobe -v quiet -of csv=p=0 -show_entries format=duration {audio_path}
        ```

        Args:
            audio_path (str): Path to audio file.

        Returns:
            float: Duration in seconds.
        """
        job = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-of",
                "csv=p=0",
                "-show_entries",
                "format=duration",
                audio_path,
            ],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        duration = float(job.stdout.decode())
        return duration

    def get_language_total_audio_duration(self, input_dir: str) -> Dict[str, float]:
        """Map language folders in `input_dir` to their respective total audio duration.

        ### Example
        ```pycon title="example_get_language_total_audio_duration.py"
        >>> logger = DataLogger()
        >>> logger.get_language_total_audio_duration("dropbox/")
        {'en-au': 3.936, 'id-id': 3.797}
        ```
        Assumes `input_dir` as `{input_dir}/{lang}/{audio}.wav`.
        Returns

        Args:
            input_dir (str): Path to input directory.

        Returns:
            Dict[str, float]: Dictionary of language to total audio duration.
        """
        languages = [f.name for f in os.scandir(input_dir) if f.is_dir()]
        language2duration = {}
        for language in languages:
            audios = glob(f"{input_dir}/{language}/*.wav")
            duration = round(sum(p_umap(self.get_audio_duration, audios)), 3)
            language2duration[language] = duration
        return language2duration

    def build_payload(
        self, date: str, label: str, language: str, duration: float
    ) -> Dict[str, Dict[str, Any]]:
        """Builds payload for AirTable record.
        AirTable record has the following structure:
        ```pycon
        >>> {
        ...     "date": {YYYY-MM-DD},
        ...     "label": {label},
        ...     "language": {lang},
        ...     "language-code": {lang-country},
        ...     "duration": {duration},
        ... }
        ```

        Args:
            date (str): Logging date.
            label (str): Audio folder label.
            language (str): Language code (lang-country). E.g. `en-us`.
            duration (float): Duration in seconds.

        Returns:
            Dict[str, Dict[str, Any]]: AirTable record payload.
        """
        return {
            "fields": {
                "date": date,
                "label": label,
                "language": language.split("-")[0],
                "language-code": language,
                "duration": duration,
            }
        }

    def log(self, url: str, input_dir: str, label: str) -> bool:
        """Logs region-grouped total audio duration in `input_dir` to AirTable at `url`.

        Args:
            url (str): AirTable URL.
            input_dir (str): Input directory to log.
            label (str): Log record label.

        Returns:
            bool: Whether upload was a success.
        """
        airtable = AirTable(url)
        language2duration = self.get_language_total_audio_duration(input_dir)
        records = [
            self.build_payload(
                str(date.today()),
                label,
                language,
                duration,
            )
            for language, duration in language2duration.items()
        ]
        return airtable.batch_add_records(records)


if __name__ == "__main__":
    logger = DataLogger()
    args = logger.parse_args(sys.argv[1:])
    status = logger.log(args.url, args.input_dir, args.label)
    print(status)
