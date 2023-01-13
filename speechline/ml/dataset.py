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

import pandas as pd
from glob import glob
from pathlib import Path


def prepare_dataframe(path_to_files: str, audio_extension: str = "wav") -> pd.DataFrame:
    """Prepares audio and ground truth files as Pandas `DataFrame`.
    Assumes files are of the following structure:

    ```
    path_to_files
    ├── langX
    │   ├── a.{audio_extension}
    │   ├── a.txt
    │   ├── b.{audio_extension}
    │   └── b.txt
    └── langY
        ├── c.{audio_extension}
        └── c.txt
    ```

    Args:
        path_to_files (str): Path to files.

    Returns:
        pd.DataFrame:
            DataFrame consisting of:

            - `audio` (audio path)
            - `id`
            - `ground_truth`
            - `language`
            - `language_code`
    """
    audios = sorted(glob(f"{path_to_files}/*/*.{audio_extension}"))
    df = pd.DataFrame({"audio": audios})
    # ID is filename stem (before extension)
    df["id"] = df["audio"].apply(lambda f: Path(f).stem)
    # language code is immediate parent directory
    df["language_code"] = df["audio"].apply(lambda f: Path(f).parent.name)
    df["language"] = df["language_code"].apply(lambda f: f.split("-")[0])
    # ground truth is same filename, except with .txt extension
    df["ground_truth"] = df["audio"].apply(lambda p: Path(p).with_suffix(".txt"))
    df["ground_truth"] = df["ground_truth"].apply(
        lambda p: open(p).read() if p.exists() else ""
    )
    return df
