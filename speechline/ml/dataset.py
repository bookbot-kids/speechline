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

                    - audio path
                    - audio ID
                    - ground truth
                    - language
                    - language code
    """
    get_ground_truth = lambda p: open(p).read() if Path(p).exists() else ""

    audios = glob(f"{path_to_files}/*/*.{audio_extension}")
    df = pd.DataFrame({"audio": audios})
    # ID is filename stem (before extension)
    df["id"] = df["audio"].apply(lambda f: Path(f).stem)
    # language code is immediate parent directory
    df["language_code"] = df["audio"].apply(lambda f: Path(f).parent.name)
    df["language"] = df["language_code"].apply(lambda f: f.split("-")[0])
    # ground truth is same filename, except with .txt extension
    df["ground_truth"] = df["audio"].apply(
        lambda f: get_ground_truth(f.replace(f".{audio_extension}", ".txt"))
    )
    return df
