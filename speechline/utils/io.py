from typing import List, Dict, Any
import json
import os
from pydub import AudioSegment


def export_transcripts_json(
    output_json_path: str,
    phoneme_offsets: List[Dict[str, Any]],
) -> None:
    """Exports phoneme transcript with offsets as JSON.
    ```json
    [
      {
        "phoneme": {phoneme},
        "start_time": {start_time},
        "end_time": {end_time}
      },
      ...
    ]
    ```

    Args:
        output_json_path (str): Path to output JSON file.
        phoneme_offsets (List[Dict[str, Any]]): List of phonemes with offsets.
    """
    with open(output_json_path, "w") as f:
        json.dump(phoneme_offsets, f, indent=2)


def export_segment_transcripts_tsv(
    output_tsv_path: str, segment: List[Dict[str, Any]]
) -> None:
    """Export segment transcripts to TSV of structure
    ```
    start_time_in_secs\tend_time_in_secs\tlabel
    ```

    Args:
        output_tsv_path (str): Path to TSV file.
        segment (List[Dict[str, Any]]): List of phonemes in segment.
    """
    with open(output_tsv_path, "w") as f:
        for s in segment:
            f.write(f'{s["start_time"]}\t{s["end_time"]}\t{s["phoneme"]}\n')


def export_segment_audio_wav(output_wav_path: str, segment: AudioSegment) -> None:
    """Export segment audio to WAV.

    Equivalent to:
    `ffmpeg -i in.aac -acodec pcm_s16le -ac 1 -ar 16000 out.wav`

    Args:
        output_wav_path (str): Path to WAV file.
        segment (AudioSegment): Audio segment to export.
    """
    parameters = ["-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000"]
    segment.export(output_wav_path, format="wav", parameters=parameters)


def get_outdir_path(path: str, outdir: str) -> str:
    """Generate path at output directory.
    Assumes `path` as: `{inputdir}/{lang}/`
    Return: `{outdir}/{lang}/`

    Args:
        path (str): Path to file.
        outdir (str): Output directory where file will be saved.

    Returns:
        str: Path to output directory.
    """
    pathname, _ = os.path.splitext(path)  # remove extension
    components = os.path.normpath(pathname).split(os.sep)  # split into components
    # keep last 2 components: {outdir}/{lang}/
    output_path = f"{os.path.join(outdir, *components[-2:-1])}"
    return output_path


def get_chunk_path(path: str, outdir: str, idx: int, extension: str) -> str:
    """Generate path to chunk at output directory.
    Assumes `path` as: `{inputdir}/{lang}/{utt_id}.{old_extension}`
    Return: `{outdir}/{lang}/{utt_id}-{idx}.{extension}`

    Args:
        path (str): Path to file.
        outdir (str): Output directory where file will be saved.
        idx (int): Index of chunk.
        extension (str): New file extension.

    Returns:
        str: Path to chunk at output directory.
    """
    outdir_path = get_outdir_path(path, outdir)
    filename = os.path.splitext(os.path.basename(path))[0]
    output_path = f"{os.path.join(outdir_path, filename)}-{str(idx)}.{extension}"
    return output_path
