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

import json
import os
from typing import List, Dict, Union, Any
from speechline.utils.logger import Logger

# Get the same logger instance that's set up in run.py
logger = Logger.get_logger()


def flatten_manifest(manifest_data: List[Any]) -> List[Dict[str, str]]:
    """
    Flattens a potentially nested manifest structure and removes empty entries.

    Args:
        manifest_data (List[Any]):
            A list that may contain dictionaries, lists of dictionaries,
            or nested lists of dictionaries.

    Returns:
        List[Dict[str, str]]:
            A flattened list containing only non-empty dictionaries.
    """
    logger.info("Flattening manifest data structure...")

    initial_item_count = sum(1 for _ in _count_items(manifest_data))
    flattened_manifest = []

    for item in manifest_data:
        if isinstance(item, list):
            for subitem in item:
                # Handle both dictionary items and nested lists
                if isinstance(subitem, dict) and subitem:  # Only add non-empty dicts
                    flattened_manifest.append(subitem)
                elif isinstance(subitem, list):
                    # Handle potentially nested list structure
                    for nested_item in subitem:
                        if isinstance(nested_item, dict) and nested_item:
                            flattened_manifest.append(nested_item)
        elif isinstance(item, dict) and item:  # Only add non-empty dicts
            flattened_manifest.append(item)

    final_item_count = len(flattened_manifest)
    logger.info(
        f"Manifest flattening complete: {initial_item_count} items processed, {final_item_count} valid entries retained"
    )

    return flattened_manifest


def _count_items(data: List[Any]) -> List[Dict]:
    """Helper generator to count all potential dictionary items in nested structure."""
    for item in data:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _count_items(item)


def write_manifest(
    manifest_data: List[Any],
    output_path: str,
    indent: int = 2,
    force_overwrite: bool = True,
) -> None:
    """
    Writes manifest data to a JSON file as a clean array of dictionaries.

    Args:
        manifest_data (List[Any]):
            Manifest data that may contain nested structures.
        output_path (str):
            Path where the JSON manifest file will be written.
        indent (int, optional):
            Number of spaces for JSON indentation. Defaults to 2.
        force_overwrite (bool, optional):
            If True, explicitly remove any existing file before writing. Defaults to True.
    """
    logger.info(f"Preparing to write manifest to {output_path}")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Explicitly delete the file if it exists and force_overwrite is True
    if force_overwrite and os.path.exists(output_path):
        logger.info(f"Removing existing manifest file at {output_path}")
        try:
            os.remove(output_path)
            logger.info(f"Successfully removed existing manifest file")
        except Exception as e:
            logger.warning(f"Could not remove existing file: {str(e)}")

    # Flatten the manifest data
    flattened_data = flatten_manifest(manifest_data)

    # Only write if we have valid entries
    if flattened_data:
        logger.info(f"Writing {len(flattened_data)} entries to manifest file")
        with open(output_path, "w") as f:  # 'w' mode should overwrite the file
            json.dump(flattened_data, f, indent=indent)
        logger.info(f"Successfully wrote manifest to {output_path}")
    else:
        logger.warning(
            f"No valid entries found, manifest file {output_path} not created"
        )


def read_manifest_as_lines(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a manifest file where each line is a valid JSON object.

    Args:
        file_path (str):
            Path to the manifest file with JSON lines format.

    Returns:
        List[Dict[str, Any]]:
            List of dictionaries parsed from each line.
    """
    logger.info(f"Reading manifest from {file_path}")

    result = []
    try:
        with open(file_path, "r") as f:
            line_count = 0
            valid_entries = 0
            for line in f:
                line_count += 1
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        entry = json.loads(line)
                        if entry:  # Skip empty objects
                            result.append(entry)
                            valid_entries += 1
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse JSON at line {line_count}: {e}"
                        )

            logger.info(f"Read {valid_entries} valid entries from {line_count} lines")
    except FileNotFoundError:
        logger.error(f"Manifest file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading manifest file {file_path}: {str(e)}")

    return result
