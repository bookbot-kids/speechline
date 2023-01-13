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
import requests
import json
import os

class AirTable:
    def __init__(self, url: str) -> None:
        """Constructor for AirTable table.

        Args:
            url (str): URL of AirTable table.
        """
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {os.environ['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json",
        }

    def add_records(self, records: List[Dict[str, Any]]) -> bool:
        """Add records to AirTable table.

        Args:
            records (List[Dict[str, Any]]): List of records in AirTable format.

        Returns:
            bool: Whether upload was a success.
        """
        try:
            response = requests.post(
                self.url, headers=self.headers, data=json.dumps({"records": records})
            )
        except Exception:
            return False

        return True if response.ok else False

    def batch_add_records(self, records: List[Dict[str, Any]]) -> bool:
        """Allow batching of record addition due to 10-element limit, then push.

        Args:
            records (List[Dict[str, Any]]): List of records in AirTable format.

        Returns:
            bool: Whether upload was a success.
        """
        BATCH_SIZE = 10
        for idx in range(0, len(records), BATCH_SIZE):
            batch = records[idx : idx + BATCH_SIZE]
            success = self.add_records(batch)
            if not success:
                return success
        return True