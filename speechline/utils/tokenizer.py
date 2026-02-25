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

from dataclasses import dataclass
from string import punctuation
from typing import List

from nltk.tokenize import TweetTokenizer


@dataclass
class WordTokenizer:
    """
    Basic word-based splitting.
    """

    tokenizer = TweetTokenizer(preserve_case=False)

    def __call__(self, text: str) -> List[str]:
        """
        Splits text into words, ignoring punctuations and case.

        Args:
            text (str):
                Text to tokenize.

        Returns:
            List[str]:
                List of tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        tokens = [token for token in tokens if token not in punctuation]
        return tokens
