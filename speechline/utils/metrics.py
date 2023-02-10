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

from itertools import permutations
from typing import Dict, List, Set

import Levenshtein


class PhonemeErrorRate:
    """
    Phoneme-Error Rate metric, with flexibility in lexicon.

    Args:
        lexicon (Dict[str, List[List[str]]]):
            Pronunciation lexicon with word (grapheme) as key,
            and list of valid phoneme-list pronunciations.
    """

    def __init__(self, lexicon: Dict[str, List[List[str]]]) -> None:
        self.lexicon = self._validate_lexicon(lexicon)

    def __call__(
        self, sequences: List[List[str]], predictions: List[List[str]]
    ) -> float:
        """
        Calculates PER given list of ground truth words, predicted phonemes,
        and corresponding lexicon.

        ### Example
        ```pycon title="example_phoneme_error_rate.py"
        >>> lexicon = {
        ...     "hello": [["h", "e", "l", "l", "o"], ["h", "a", "l", "l", "o"]],
        ...     "guy": [["g", "a", "i"]]
        ... }
        >>> per = PhonemeErrorRate(lexicon)
        >>> sequences = [
        ...     ["hello", "hello"],
        ...     ["hello", "guy"]
        ... ]
        >>> predictions = [
        ...     ["h", "e", "l", "l", "o", "b", "e", "l", "l", "o"],
        ...     ["h", "a", "l", "l", "o", "g", "a", "i"]
        ... ]
        >>> per(sequences=sequences, predictions=predictions)
        0.05555555555555555
        ```

        Args:
            sequences (List[List[str]]):
                List of list of ground truth words in a batch.
            predictions (List[List[str]]):
                List of list of predicted phonemes in a batch.

        Raises:
            ValueError: Mistmatch in the number of predictions and sequences.

        Returns:
            float:
                Phoneme error rate.
        """
        if len(sequences) != len(predictions):
            raise ValueError(
                f"Mismatch in the number of predictions ({len(predictions)}) and sequences ({len(sequences)})"  # noqa: E501
            )

        errors, total = 0, 0
        for words, prediction in zip(sequences, predictions):
            measures = self.compute_measures(words, prediction)
            errors += measures["errors"]
            total += measures["total"]
        return errors / total

    def compute_measures(
        self, words: List[str], prediction: List[str]
    ) -> Dict[str, int]:
        """
        Computes the number of phoneme-level errors.

        ### Example
        ```pycon title="example_compute_measures.py"
        >>> lexicon = {
        ...     "hello": [["h", "e", "l", "l", "o"], ["h", "a", "l", "l", "o"]],
        ...     "guy": [["g", "a", "i"]]
        ... }
        >>> words = ["hello", "guy"]
        >>> per = PhonemeErrorRate(lexicon)
        >>> per.compute_measures(
        ...     words,
        ...     prediction=["h", "a", "l", "l", "o", "g", "a", "i"]
        ... )
        {'errors': 0, 'total': 8}
        >>> per.compute_measures(
        ...     words,
        ...     prediction=["h", "a", "l", "a", "i"]
        ... )
        {'errors': 3, 'total': 8}
        >>> per.compute_measures(
        ...     words,
        ...     prediction=["h", "a", "l", "l", "o", "b", "h", "a", "i"]
        ... )
        {'errors': 2, 'total': 8}
        ```

        Args:
            words (List[str]):
                List of ground truth words.
            prediction (List[str]):
                List of predicted phonemes.

        Returns:
            Dict[str, int]:
                A dictionary with number of errors and total number of true phonemes.
        """
        reference = [p for word in words for p in self.lexicon[word][0]]
        stack = self._build_pronunciation_stack(words)

        editops = Levenshtein.editops(reference, prediction)
        # get initial number of errors
        errors = len(editops)

        for tag, i, j in editops:
            # if substitution is a valid phoneme swap, reduce error by one
            if (
                tag == "replace"
                # there are >1 valid phonemes at position
                and len(stack[i]) > 1
                # pair of phoneme is in list of valid phoneme pairs
                and (reference[i], prediction[j]) in permutations(stack[i], 2)
            ):
                errors -= 1

        return {"errors": errors, "total": len(reference)}

    def _build_pronunciation_stack(self, words: List[str]) -> List[Set[str]]:
        """
        Builds a list of expected pronunciation "stack".

        ### Example
        ```pycon title="example_build_pronunciation_stack.py"
        >>> lexicon = {
        ...     "hello": [["h", "e", "l", "l", "o"], ["h", "a", "l", "l", "o"]],
        ...     "guy": [["g", "a", "i"]]
        ... }
        >>> words = ["hello", "guy"]
        >>> per = PhonemeErrorRate(lexicon)
        >>> per._build_pronunciation_stack(words)
        [{'h'}, {'a', 'e'}, {'l'}, {'l'}, {'o'}, {'g'}, {'a'}, {'i'}]
        ```

        Args:
            words (List[str]):
                List of words whose pronunciation stack will be built.

        Returns:
            List[Set[str]]:
                List of possible phonemes of the input words.
        """
        stack = []
        for word in words:
            pronunciations = self.lexicon[word]
            length = len(pronunciations[0])
            word_stack = [
                set(pron[i] for pron in pronunciations) for i in range(length)
            ]
            stack += word_stack
        return stack

    def _validate_lexicon(
        self, lexicon: Dict[str, List[List[str]]]
    ) -> Dict[str, List[List[str]]]:
        """
        Validates lexicon, where all pronunciation variants
        must have the same number of phonemes.

        Args:
            lexicon (Dict[str, List[List[str]]]):
                Pronunciation lexicon with word (grapheme) as key,
                and list of valid phoneme-list pronunciations.

        Raises:
            ValueError: Pronunciation variants have differing phoneme lengths.

        Returns:
            Dict[str, List[List[str]]]:
                Validated lexicon.
        """
        for _, pronunciations in lexicon.items():
            if len(pronunciations) > 1:
                base_length = len(pronunciations[0])
                if not all(len(pron) == base_length for pron in pronunciations):
                    raise ValueError(
                        "Pronunciation variants must have the same number of phonemes!"
                    )
        return lexicon
