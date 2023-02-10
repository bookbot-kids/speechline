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

    def __call__(self, words: List[str], prediction: List[str]) -> float:
        """
        Calculates PER given list of ground truth words, predicted phonemes,
        and corresponding lexicon.

        ### Example
        ```pycon title="example_phoneme_error_rate.py"
        >>> lexicon = {
        ...     "hello": [["h", "e", "l", "l", "o"], ["h", "a", "l", "l", "o"]],
        ...     "guy": [["g", "a", "i"]]
        ... }
        >>> words = ["hello", "guy"]
        >>> per = PhonemeErrorRate(lexicon)
        >>> per(words, prediction=["h", "a", "l", "l", "o", "g", "a", "i"])
        0.0
        >>> per(words, prediction=["h", "a", "l", "a", "i"])
        0.375
        >>> per(words, prediction=["h", "a", "l", "l", "o", "b", "h", "a", "i"])
        0.25
        ```

        Args:
            words (List[str]):
                List of ground truth words.
            prediction (List[str]):
                List of predicted phonemes.

        Returns:
            float:
                Phoneme-error rate.
        """
        errs, idx = 0, 0

        reference = [p for word in words for p in self.lexicon[word][0]]
        stack = self._build_pronunciation_stack(words)

        opcodes = Levenshtein.opcodes(reference, prediction)
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != "equal":
                # if there happens to be multiple valid phoneme swaps in current index
                if i1 == idx and idx < len(stack) and len(stack[idx]) > 1:
                    # get current substring
                    expected = reference[i1:i2]
                    predicted = prediction[j1:j2]

                    for phn in stack[idx]:
                        # remove valid phoneme swaps from substring
                        if phn in expected:
                            expected.remove(phn)
                        if phn in predicted:
                            predicted.remove(phn)

                    # rematch remaining phonemes, and update costs accordingly
                    editops = Levenshtein.editops(expected, predicted)
                    errs += len(editops)
                else:
                    # calculate basic error count of A/D/S
                    errs += self._calculate_error(tag, i1, i2, j1, j2)
            idx += i2 - i1

        return errs / len(reference)

    def _calculate_error(self, tag: str, i1: int, i2: int, j1: int, j2: int) -> int:
        """
        Calculates the total number of:

        - Additions
        - Deletions
        - Substitutions

        Args:
            tag (str):
                Opcode tags `{'replace', 'delete', 'insert', 'equal'}`.
            i1 (int):
                Start index in sequence `a`.
            i2 (int):
                End index in sequence `a`.
            j1 (int):
                Start index in sequence `b`.
            j2 (int):
                End index in sequence `b`.

        Returns:
            int:
                Total number of errors.
        """
        len_ori = i2 - i1
        len_pred = j2 - j1
        errs = 0
        # if deletion, see how many original phonemes were deleted
        if tag == "delete":
            errs += len_ori
        # if insertion, see how many extra phonemes were inserted
        elif tag == "insert":
            errs += len_pred
        elif tag == "replace":
            # original phonemes were replaced by much longer phonemes
            # punish by the number of substituted new phonemes
            # as the additional ones count as "additions"
            if len_ori <= len_pred:
                errs += len_pred
            # otherwise, if the substitution phonemes are shorter than original
            # punish by how many original phonemes were expected
            # as the "unreplaced" ones count as "deletions"
            else:
                errs += len_ori
        return errs

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
