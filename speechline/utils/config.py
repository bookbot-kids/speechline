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


class Config:
    def __init__(self, path: str) -> None:
        self.config = json.load(open(path))
        self.__dict__.update(**self.config)
        self.validate_config()

    def validate_config(self) -> None:
        classifier_lang = self.models["classifier"].keys()
        transcriber_lang = self.models["transcriber"].keys()
        missing_lang = (
            set(self.languages) - set(classifier_lang) - set(transcriber_lang)
        )
        if len(missing_lang) > 0:
            raise AttributeError(
                f"ML models must be provided for language {missing_lang}"
            )
