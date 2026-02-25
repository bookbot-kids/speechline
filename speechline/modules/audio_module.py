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

from transformers import Pipeline


class AudioModule:
    """
    Base AudioModule. Inherit this class for other audio models.
    An AudioModule should have an inference pipeline,
    and an inference function utilizing the pipeline.

    Args:
        pipeline (Pipeline):
            HuggingFace `transformers` `Pipeline` for inference.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.sampling_rate = self.pipeline.feature_extractor.sampling_rate
