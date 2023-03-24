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

from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AudioClassificationPipeline
from transformers.utils import ModelOutput


class AudioMultiLabelClassificationPipeline(AudioClassificationPipeline):
    """
    Subclass of `AudioClassificationPipeline`.
    Performs multi-label audio classification instead of multi-class classification.
    Applies Sigmoid on logits instead of Softmax.
    """

    def _sanitize_parameters(self, **kwargs) -> Tuple[Dict, Dict, Dict]:
        """
        Forces post-processor to return all probabilities.

        Returns:
            Tuple[Dict, Dict, Dict]:
                Tuple consisting of:

                    1. Preprocess parameters (empty).
                    2. Forward parameters (empty).
                    3. Postprocess parameters (`top_k = num_labels`).
        """
        postprocess_params = {"top_k": self.model.config.num_labels}
        return {}, {}, postprocess_params

    def postprocess(self, model_outputs: ModelOutput, **kwargs) -> np.ndarray:
        """
        Applies Sigmoid on logits.

        Args:
            model_outputs (ModelOutput):
                Generic HuggingFace model outputs.

        Returns:
            np.ndarray:
                List of probabilities.
        """
        probs = model_outputs.logits[0]
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(probs).cpu().numpy()
        return scores
