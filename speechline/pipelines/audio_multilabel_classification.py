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


import numpy as np
import torch
from transformers import AudioClassificationPipeline


class MultiLabelAudioClassificationPipeline(AudioClassificationPipeline):
    """
    Subclass of `AudioClassificationPipeline`.
    Pads/truncates audio array to maximum length before performing audio classification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def _sanitize_parameters(self, top_k=None, **kwargs):
        # No parameters on this pipeline right now
        postprocess_params = {}
        if top_k is not None:
            if top_k > self.model.config.num_labels:
                top_k = self.model.config.num_labels
            postprocess_params["top_k"] = top_k
        postprocess_params["top_k"] = self.model.config.num_labels
        return {}, {}, postprocess_params


    def postprocess(self, model_outputs, top_k=None):

        probs = model_outputs.logits[0]
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(probs).cpu().numpy()


        # scores, ids = scores.topk(self.model.config.num_labels)
            
        # scores = scores.cpu().numpy().tolist()
        # ids = ids.tolist()
        

        return scores
