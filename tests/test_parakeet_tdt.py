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

import pytest
from datasets import Dataset, Audio


@pytest.mark.skipif(
    not pytest.importorskip("nemo"),
    reason="NeMo toolkit not installed"
)
class TestParakeetTDT:
    """Test suite for Parakeet TDT transcriber."""
    
    def test_transcriber_initialization(self):
        """Test that ParakeetTDTTranscriber can be initialized."""
        from speechline.transcribers import ParakeetTDTTranscriber
        
        transcriber = ParakeetTDTTranscriber(
            model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
            transcriber_device="cpu"
        )
        
        assert transcriber is not None
        assert transcriber.sampling_rate == 16000
        assert transcriber.device == "cpu"
    
    def test_transcriber_predict_text(self, audio_path="tests/test_ml/en-au/en-AU-NatashaNeural_0a0c408337aef825b34f97c35420671a.wav"):
        """Test basic transcription without offsets."""
        from speechline.transcribers import ParakeetTDTTranscriber
        
        transcriber = ParakeetTDTTranscriber(
            model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
            transcriber_device="cpu"
        )
        
        dataset = Dataset.from_dict({
            "audio": [audio_path]
        }).cast_column("audio", Audio(sampling_rate=transcriber.sampling_rate))
        
        results = transcriber.predict(
            dataset,
            chunk_length_s=30,
            output_offsets=False
        )
        
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert len(results[0]) > 0
    
    def test_transcriber_predict_with_offsets(self, audio_path="tests/test_ml/en-au/en-AU-NatashaNeural_0a0c408337aef825b34f97c35420671a.wav"):
        """Test transcription with word-level offsets."""
        from speechline.transcribers import ParakeetTDTTranscriber
        
        transcriber = ParakeetTDTTranscriber(
            model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
            transcriber_device="cpu"
        )
        
        dataset = Dataset.from_dict({
            "audio": [audio_path]
        }).cast_column("audio", Audio(sampling_rate=transcriber.sampling_rate))
        
        results = transcriber.predict(
            dataset,
            chunk_length_s=30,
            output_offsets=True,
            return_timestamps="word"
        )
        
        assert len(results) == 1
        assert isinstance(results[0], list)
        
        if len(results[0]) > 0:
            first_word = results[0][0]
            assert "text" in first_word
            assert "start_time" in first_word
            assert "end_time" in first_word
            assert isinstance(first_word["start_time"], (int, float))
            assert isinstance(first_word["end_time"], (int, float))
            assert first_word["end_time"] >= first_word["start_time"]
    
    def test_transcriber_batch_processing(self):
        """Test batch processing with multiple audio files."""
        from speechline.transcribers import ParakeetTDTTranscriber
        
        audio_paths = [
            "tests/test_ml/en-au/en-AU-NatashaNeural_0a0c408337aef825b34f97c35420671a.wav",
            "tests/test_ml/en-au/en-AU-NatashaNeural_0a0c408337aef825b34f97c35420671b.wav"
        ]
        
        transcriber = ParakeetTDTTranscriber(
            model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
            transcriber_device="cpu"
        )
        
        dataset = Dataset.from_dict({
            "audio": audio_paths
        }).cast_column("audio", Audio(sampling_rate=transcriber.sampling_rate))
        
        results = transcriber.predict(
            dataset,
            chunk_length_s=30,
            output_offsets=False
        )
        
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
    
    def test_transcriber_keep_whitespace(self, audio_path="tests/test_ml/en-au/en-AU-NatashaNeural_0a0c408337aef825b34f97c35420671a.wav"):
        """Test transcription with whitespace preservation."""
        from speechline.transcribers import ParakeetTDTTranscriber
        
        transcriber = ParakeetTDTTranscriber(
            model_checkpoint="nvidia/parakeet-tdt-0.6b-v2",
            transcriber_device="cpu"
        )
        
        dataset = Dataset.from_dict({
            "audio": [audio_path]
        }).cast_column("audio", Audio(sampling_rate=transcriber.sampling_rate))
        
        results_no_ws = transcriber.predict(
            dataset,
            output_offsets=False,
            keep_whitespace=False
        )
        
        results_with_ws = transcriber.predict(
            dataset,
            output_offsets=False,
            keep_whitespace=True
        )
        
        # Results might differ in leading/trailing whitespace
        assert len(results_no_ws) == 1
        assert len(results_with_ws) == 1