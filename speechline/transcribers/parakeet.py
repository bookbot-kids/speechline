import json
import os
import subprocess
import tempfile
from typing import Dict, List, Union

from datasets import Dataset, Audio

from ..utils.io import export_transcripts_json

class ParakeetTranscriber:
    def __init__(self, model_name: str, transcribe_device: str = "cuda") -> None:
        self.model_name = model_name
        self.transcribe_device = transcribe_device
        self.sampling_rate = 16000
        self.batch_size = 16
        
    def predict(
        self,
        dataset: Dataset,
        chunk_length_s: int = 0,
        output_offsets: bool = True,
        return_timestamps: bool = True,
        keep_whitespace: bool = False,
        **kwargs,
    ) -> Union[List[List[Dict[str, Union[str, float]]]], List[str]]:
        # Create temporary directory for manifest and alignments
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = os.path.join(temp_dir, "manifest.json")
            output_dir = os.path.join(temp_dir, "alignment_output")
            
            # Create manifest file
            with open(manifest_path, "w") as f:
                for item in dataset:
                    json.dump({"audio_filepath": item["audio"]["path"]}, f)
                    f.write("\n")
            # Run Parakeet aligner
            cmd = [
                "python",
                "/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/NeMo/tools/nemo_forced_aligner/align.py",
                f"pretrained_name={self.model_name}",
                f"manifest_filepath={manifest_path}",
                f"output_dir={output_dir}",
                f"transcribe_device={self.transcribe_device}",
                f"batch_size={self.batch_size}",
                "align_using_pred_text=true",
            ]

            # Only add chunk length parameter if it's greater than 0
            if chunk_length_s is not None and chunk_length_s > 0:
                cmd.append(f"chunk_len_in_secs={chunk_length_s}")
            
            subprocess.run(cmd, check=True)

            # Read manifest with output paths
            manifest_with_paths = os.path.join(output_dir, os.path.basename(manifest_path).replace('.json', '_with_output_file_paths.json'))
            
            # Read alignment results and format them
            all_offsets = []
            with open(manifest_with_paths) as f:
                for line in f:
                    entry = json.loads(line)
                    
                    # Read word-level CTM file
                    offsets = []
                    with open(entry["words_level_ctm_filepath"]) as ctm_file:
                        for line in ctm_file:
                            # Format: <utt_id> 1 <start_time> <duration> <word>
                            parts = line.strip().split()
                            if len(parts) >= 5:  # Ensure we have all required fields
                                start_time = float(parts[2])
                                duration = float(parts[3])
                                word = parts[4]
                                
                                if not word.strip() and not keep_whitespace:
                                    continue
                                    
                                offsets.append({
                                    "text": word,
                                    "start_time": start_time,
                                    "end_time": start_time + duration
                                })
                    
                    all_offsets.append(offsets)

            return all_offsets
        
        
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test ParakeetTranscriber')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default="nvidia/parakeet-ctc-1.1b", help='Model name')
    args = parser.parse_args()
    
    # Create a simple dataset with one audio file
    dataset = Dataset.from_dict({
        "audio": [args.audio_path],
    }).cast_column("audio", Audio())
    
    # Initialize transcriber and run prediction
    transcriber = ParakeetTranscriber(args.model, transcribe_device="cpu")
    results = transcriber.predict(
        dataset=dataset,
        chunk_length_s=30,
        output_offsets=True,
        return_timestamps=True
    )
    
    # Print results
    print("\nTranscription Results:")
    print("-" * 50)
    for word_data in results[0]:  # First (and only) audio file
        print(f"Word: {word_data['text']:<20} "
              f"Start: {word_data['start_time']:.2f}s "
              f"End: {word_data['end_time']:.2f}s")