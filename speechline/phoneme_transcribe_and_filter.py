import argparse
import logging
import os
import sys
import torch
from datetime import datetime
from typing import List
from datasets import Dataset, load_dataset, Audio  # Add Audio to imports
from azure.cosmos import CosmosClient
from lexikos import Lexicon as Lexicos
from g2p_id import G2p
from gruut import sentences
import pandas as pd
from glob import glob
from pathlib import Path
import csv
from tqdm import tqdm  # Add this import at the top with other imports

sys.path.append("../")
from speechline.transcribers import Wav2Vec2Transcriber
from speechline.segmenters import PhonemeOverlapSegmenter
from speechline.utils.tokenizer import WordTokenizer

COSMOS_DB_KEY = os.getenv('COSMOS_DB_KEY')
COSMOS_URL = "https://bookbot.documents.azure.com:443/"

class Cosmos:
    def __init__(self, url, key, database_name):
        self.client = CosmosClient(url, credential=key, enable_diagnostics_logging=False)
        self.database = self.client.get_database_client(database_name)
        self.word_container = self.database.get_container_client("WordUniversal")

    def get_lexicon(self, language_code):
        """Retrieve the lexicon for a specific language from CosmosDB."""
        query = f'SELECT * FROM c WHERE c.language = "{language_code}" and not is_defined(c.deletedAt)'
        query_iterable = self.word_container.query_items(
            query=query,
            partition_key="default",
            max_item_count=10000,
        )
        lexicon = {}
        for item in query_iterable:
            if "lexicons" in item:
                lexicon[item["word"]] = set(item["lexicons"])
        return lexicon

class Lexicon(PhonemeOverlapSegmenter):
    def __init__(self, language, cosmos_client):
        self.language = language
        self.cosmos_lexicon = cosmos_client.get_lexicon(language)
        self._init_g2p(language)   
        
        # If language is english, use Lexicos
        if language == "en":
            lexicos_lexicon = Lexicos()
            for k, v in lexicos_lexicon.items():
                self.cosmos_lexicon[k] = self.cosmos_lexicon[k].union(set(v)) if k in self.cosmos_lexicon else set(v)
        super().__init__(self.cosmos_lexicon)
        
    def gruut_g2p(self, text: str) -> List[str]:
        phonemes = []
        for words in sentences(text, lang=self.language):
            for word in words:
                if word.is_major_break or word.is_minor_break:
                    phonemes.append(word.text)
                elif word.phonemes:
                    phonemes.append(" ".join(word.phonemes))
        return "".join(phonemes)
    
    def g2p_id(self, text: str) -> List[str]:
        g2p = G2p()
        results = g2p(text)
        results = [phoneme for word in results for phoneme in word ]
        return " ".join(results)
    
    def _init_g2p(self, language):
        if language == "en":
            self.g2p = self.gruut_g2p
        elif language == "id":
            self.g2p = self.g2p_id
        elif language == "sw":
            self.g2p = self.gruut_g2p
        
    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        return text  
    
    def _generate_combinations(self, ground_truth: List[str]) -> List[List[str]]:
        """
        Generate all possible phoneme combinations for a given word.

        Args:
            ground_truth (List[str]):
                List of words.

        Returns:
            List[List[str]]:
                List of phoneme combinations.
        """
        combinations = []
        for word in ground_truth:
            normalized_word = self._normalize_text(word)
            if normalized_word in self.lexicon:
                phonemes = self.lexicon[normalized_word]
            else:
                phonemes = self.g2p(normalized_word)
            combinations.append(phonemes)
        return combinations

def check_phoneme_match(phoneme_transcript, ground_truth):
    """
    Check if each phoneme in transcript exists in corresponding ground truth set
    
    Args:
        phoneme_transcript (List[str]): List of phonemes from transcript
        ground_truth (List[Set[str]]): List of sets containing valid phonemes
    
    Returns:
        bool: True if all phonemes match their ground truth sets
    Example:
        phoneme_transcript = ['ɪn', 'ðɛɹ', 'deɪ']
        ground_truth = [{'ɪ n', 'ɪ ŋ'}, {'ð ɛ ɹ', 'ð ɛ r'}, {'d e ɪ', 'd e ɪ'}]
    """
    # Check lengths match first
    if len(phoneme_transcript) != len(ground_truth):
        return False
        
    # Check each phoneme against its ground truth set
    for phoneme, valid_phonemes in zip(phoneme_transcript, ground_truth):
        # Remove spaces from transcript phoneme for comparison
        valid_phonemes_no_spaces = {p.replace(" ", "") for p in valid_phonemes}
        
        if phoneme not in valid_phonemes_no_spaces:
            return False
            
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Phoneme transcription and filtering script')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the wav2vec model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the input dataset')
    parser.add_argument('--language', type=str, required=True,
                      help='Language code (e.g., "en", "id", "sw")')
    parser.add_argument('--hf_dataset', type=str, required=True,
                      help='Name for the output HuggingFace dataset')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory for storing log files')
    return parser.parse_args()

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    log_file = os.path.join(log_dir, f'{timestamp}_phoneme_transcribe_and_filter.txt')
    
    # Disable noisy third-party loggers
    noisy_loggers = [
        'azure.core.pipeline.policies.http_logging_policy',
        'azure.cosmos',
        'azure.core',
        'azure.cosmos.http_logger'  # Added this specific logger as well
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_tsv(path: str) -> str:
    """Join text transcripts of TSV annotation."""
    with open(path) as fd:
        rows = csv.reader(fd, delimiter="\t", quotechar='"')
        return " ".join(row[2] for row in rows)

def create_dataset_from_dir(input_dir: str) -> Dataset:
    """Creates dataset from local directory containing wav and tsv files.
    
    Args:
        input_dir (str): Path to input directory containing wav/tsv files
        
    Returns:
        Dataset: Dataset containing audio and text columns
    """
    audios = glob(f"{input_dir}/**/*.wav")
    df = pd.DataFrame({"audio": audios})
    
    # Extract metadata from paths
    df["id"] = df["audio"].apply(lambda x: x.split("/")[-1].replace(".wav", ""))
    df["language"] = df["audio"].apply(lambda x: x.split("/")[-2])
    df["speaker"] = df["audio"].apply(lambda x: x.split("/")[-1].split("_")[0])
    df["text"] = df["audio"].apply(lambda x: parse_tsv(Path(x).with_suffix(".tsv")))
    
    return Dataset.from_pandas(df).cast_column("audio", Audio())

def transcribe_audio(datum, transcriber):
    """Process a single audio sample with the transcriber"""
    try:
        audio_data = datum["audio"].copy()
        result = transcriber.pipeline(audio_data, chunk_length_s=30)
        return {
            "phoneme_transcript": result["text"],
            # "list_phoneme_transcript": result["text"].split()
        }
    except Exception as e:
        # Return None values to indicate failed transcription
        return {
            "phoneme_transcript": None,
            "list_phoneme_transcript": None
        }

def filter_transcription(datum, lexicon, tokenizer):
    """Filter function to check if transcription matches ground truth"""
    # Skip samples that failed transcription
    if not datum["phoneme_transcript"]:
        return False
        
    list_ground_truth_phonemes = lexicon._generate_combinations(tokenizer(datum["text"]))
    return check_phoneme_match(datum["phoneme_transcript"].split(), list_ground_truth_phonemes)

def main():
    args = parse_args()
    logger = setup_logging(args.log_dir)
    
    try:
        # Initialize components
        transcriber = Wav2Vec2Transcriber(args.model_path, None)
        cosmos_client = Cosmos(COSMOS_URL, COSMOS_DB_KEY, "Bookbot")
        lexicon = Lexicon(args.language, cosmos_client)
        tokenizer = WordTokenizer()
        
        # Load dataset
        if os.path.isdir(args.dataset_path):
            dataset = create_dataset_from_dir(args.dataset_path)
        else:
            dataset = load_dataset(args.dataset_path, split="train", num_proc=os.cpu_count())
            
        logger.info(f"Loaded {len(dataset)} dataset samples")
        
        # Step 1: Transcribe audio using map
        logger.info("Transcribing audio samples...")
        transcripts = transcriber.predict(dataset)
        assert len(transcripts) == len(dataset)
        transcript_column_name = "phoneme_transcript"
        dataset = dataset.add_column(transcript_column_name, transcripts)
        
        # Step 2: Filter using the transcriptions
        logger.info("Filtering samples...")
        filtered_dataset = dataset.filter(
            lambda x: filter_transcription(x, lexicon, tokenizer),
            desc="Filtering samples"
        )
        
        logger.info(f"Successfully filtered {len(filtered_dataset)} samples from {len(dataset)} samples")
        
        # Save locally first
        output_dir = "../bookbot_en_training_filtered_hf_dataset"
        os.makedirs(output_dir, exist_ok=True)
        filtered_dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved locally to {output_dir}")
        
        # Push to HuggingFace Hub
        try:
            filtered_dataset.push_to_hub(args.hf_dataset, private=True)
            logger.info(f"Dataset successfully pushed to {args.hf_dataset}")
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace Hub: {str(e)}")
            logger.info(f"Dataset is still saved locally at {output_dir}")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    main()