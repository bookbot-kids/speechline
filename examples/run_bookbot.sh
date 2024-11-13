export INPUT_DIR="bookbot_en/"
export OUTPUT_DIR="bookbot_en/word_level_segmentation"

python speechline/run.py --input_dir=$INPUT_DIR --output_dir="bookbot_en/word_level_segmentation/" --config="examples/bb_config_word.json"

# python speechline/run.py --input_dir=$INPUT_DIR --output_dir="bookbot_en/phoneme_level_segmentation/" --config="examples/bb_config.json"


# python -m speechline.transcribers.parakeet --audio_path speechline/bookbot_en/en-ar/guest_78702eb3-bcfc-48f3-9f31-e2e2e5733958_1722800344486.wav --model nvidia/parakeet-ctc-1.1b