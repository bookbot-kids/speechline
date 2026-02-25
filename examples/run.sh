export INPUT_DIR="../bookbot_en_sample/"
export OUTPUT_DIR="../bookbot_en_sample_w2v_word_segment"

python speechline/run.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR --config="examples/bb_config.json"