python speechline/phoneme_transcribe_and_filter.py \
    --model_path "bookbot/w2v-bert-2.0-bb-libri-cv-giga-dean2zak" \
    --dataset_path "../bookbot_en_training" \
    --language "en" \
    --hf_dataset "bookbot/bookbot_en_v3_parakeet-ctc-1.1b_filtered_phoneme" \
    --log_dir "logs"
