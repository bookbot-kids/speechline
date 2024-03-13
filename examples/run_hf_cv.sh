python speechline/run_hf.py \
    --dataset_name mozilla-foundation/common_voice_16_1 \
    --dataset_config en \
    --dataset_split train \
    --audio_column_name audio \
    --text_column_name sentence \
    --output_dir training \
    --config examples/cv_config.json