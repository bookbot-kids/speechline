# Create manifest file
python scripts/create_data_explorer_manifest.py \
    --input_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/speechline/bookbot_en/word_level_segmentation/en-ar \
    --output_path /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/speechline/bookbot_en/word_level_segmentation/manifest.jsonl

# Run NVIDIA Data Explorer
python ../NeMo/tools/speech_data_explorer/data_explorer.py /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/speechline/bookbot_en/word_level_segmentation/manifest.jsonl