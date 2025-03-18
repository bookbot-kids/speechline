MODEL_NAME=bookbot/bookbot/w2v-bert-2.0-bb-libri-cv-giga-dean2zak

# python scripts/filter_dataset.py \
#     --dataset_name bookbot/bookbot_en_phonemes \
#     --model_name $MODEL_NAME \
#     --torch_dtype bfloat16 \
#     --repo_id bookbot/bookbot_en_phonemes_filtered_w2v-bert-2.0


# python scripts/filter_dataset.py \
#     --dataset_name bookbot/common_voice_16_1_en_wav2vec2-conformer \
#     --model_name $MODEL_NAME \
#     --torch_dtype bfloat16 \
#     --repo_id bookbot/common_voice_16_1_en_wav2vec2-conformer_filtered_w2v-bert-2.0

# python scripts/filter_dataset.py \
#     --dataset_name bookbot/gigaspeech_wav2vec2-conformer \
#     --model_name $MODEL_NAME \
#     --torch_dtype bfloat16 \
#     --repo_id bookbot/gigaspeech_wav2vec2-conformer_filtered_w2v-bert-2.0

python scripts/filter_dataset.py \
    --dataset_name bookbot/bookbot_en_v3 \
    --model_name $MODEL_NAME \
    --torch_dtype bfloat16 \
    --repo_id bookbot/bookbot_en_v3_filtered