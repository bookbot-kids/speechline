#!/bin/bash

SEVEN_DAYS_AGO=$(date -d "3 years ago" +%Y-%m-%d)

# python scripts/download_bookbot_audio.py \
#     --language en\
#     --bucket bookbot-speech \
#     --output-dir ../bookbot_en \
#     --after-date "$SEVEN_DAYS_AGO" \
#     --hf-dataset bookbot/bookbot_en_v3 

# python scripts/download_bookbot_audio.py \
#     --language id\
#     --bucket bookbot-speech \
#     --output-dir ../bookbot_id \
#     --after-date "$SEVEN_DAYS_AGO" \
#     --hf-dataset bookbot/bookbot_id_v4 

python scripts/download_bookbot_audio.py \
    --language sw \
    --bucket bookbot-speech \
    --output-dir ../bookbot_sw \
    --after-date "$SEVEN_DAYS_AGO" \
    --hf-dataset bookbot/bookbot_sw_v3