# python scripts/transcriber.py "/mnt/Store05/Datasets/English/TTSDataset/Enhanced-Segmented" \
#     --num-gpus 1 \
#     --workers-per-gpu 2 \
#     --batch-size 200

python scripts/transcriber.py "/mnt/Store04/Datasets/English/Podcast" \
    --num-gpus 1 \
    --workers-per-gpu 2 \
    --batch-size 20
