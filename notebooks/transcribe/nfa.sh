python /home/s44504/3b01c699-3670-469b-801f-13880b9cac56/NeMo/tools/nemo_forced_aligner/align.py \
    pretrained_name="nvidia/parakeet-ctc-1.1b" \
    manifest_filepath=manifest-bb.json \
    output_dir=./alignment_output_bb \
    align_using_pred_text=true