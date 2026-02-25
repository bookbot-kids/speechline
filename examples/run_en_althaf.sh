export INPUT_DIR="/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/en-Multi-Word-48kHz"
export OUTPUT_DIR="/home/s44504/3b01c699-3670-469b-801f-13880b9cac56/en-Multi-Word-48kHz-segmented"

python speechline/run.py --input_dir=$INPUT_DIR --output_dir=$OUTPUT_DIR --config="examples/en_althaf.json"
