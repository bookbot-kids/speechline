#!/bin/bash

# Skip diffusers import errors since we don't need diffusers for ASR
export DIFFUSERS_VERBOSITY=error

# Run the validation
python speechline/run.py -i /mnt/Store07/Bookbot -c examples/validation_config.json "$@"