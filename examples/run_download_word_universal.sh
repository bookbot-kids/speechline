#!/bin/bash

# Script to download WordUniversal data from Cosmos DB as CSV
# Make sure COSMOS_DB_KEY environment variable is set before running

# Download all languages
python scripts/download_word_universal.py --output worduniversal_all.csv

# Or download specific language (uncomment as needed):
# python scripts/download_word_universal.py --output worduniversal_en.csv --language en
# python scripts/download_word_universal.py --output worduniversal_id.csv --language id
# python scripts/download_word_universal.py --output worduniversal_sw.csv --language sw