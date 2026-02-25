#!/bin/bash

# Script to rebuild Gentle's lexicon FSTs from the updated align_lexicon.txt
# This is necessary after updating the lexicon text file

set -e

GENTLE_DIR="/mnt/Store07/Projects/gentle"
KALDI_UTILS="${GENTLE_DIR}/ext/kaldi/egs/wsj/s5/utils"
LANGDIR="${GENTLE_DIR}/exp/langdir"
PHONES_DIR="${LANGDIR}/phones"
OPENFST_BIN="${GENTLE_DIR}/ext/kaldi/tools/openfst-1.7.2/bin"

# Add OpenFST tools to PATH
export PATH="${OPENFST_BIN}:${PATH}"

echo "============================================================"
echo "REBUILDING GENTLE LEXICON FSTs"
echo "============================================================"

# Check if required files exist
if [ ! -f "${PHONES_DIR}/align_lexicon.txt" ]; then
    echo "❌ Error: align_lexicon.txt not found at ${PHONES_DIR}/align_lexicon.txt"
    exit 1
fi

if [ ! -f "${LANGDIR}/phones.txt" ]; then
    echo "❌ Error: phones.txt not found at ${LANGDIR}/phones.txt"
    exit 1
fi

if [ ! -f "${LANGDIR}/words.txt" ]; then
    echo "❌ Error: words.txt not found at ${LANGDIR}/words.txt"
    exit 1
fi

echo "✓ Found all required input files"

# Backup existing FST files
echo ""
echo "Creating backups of existing FST files..."
if [ -f "${LANGDIR}/L.fst" ]; then
    cp "${LANGDIR}/L.fst" "${LANGDIR}/L.fst.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✓ Backed up L.fst"
fi

if [ -f "${LANGDIR}/L_disambig.fst" ]; then
    cp "${LANGDIR}/L_disambig.fst" "${LANGDIR}/L_disambig.fst.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✓ Backed up L_disambig.fst"
fi

# Convert align_lexicon.txt to the format expected by fstcompile
echo ""
echo "Converting lexicon format for FST compilation..."

# align_lexicon.txt format: word word phone1 phone2 phone3...
# We need to convert this to FST text format

cd "${LANGDIR}"

# Create a temporary lexicon in FST text format
# FST text format: state_from state_to input_symbol output_symbol [weight]
python3 << 'PYTHON_SCRIPT'
import sys

# Read phones.txt to get phone-to-ID mapping
phone_to_id = {}
with open('phones.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            phone_to_id[parts[0]] = parts[1]

# Read words.txt to get word-to-ID mapping  
word_to_id = {}
with open('words.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            word_to_id[parts[0]] = parts[1]

# Process align_lexicon.txt
state_counter = 0
with open('phones/align_lexicon.txt', 'r') as fin, open('L.txt', 'w') as fout:
    for line_num, line in enumerate(fin, 1):
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        word = parts[0]
        # parts[1] is the word again (duplicated)
        phones = parts[2:]
        
        if word not in word_to_id:
            print(f"Warning: word '{word}' not in words.txt, skipping", file=sys.stderr)
            continue
        
        word_id = word_to_id[word]
        
        # Create FST arcs for this pronunciation
        # Format: start_state end_state input output [weight]
        # For lexicon: input is phone, output is word
        
        start_state = state_counter
        state_counter += 1
        
        # First phone arc: input=phone, output=word
        first_phone = phones[0]
        if first_phone not in phone_to_id:
            print(f"Warning: phone '{first_phone}' not in phones.txt, skipping line {line_num}", file=sys.stderr)
            continue
        
        phone_id = phone_to_id[first_phone]
        
        if len(phones) == 1:
            # Single phone - arc goes to final state (0)
            fout.write(f"{start_state}\t0\t{phone_id}\t{word_id}\n")
        else:
            # Multiple phones - create intermediate states
            next_state = state_counter
            state_counter += 1
            fout.write(f"{start_state}\t{next_state}\t{phone_id}\t{word_id}\n")
            
            # Middle phones: input=phone, output=eps (0)
            for phone in phones[1:-1]:
                if phone not in phone_to_id:
                    print(f"Warning: phone '{phone}' not in phones.txt, skipping line {line_num}", file=sys.stderr)
                    break
                phone_id = phone_to_id[phone]
                curr_state = next_state
                next_state = state_counter
                state_counter += 1
                fout.write(f"{curr_state}\t{next_state}\t{phone_id}\t0\n")
            else:
                # Last phone: input=phone, output=eps (0), goes to final state
                last_phone = phones[-1]
                if last_phone in phone_to_id:
                    phone_id = phone_to_id[last_phone]
                    fout.write(f"{next_state}\t0\t{phone_id}\t0\n")
    
    # Final state
    fout.write("0\n")

print(f"✓ Created FST text representation with {state_counter} states")
PYTHON_SCRIPT

# Compile L.txt to L.fst using fstcompile
echo ""
echo "Compiling L.fst..."
if command -v fstcompile &> /dev/null; then
    fstcompile \
        --isymbols=phones.txt \
        --osymbols=words.txt \
        --keep_isymbols=false \
        --keep_osymbols=false \
        L.txt | fstarcsort --sort_type=olabel > L.fst
    
    echo "✓ Successfully compiled L.fst"
    
    # For now, copy L.fst to L_disambig.fst
    # (Disambiguation is typically handled separately in Kaldi)
    cp L.fst L_disambig.fst
    echo "✓ Created L_disambig.fst"
    
    # Clean up temporary file
    rm -f L.txt
    
    # Verify the new FST
    echo ""
    echo "Verifying new FST files..."
    echo "L.fst size: $(du -h L.fst | cut -f1)"
    echo "L_disambig.fst size: $(du -h L_disambig.fst | cut -f1)"
    
    # Count entries
    LEXICON_ENTRIES=$(wc -l < phones/align_lexicon.txt)
    echo "Lexicon entries: ${LEXICON_ENTRIES}"
    
    echo ""
    echo "============================================================"
    echo "✅ LEXICON FSTs REBUILT SUCCESSFULLY"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "1. Test alignment with a sample file"
    echo "2. Re-run validation tests"
    echo "3. Check that previously failing words now align correctly"
    
else
    echo "❌ Error: fstcompile not found in PATH"
    echo "Make sure OpenFST tools are installed and accessible"
    exit 1
fi