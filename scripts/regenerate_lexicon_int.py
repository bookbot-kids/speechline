#!/usr/bin/env python3
"""
Regenerate Gentle's align_lexicon.int file from align_lexicon.txt

This script reads the text lexicon and symbol tables (words.txt, phones.txt)
and creates the integer representation used by Kaldi/Gentle.
"""

import sys
from pathlib import Path

def load_symbol_table(symbol_file):
    """Load a Kaldi symbol table (symbol -> ID mapping)"""
    symbols = {}
    with open(symbol_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                symbol, idx = parts
                symbols[symbol] = idx
    return symbols

def regenerate_lexicon_int(lang_dir, graph_dir=None):
    """
    Regenerate align_lexicon.int from align_lexicon.txt
    
    Args:
        lang_dir: Path to langdir directory
        graph_dir: Path to graph_pp directory (optional, defaults to tdnn_7b_chain_online/graph_pp)
    """
    lang_dir = Path(lang_dir)
    
    if graph_dir is None:
        graph_dir = lang_dir.parent / 'tdnn_7b_chain_online' / 'graph_pp'
    else:
        graph_dir = Path(graph_dir)
    
    # Load symbol tables
    print("Loading symbol tables...")
    words_file = graph_dir / 'words.txt'
    phones_file = graph_dir / 'phones.txt'
    
    if not words_file.exists():
        print(f"ERROR: words.txt not found at {words_file}")
        return False
    
    if not phones_file.exists():
        print(f"ERROR: phones.txt not found at {phones_file}")
        return False
    
    word_to_id = load_symbol_table(words_file)
    phone_to_id = load_symbol_table(phones_file)
    
    print(f"  Loaded {len(word_to_id)} words")
    print(f"  Loaded {len(phone_to_id)} phones")
    
    # Read align_lexicon.txt from both locations
    input_files = [
        graph_dir / 'phones' / 'align_lexicon.txt',
        lang_dir / 'phones' / 'align_lexicon.txt'
    ]
    
    for lexicon_txt in input_files:
        if not lexicon_txt.exists():
            print(f"WARNING: {lexicon_txt} not found, skipping")
            continue
        
        lexicon_int = lexicon_txt.with_suffix('.int')
        
        # Backup existing .int file
        if lexicon_int.exists():
            backup_file = lexicon_int.with_suffix('.int.backup')
            print(f"\nBacking up {lexicon_int} to {backup_file}")
            lexicon_int.rename(backup_file)
        
        print(f"\nProcessing {lexicon_txt}")
        print(f"  Output: {lexicon_int}")
        
        # Convert text to int format
        entries_written = 0
        entries_skipped = 0
        
        with open(lexicon_txt, 'r', encoding='utf-8') as fin, \
             open(lexicon_int, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                word = parts[0]
                # parts[1] is duplicate word
                phones = parts[2:]
                
                # Map word to ID
                if word not in word_to_id:
                    if entries_skipped < 10:  # Only show first 10 warnings
                        print(f"  WARNING: Word '{word}' not in words.txt (line {line_num})")
                    entries_skipped += 1
                    continue
                
                word_id = word_to_id[word]
                
                # Map phones to IDs
                phone_ids = []
                skip_entry = False
                for phone in phones:
                    if phone not in phone_to_id:
                        if entries_skipped < 10:
                            print(f"  WARNING: Phone '{phone}' not in phones.txt (line {line_num})")
                        skip_entry = True
                        break
                    phone_ids.append(phone_to_id[phone])
                
                if skip_entry:
                    entries_skipped += 1
                    continue
                
                # Write integer format: word_id word_id phone_id1 phone_id2 ...
                fout.write(f"{word_id} {word_id} {' '.join(phone_ids)}\n")
                entries_written += 1
        
        print(f"  ✓ Wrote {entries_written} entries")
        if entries_skipped > 0:
            print(f"  ⚠ Skipped {entries_skipped} entries (not in symbol tables)")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python regenerate_lexicon_int.py <langdir_path> [graph_dir_path]")
        print()
        print("Example:")
        print("  python regenerate_lexicon_int.py /path/to/gentle/exp/langdir")
        print("  python regenerate_lexicon_int.py /path/to/gentle/exp/langdir /path/to/gentle/exp/tdnn_7b_chain_online/graph_pp")
        sys.exit(1)
    
    lang_dir = sys.argv[1]
    graph_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("REGENERATING align_lexicon.int FILES")
    print("=" * 60)
    
    success = regenerate_lexicon_int(lang_dir, graph_dir)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: align_lexicon.int files regenerated")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test alignment with a sample file")
        print("2. Re-run validation tests to check improvement")
    else:
        print("\n" + "=" * 60)
        print("❌ ERROR: Failed to regenerate files")
        print("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()