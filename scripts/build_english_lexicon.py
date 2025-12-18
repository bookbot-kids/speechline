#!/usr/bin/env python3
"""
Build English Lexicon Database

Consolidates multiple lexicon sources into a single SQLite database.
Handles deduplication and phoneme conversion (IPA ↔ ARPA ↔ Artemes).

Usage:
    # Basic build (all sources)
    python scripts/build_english_lexicon.py
    
    # With Common Voice TSV
    python scripts/build_english_lexicon.py --cv-path /path/to/validated.tsv
    
    # With transcript files (NEW)
    python scripts/build_english_lexicon.py --transcript-dirs /mnt/Bookbot "/mnt/Common Voice/clips"
    
    # Full build with all sources
    python scripts/build_english_lexicon.py --cv-path validated.tsv --transcript-dirs /mnt/Bookbot
    
    # Custom output
    python scripts/build_english_lexicon.py --output data/custom_lexicon.db
"""

import sqlite3
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Iterator
from collections import defaultdict
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechline.phonetics import normalize_ipa, ipa_to_arpa, ipa_to_artemes, arpa_to_ipa
from speechline.utils.g2p import g2p_en


class LexiconBuilder:
    """Builds consolidated English lexicon database from multiple sources."""
    
    def __init__(self, db_path: str = "data/english_lexicon.db"):
        self.db_path = db_path
        self.conn = None
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def create_database(self):
        """Create SQLite database with schema."""
        db_file = Path(self.db_path)
        
        # Check if database already exists
        if db_file.exists():
            print(f"📂 Using existing database: {self.db_path}")
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            print("✓ Database connection established\n")
            return
        
        # Create new database only if it doesn't exist
        print(f"Creating new database: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create schema
        self.conn.executescript("""
            CREATE TABLE english (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE,
                ipa TEXT NOT NULL,
                arpa TEXT,
                arteme TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX idx_word ON english(word);
            
            CREATE TABLE import_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                words_added INTEGER DEFAULT 0,
                pronunciations_added INTEGER DEFAULT 0,
                duplicates_skipped INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        self.conn.commit()
        print("✓ Database created\n")
    
    def validate_word(self, word: str) -> bool:
        """Validate word (no dashes, spaces, or multiple words)."""
        if not word or len(word) == 0:
            return False
        if '-' in word or ' ' in word:
            return False
        return True
    
    def process_pronunciations(self, ipa_list: List[str]) -> Dict[str, str]:
        """
        Convert IPA pronunciations to ARPA and Arteme with deduplication.
        
        Args:
            ipa_list: List of IPA pronunciations (may contain duplicates)
        
        Returns:
            Dict with 'ipa', 'arpa', 'arteme' keys (space/pipe delimited)
        """
        # Normalize and deduplicate IPA
        unique_ipa = []
        seen_ipa = set()
        
        for ipa in ipa_list:
            if not ipa or not ipa.strip():
                continue
            
            try:
                normalized = normalize_ipa(ipa.strip())
                if normalized and normalized not in seen_ipa:
                    unique_ipa.append(normalized)
                    seen_ipa.add(normalized)
            except Exception as e:
                print(f"  Warning: Failed to normalize IPA '{ipa}': {e}")
                continue
        
        if not unique_ipa:
            return {'ipa': '', 'arpa': '', 'arteme': ''}
        
        # Convert each IPA to ARPA and Arteme
        arpa_set = set()
        arteme_set = set()
        
        for ipa in unique_ipa:
            try:
                # Generate ARPA
                arpa = ipa_to_arpa(ipa)
                if arpa and arpa.strip():
                    arpa_set.add(arpa.strip())
                
                # Generate Arteme
                arteme = ipa_to_artemes(ipa)
                if arteme and arteme.strip():
                    arteme_set.add(arteme.strip())
            except Exception as e:
                print(f"  Warning: Conversion failed for IPA '{ipa}': {e}")
                continue
        
        # Format for storage
        return {
            'ipa': ' '.join(sorted(unique_ipa)),
            'arpa': ' | '.join(sorted(arpa_set)) if arpa_set else '',
            'arteme': ' '.join(sorted(arteme_set)) if arteme_set else ''
        }
    
    def insert_or_merge_word(self, word: str, ipa_variants: List[str], source: str):
        """
        Insert new word or merge pronunciations if word exists.
        
        Args:
            word: Word to insert/merge
            ipa_variants: List of IPA pronunciations
            source: Data source name
        """
        # Validate word
        if not self.validate_word(word):
            self.stats[source]['invalid_words'] += 1
            return
        
        # Process pronunciations
        pronunciations = self.process_pronunciations(ipa_variants)
        
        if not pronunciations['ipa']:
            self.stats[source]['errors'] += 1
            return
        
        # Check if word exists
        existing = self.conn.execute(
            "SELECT * FROM english WHERE word = ?", (word,)
        ).fetchone()
        
        if existing is None:
            # New word
            try:
                self.conn.execute("""
                    INSERT INTO english (word, ipa, arpa, arteme)
                    VALUES (?, ?, ?, ?)
                """, (word, pronunciations['ipa'], pronunciations['arpa'], pronunciations['arteme']))
                
                self.stats[source]['words_added'] += 1
                self.stats[source]['pronunciations_added'] += len(ipa_variants)
            except Exception as e:
                print(f"  Error inserting word '{word}': {e}")
                self.stats[source]['errors'] += 1
        else:
            # Merge with existing
            existing_ipa = set(existing['ipa'].split())
            new_ipa = set(pronunciations['ipa'].split())
            
            if not new_ipa.issubset(existing_ipa):
                # There are new pronunciations to add
                merged_ipa_list = list(existing_ipa | new_ipa)
                merged = self.process_pronunciations(merged_ipa_list)
                
                try:
                    self.conn.execute("""
                        UPDATE english
                        SET ipa = ?, arpa = ?, arteme = ?
                        WHERE word = ?
                    """, (merged['ipa'], merged['arpa'], merged['arteme'], word))
                    
                    new_count = len(new_ipa - existing_ipa)
                    self.stats[source]['pronunciations_added'] += new_count
                except Exception as e:
                    print(f"  Error merging word '{word}': {e}")
                    self.stats[source]['errors'] += 1
            else:
                self.stats[source]['duplicates_skipped'] += 1
    
    def import_processed_csv(self, csv_path: str = "data/english_words_processed.csv"):
        """Import from english_words_processed.csv."""
        source = "processed_csv"
        print(f"Importing from {csv_path}...")
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"  ✗ File not found: {csv_path}")
            return
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    word = row['word'].lower().strip()
                    ipa = row['ipa'].strip()
                    
                    self.insert_or_merge_word(word, [ipa], source)
                
                self.conn.commit()
                self._print_stats(source)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats[source]['errors'] += 1
    
    def import_worduniversal(self, csv_path: str = "data/worduniversal_en_full.csv"):
        """Import from worduniversal_en_full.csv with semicolon-delimited IPA."""
        source = "worduniversal"
        print(f"Importing from {csv_path}...")
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"  ✗ File not found: {csv_path}")
            return
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    word = row['word'].lower().strip()
                    
                    # Parse lexicons field (semicolon-delimited IPA)
                    lexicons_str = row.get('lexicons', '')
                    if not lexicons_str or lexicons_str == 'lexicons':
                        continue
                    
                    # Split by semicolon and clean
                    ipa_variants = [ipa.strip() for ipa in lexicons_str.split(';') if ipa.strip()]
                    
                    if ipa_variants:
                        self.insert_or_merge_word(word, ipa_variants, source)
                
                self.conn.commit()
                self._print_stats(source)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats[source]['errors'] += 1
    
    def import_dict_file(self, dict_path: str, dict_type: str, source_name: str):
        """
        Import from .dict file (tab-separated).
        
        Args:
            dict_path: Path to .dict file
            dict_type: 'ipa' or 'arpa'
            source_name: Name for statistics
        """
        print(f"Importing from {dict_path}...")
        
        dict_file = Path(dict_path)
        if not dict_file.exists():
            print(f"  ✗ File not found: {dict_path}")
            return
        
        try:
            with open(dict_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Split by tab or multiple spaces
                    parts = re.split(r'\t+|\s{2,}', line, maxsplit=1)
                    if len(parts) != 2:
                        continue
                    
                    word, phonemes = parts
                    word = word.lower().strip()
                    phonemes = phonemes.strip()
                    
                    if dict_type == 'arpa':
                        # Convert ARPA to IPA
                        try:
                            ipa = arpa_to_ipa(phonemes)
                            if ipa:
                                self.insert_or_merge_word(word, [ipa], source_name)
                        except Exception as e:
                            if line_num % 10000 == 0:
                                print(f"  Warning: Line {line_num}: {e}")
                            self.stats[source_name]['errors'] += 1
                    else:
                        # Already IPA
                        self.insert_or_merge_word(word, [phonemes], source_name)
                
                self.conn.commit()
                self._print_stats(source_name)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats[source_name]['errors'] += 1
    
    def import_common_voice(self, tsv_path: str):
        """
        Import words from Common Voice validated.tsv.
        Uses G2P for unknown words and handles plurals intelligently.
        """
        source = "common_voice"
        print(f"Scanning Common Voice: {tsv_path}...")
        
        tsv_file = Path(tsv_path)
        if not tsv_file.exists():
            print(f"  ✗ File not found: {tsv_path}")
            return
        
        try:
            # Extract unique words
            unique_words = set()
            
            with open(tsv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                
                for row in reader:
                    sentence = row.get('sentence', '')
                    if not sentence:
                        continue
                    
                    # Tokenize and clean
                    words = re.findall(r'\b[a-z]+\b', sentence.lower())
                    unique_words.update(words)
            
            print(f"  Found {len(unique_words):,} unique words")
            
            # Process each word
            processed = 0
            added = 0
            plurals = 0
            g2p_used = 0
            
            for word in sorted(unique_words):
                if not self.validate_word(word):
                    continue
                
                processed += 1
                
                # Check if word already exists in DB
                existing = self.conn.execute(
                    "SELECT id FROM english WHERE word = ?", (word,)
                ).fetchone()
                
                if existing:
                    continue
                
                # Check for plural handling (word ends with 's')
                if word.endswith('s') and len(word) > 2:
                    base_word = word[:-1]
                    base_existing = self.conn.execute(
                        "SELECT ipa FROM english WHERE word = ?", (base_word,)
                    ).fetchone()
                    
                    if base_existing:
                        # Add 's' or 'z' to base pronunciations
                        base_ipas = base_existing['ipa'].split()
                        new_ipas = []
                        
                        for base_ipa in base_ipas:
                            # Determine if final phoneme is voiced
                            last_char = base_ipa[-1] if base_ipa else ''
                            # Common voiceless endings: p, t, k, f, θ, s, ʃ, tʃ
                            voiceless = last_char in 'ptkfθsʃ'
                            suffix = 's' if voiceless else 'z'
                            new_ipas.append(base_ipa + suffix)
                        
                        self.insert_or_merge_word(word, new_ipas, f"{source}_plural")
                        plurals += 1
                        added += 1
                        continue
                
                # Use G2P for unknown words
                try:
                    ipa_list = g2p_en(word)
                    if ipa_list:
                        # g2p_en returns list of phoneme sequences
                        ipa_str = ' '.join(ipa_list)
                        self.insert_or_merge_word(word, [ipa_str], f"{source}_g2p")
                        g2p_used += 1
                        added += 1
                except Exception as e:
                    if processed % 1000 == 0:
                        print(f"  G2P error for '{word}': {e}")
                    self.stats[source]['errors'] += 1
                
                # Progress update
                if processed % 5000 == 0:
                    print(f"  Processed {processed:,} words, added {added:,} new words")
            
            self.conn.commit()
            
            print(f"  ✓ Processed: {processed:,} words")
            print(f"  ✓ Added: {added:,} new words")
            print(f"  ✓ Plurals handled: {plurals:,}")
            print(f"  ✓ G2P used: {g2p_used:,}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats[source]['errors'] += 1
    
    def scan_transcript_files(self, base_path: str) -> Iterator[Path]:
        """
        Scan directory for .txt transcript files.
        For Bookbot: scans en-* subdirectories recursively.
        For Common Voice clips: scans directory directly.
        
        Args:
            base_path: Path to directory to scan
            
        Yields:
            Path objects for each .txt file found
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            print(f"  ✗ Directory not found: {base_path}")
            return
        
        # Check if this looks like a Bookbot directory (has en-* subdirs)
        en_dirs = list(base_path.glob("en-*"))
        
        if en_dirs:
            # Bookbot structure: scan en-* subdirectories recursively
            for en_dir in en_dirs:
                if en_dir.is_dir():
                    for txt_file in en_dir.rglob("*.txt"):
                        if txt_file.is_file():
                            yield txt_file
        else:
            # Common Voice clips structure: scan directory directly
            for txt_file in base_path.glob("*.txt"):
                if txt_file.is_file():
                    yield txt_file
    
    def extract_words_from_transcript(self, txt_path: Path) -> Set[str]:
        """
        Extract unique words from a transcript file.
        Reads first line only (text transcript).
        
        Args:
            txt_path: Path to transcript file
            
        Returns:
            Set of lowercase words
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            if not first_line:
                return set()
            
            # Extract words (letters only, lowercase)
            words = re.findall(r'\b[a-z]+\b', first_line.lower())
            return set(words)
            
        except Exception as e:
            print(f"  Error reading {txt_path}: {e}")
            return set()
    
    def import_from_transcripts(self, transcript_dirs: List[str]):
        """
        Import words from transcript files in specified directories.
        Uses G2P for unknown words and handles plurals intelligently.
        
        Args:
            transcript_dirs: List of directory paths to scan
        """
        source = "transcripts"
        print(f"Scanning transcript directories...")
        
        if not transcript_dirs:
            print("  ✗ No transcript directories provided")
            return
        
        try:
            # Phase 1: Collect all unique words from all transcript files
            all_words = set()
            files_scanned = 0
            
            for transcript_dir in transcript_dirs:
                print(f"  Scanning: {transcript_dir}")
                dir_files = 0
                
                for txt_file in self.scan_transcript_files(transcript_dir):
                    words = self.extract_words_from_transcript(txt_file)
                    all_words.update(words)
                    dir_files += 1
                    files_scanned += 1
                    
                    if files_scanned % 1000 == 0:
                        print(f"    Scanned {files_scanned} files, found {len(all_words):,} unique words so far...")
                
                print(f"    Found {dir_files} transcript files")
            
            print(f"\n  Total files scanned: {files_scanned:,}")
            print(f"  Total unique words found: {len(all_words):,}")
            
            # Phase 2: Process each unique word
            processed = 0
            added = 0
            plurals = 0
            g2p_used = 0
            new_words_list = []
            
            sorted_words = sorted(all_words)
            
            print(f"\n  Checking words against database...")
            
            # First pass: collect all new words
            for word in sorted_words:
                if not self.validate_word(word):
                    continue
                
                processed += 1
                
                # Check if word already exists in DB
                existing = self.conn.execute(
                    "SELECT id FROM english WHERE word = ?", (word,)
                ).fetchone()
                
                if existing:
                    continue
                
                # Word doesn't exist - will be added
                new_words_list.append(word)
            
            # Display new words that will be added
            if new_words_list:
                print(f"\n  📝 NEW WORDS TO ADD ({len(new_words_list):,} words):")
                print("  " + "-" * 58)
                
                # Display in columns (5 words per line)
                for i in range(0, len(new_words_list), 5):
                    words_chunk = new_words_list[i:i+5]
                    formatted = "  " + "  ".join(f"{w:<12}" for w in words_chunk)
                    print(formatted)
                
                print("  " + "-" * 58)
                print()
            else:
                print(f"\n  ✓ No new words to add (all words already in lexicon)")
                print()
                return
            
            # Second pass: process and add the new words
            print(f"  Adding new words to database...")
            
            for word in new_words_list:
                # Check for plural handling (word ends with 's')
                if word.endswith('s') and len(word) > 2:
                    base_word = word[:-1]
                    base_existing = self.conn.execute(
                        "SELECT ipa FROM english WHERE word = ?", (base_word,)
                    ).fetchone()
                    
                    if base_existing:
                        # Add 's' or 'z' to base pronunciations
                        base_ipas = base_existing['ipa'].split()
                        new_ipas = []
                        
                        for base_ipa in base_ipas:
                            # Determine if final phoneme is voiced
                            last_char = base_ipa[-1] if base_ipa else ''
                            # Common voiceless endings: p, t, k, f, θ, s, ʃ, tʃ
                            voiceless = last_char in 'ptkfθsʃ'
                            suffix = 's' if voiceless else 'z'
                            new_ipas.append(base_ipa + suffix)
                        
                        self.insert_or_merge_word(word, new_ipas, f"{source}_plural")
                        plurals += 1
                        added += 1
                        continue
                
                # Use G2P for unknown words
                try:
                    ipa_list = g2p_en(word)
                    if ipa_list:
                        # g2p_en returns list of phoneme sequences
                        ipa_str = ' '.join(ipa_list)
                        self.insert_or_merge_word(word, [ipa_str], f"{source}_g2p")
                        g2p_used += 1
                        added += 1
                except Exception as e:
                    if processed % 1000 == 0:
                        print(f"  G2P error for '{word}': {e}")
                    self.stats[source]['errors'] += 1
                
                # Progress update
                if processed % 1000 == 0:
                    print(f"  Processed {processed:,} words, added {added:,} new words")
            
            self.conn.commit()
            
            print(f"\n  ✓ Processed: {processed:,} words")
            print(f"  ✓ Added: {added:,} new words")
            print(f"  ✓ Plurals handled: {plurals:,}")
            print(f"  ✓ G2P used: {g2p_used:,}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.stats[source]['errors'] += 1
    
    def validate_transcripts(self, transcript_dirs: List[str]) -> Dict[str, Set[str]]:
        """
        Validate transcript directories and report missing words.
        
        Args:
            transcript_dirs: List of directory paths to scan
            
        Returns:
            Dictionary with 'missing_words' and 'existing_words' sets
        """
        print("="*70)
        print("LEXICON VALIDATION - Checking for Missing Words")
        print("="*70)
        print()
        
        if not transcript_dirs:
            print("  ✗ No transcript directories provided")
            return {'missing_words': set(), 'existing_words': set()}
        
        try:
            # Phase 1: Collect all unique words from all transcript files
            all_words = set()
            files_scanned = 0
            dir_stats = {}
            
            for transcript_dir in transcript_dirs:
                print(f"📂 Scanning: {transcript_dir}")
                dir_files = 0
                dir_words = set()
                
                for txt_file in self.scan_transcript_files(transcript_dir):
                    words = self.extract_words_from_transcript(txt_file)
                    dir_words.update(words)
                    all_words.update(words)
                    dir_files += 1
                    files_scanned += 1
                    
                    if files_scanned % 1000 == 0:
                        print(f"  Scanned {files_scanned:,} files, found {len(all_words):,} unique words...")
                
                dir_stats[transcript_dir] = {
                    'files': dir_files,
                    'words': len(dir_words)
                }
                print(f"  ✓ Files: {dir_files:,}")
                print(f"  ✓ Unique words: {len(dir_words):,}")
                print()
            
            print(f"📊 Total files scanned: {files_scanned:,}")
            print(f"📊 Total unique words: {len(all_words):,}")
            print()
            
            # Phase 2: Check which words are in database
            print("🔍 Checking words against database...")
            print()
            
            missing_words = set()
            existing_words = set()
            invalid_words = set()
            
            for word in sorted(all_words):
                if not self.validate_word(word):
                    invalid_words.add(word)
                    continue
                
                # Check if word exists in DB
                existing = self.conn.execute(
                    "SELECT id FROM english WHERE word = ?", (word,)
                ).fetchone()
                
                if existing:
                    existing_words.add(word)
                else:
                    missing_words.add(word)
            
            # Phase 3: Report results
            print("="*70)
            print("VALIDATION RESULTS")
            print("="*70)
            print()
            
            # Summary statistics
            total_valid = len(existing_words) + len(missing_words)
            coverage_pct = (len(existing_words) / total_valid * 100) if total_valid > 0 else 0
            
            print(f"📈 Coverage Statistics:")
            print(f"  Total valid words found: {total_valid:,}")
            print(f"  Words in database: {len(existing_words):,} ({coverage_pct:.2f}%)")
            print(f"  Missing from database: {len(missing_words):,}")
            print(f"  Invalid words (filtered): {len(invalid_words):,}")
            print()
            
            # Missing words details
            if missing_words:
                print("❌ MISSING WORDS FROM DATABASE:")
                print("-" * 70)
                
                # Sort by length and alphabetically
                sorted_missing = sorted(missing_words, key=lambda w: (len(w), w))
                
                # Display in columns (6 words per line)
                for i in range(0, len(sorted_missing), 6):
                    words_chunk = sorted_missing[i:i+6]
                    formatted = "  " + "  ".join(f"{w:<12}" for w in words_chunk)
                    print(formatted)
                
                print("-" * 70)
                print()
                
                # Check for potential plurals
                plural_candidates = []
                for word in sorted_missing:
                    if word.endswith('s') and len(word) > 2:
                        base_word = word[:-1]
                        if base_word in existing_words:
                            plural_candidates.append((word, base_word))
                
                if plural_candidates:
                    print(f"ℹ️  Potential Plurals ({len(plural_candidates)} words):")
                    print("   These can be auto-generated from base words:")
                    for plural, base in plural_candidates[:10]:
                        print(f"   - {plural} (from {base})")
                    if len(plural_candidates) > 10:
                        print(f"   ... and {len(plural_candidates) - 10} more")
                    print()
                
                # Categorize by word type
                proper_nouns = [w for w in sorted_missing if w[0].isupper()]
                technical = [w for w in sorted_missing if len(w) > 10]
                
                if proper_nouns:
                    print(f"ℹ️  Proper Nouns ({len(proper_nouns)}):")
                    print("   ", ", ".join(proper_nouns[:10]))
                    if len(proper_nouns) > 10:
                        print(f"   ... and {len(proper_nouns) - 10} more")
                    print()
                
                if technical:
                    print(f"ℹ️  Long/Technical Words ({len(technical)}):")
                    print("   ", ", ".join(technical[:10]))
                    if len(technical) > 10:
                        print(f"   ... and {len(technical) - 10} more")
                    print()
            else:
                print("✅ EXCELLENT: All words are in the database!")
                print("   100% coverage - no missing words found")
                print()
            
            # Invalid words (for information)
            if invalid_words and len(invalid_words) < 50:
                print("ℹ️  Invalid Words (filtered out):")
                print("   Contains dashes, spaces, or special characters:")
                print("   ", ", ".join(sorted(invalid_words)[:20]))
                if len(invalid_words) > 20:
                    print(f"   ... and {len(invalid_words) - 20} more")
                print()
            
            print("="*70)
            print()
            
            # Suggestions
            if missing_words:
                print("💡 Next Steps:")
                print()
                print("To add missing words to the database, run:")
                print()
                print("  python scripts/build_english_lexicon.py \\")
                print("      --transcript-dirs \\")
                for dir_path in transcript_dirs:
                    print(f"          \"{dir_path}\" \\")
                print()
                print("This will:")
                print("  1. Use G2P (gruut) to generate pronunciations")
                print("  2. Auto-handle plurals from base words")
                print("  3. Add IPA → ARPA → Artemes conversions")
                print("  4. Update the database")
                print()
            
            return {
                'missing_words': missing_words,
                'existing_words': existing_words,
                'invalid_words': invalid_words
            }
            
        except Exception as e:
            print(f"  ✗ Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return {'missing_words': set(), 'existing_words': set(), 'invalid_words': set()}
    
    def _print_stats(self, source: str):
        """Print statistics for a source."""
        stats = self.stats[source]
        print(f"  ✓ Words added: {stats['words_added']:,}")
        print(f"  ✓ Pronunciations: {stats['pronunciations_added']:,}")
        if stats['duplicates_skipped'] > 0:
            print(f"  ✓ Duplicates skipped: {stats['duplicates_skipped']:,}")
        if stats['invalid_words'] > 0:
            print(f"  ✗ Invalid words: {stats['invalid_words']:,}")
        if stats['errors'] > 0:
            print(f"  ✗ Errors: {stats['errors']:,}")
        print()
    
    def save_statistics(self):
        """Save import statistics to database."""
        for source, stats in self.stats.items():
            self.conn.execute("""
                INSERT INTO import_stats (source, words_added, pronunciations_added, 
                                         duplicates_skipped, errors)
                VALUES (?, ?, ?, ?, ?)
            """, (source, stats['words_added'], stats['pronunciations_added'],
                  stats['duplicates_skipped'], stats['errors']))
        
        self.conn.commit()
    
    def print_summary(self):
        """Print final summary statistics."""
        total_words = self.conn.execute("SELECT COUNT(*) FROM english").fetchone()[0]
        
        # Count total pronunciations
        total_ipa = 0
        for row in self.conn.execute("SELECT ipa FROM english"):
            total_ipa += len(row[0].split())
        
        print("\n" + "="*60)
        print("DATABASE BUILD COMPLETE")
        print("="*60)
        print(f"Database: {self.db_path}")
        print(f"Total unique words: {total_words:,}")
        print(f"Total IPA pronunciations: {total_ipa:,}")
        print(f"Average pronunciations per word: {total_ipa/total_words:.2f}")
        print("="*60 + "\n")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Build English Lexicon Database from multiple sources'
    )
    parser.add_argument(
        '--output',
        default='data/english_lexicon.db',
        help='Output database path (default: data/english_lexicon.db)'
    )
    parser.add_argument(
        '--cv-path',
        help='Path to Common Voice validated.tsv (optional)'
    )
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip Common Voice import'
    )
    parser.add_argument(
        '--transcript-dirs',
        nargs='+',
        help='Paths to transcript directories (Bookbot, Common Voice clips, etc.)'
    )
    parser.add_argument(
        '--transcript-only',
        action='store_true',
        help='Only import from transcript directories (skip CSV and lexicon imports)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate mode: Check for missing words without importing them'
    )
    
    args = parser.parse_args()
    
    # Create builder
    builder = LexiconBuilder(db_path=args.output)
    
    try:
        # Create database
        builder.create_database()
        
        # Validation mode - check for missing words
        if args.validate:
            if not args.transcript_dirs:
                print("ERROR: --validate requires --transcript-dirs")
                print()
                print("Example:")
                print("  python scripts/build_english_lexicon.py \\")
                print("      --validate \\")
                print("      --transcript-dirs \\")
                print("          \"/mnt/Store07/Bookbot\" \\")
                print("          \"/mnt/Store07/Common Voice/en/clips\"")
                sys.exit(1)
            
            builder.validate_transcripts(args.transcript_dirs)
            return
        
        # Skip standard imports if --transcript-only is specified
        if not args.transcript_only:
            # Import from all sources
            print("PHASE 1: Import from processed CSV")
            print("-" * 60)
            builder.import_processed_csv("data/english_words_processed.csv")
            
            print("PHASE 2: Import from Word Universal")
            print("-" * 60)
            builder.import_worduniversal("data/worduniversal_en_full.csv")
            
            print("PHASE 3: Import from lexicon dictionaries")
            print("-" * 60)
            builder.import_dict_file("data/lexicons/english_us_mfa.dict", "ipa", "mfa_us")
            builder.import_dict_file("data/lexicons/english_us_arpa.dict", "arpa", "arpa_us")
            builder.import_dict_file("data/lexicons/english_uk_mfa.dict", "ipa", "mfa_uk")
            builder.import_dict_file("data/lexicons/english_india_mfa.dict", "ipa", "mfa_india")
            builder.import_dict_file("data/lexicons/english_nigeria_mfa.dict", "ipa", "mfa_nigeria")
            builder.import_dict_file("data/lexicons/english_nonnative_mfa.dict", "ipa", "mfa_nonnative")
            
            # Import from Common Voice (optional)
            if not args.skip_cv:
                print("PHASE 4: Import from Common Voice")
                print("-" * 60)
                if args.cv_path:
                    builder.import_common_voice(args.cv_path)
                else:
                    default_cv = Path("/mnt/Store07/Common Voice/en/validated.tsv")
                    if default_cv.exists():
                        builder.import_common_voice(str(default_cv))
                    else:
                        print("  Skipping Common Voice (no path provided)")
                        print()
        else:
            print("📋 TRANSCRIPT-ONLY MODE: Skipping standard CSV and lexicon imports")
            print("-" * 60)
            print()
        
        # Import from transcript files (optional)
        if args.transcript_dirs:
            phase_num = "PHASE 5" if not args.transcript_only else "PHASE 1"
            print(f"{phase_num}: Import from Transcript Files")
            print("-" * 60)
            builder.import_from_transcripts(args.transcript_dirs)
        
        # Save statistics and print summary
        builder.save_statistics()
        builder.print_summary()
        
    finally:
        builder.close()


if __name__ == "__main__":
    main()