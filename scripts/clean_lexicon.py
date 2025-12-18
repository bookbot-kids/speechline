#!/usr/bin/env python3
"""
Automated Lexicon Cleanup Script

Identifies and removes truncated pronunciations from the English lexicon database.
Respects exception words with legitimate short pronunciations.

Usage:
    # Dry run (preview changes)
    python scripts/clean_lexicon_auto.py --dry-run
    
    # Actually clean the database
    python scripts/clean_lexicon_auto.py
    
    # Export report before cleaning
    python scripts/clean_lexicon_auto.py --export data/cleanup_report.csv
"""

import sqlite3
import sys
import csv
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speechline.phonetics import MULTI_CHAR_SEQUENCES, IPA_TO_ARTEME


class AutoLexiconCleaner:
    """Automatically clean truncated pronunciations from lexicon database"""
    
    # Exception words: legitimate short pronunciations that should not be removed
    EXCEPTION_WORDS = {
        'purr', 'shhh', 'sshh', 'they\'re', 'thought', 'through', 'tshe', 
        'urgh', 'were', 'whir', 'whirred', 'aah', 'aaj', 'aarre', 'addes', 
        'ade\'s', 'ah\'', 'ahh', 'ai\'', 'aide\'s', 'aides', 'aie', 'aight', 
        'ain\'t', 'air', 'aired', 'aires', 'aisle', 'corps', 'coups', 
        'aughts', 'eighth', 'eights', 'furred', 'higher', 'mooooooooo', 
        'oughts', 'sceaux', 'weighs'
    }
    
    # IPA vowel phonemes for validation
    IPA_VOWELS = set('aɑæeɛiɪoɔɒuʊəɚɝʌ')
    IPA_DIPHTHONGS = {'aɪ', 'aʊ', 'eɪ', 'oɪ', 'oʊ', 'ɔɪ'}
    
    def __init__(self, db_path: str = "data/english_lexicon.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Statistics
        self.stats = {
            'total_words': 0,
            'words_checked': 0,
            'words_skipped': 0,
            'words_updated': 0,
            'words_deleted': 0,
            'pronunciations_removed': 0,
            'pronunciations_kept': 0
        }
        
        # Track what will be removed
        self.to_remove = []
    
    def count_ipa_phonemes(self, ipa_string: str) -> int:
        """Count phonemes in IPA string using greedy matching"""
        if not ipa_string or not ipa_string.strip():
            return 0
        
        ipa_string = ipa_string.replace(' ', '')
        phoneme_count = 0
        pos = 0
        
        while pos < len(ipa_string):
            matched = False
            for seq in sorted(MULTI_CHAR_SEQUENCES, key=len, reverse=True):
                if ipa_string[pos:pos+len(seq)] == seq:
                    phoneme_count += 1
                    pos += len(seq)
                    matched = True
                    break
            
            if not matched:
                char = ipa_string[pos]
                if char in IPA_TO_ARTEME or char in self.IPA_VOWELS:
                    phoneme_count += 1
                pos += 1
        
        return phoneme_count
    
    def count_arpa_phonemes(self, arpa_string: str) -> int:
        """Count phonemes in ARPA string (space-separated)"""
        if not arpa_string or not arpa_string.strip():
            return 0
        # ARPA phonemes are space-separated
        phonemes = arpa_string.strip().split()
        return len(phonemes)
    
    def count_arteme_phonemes(self, arteme_string: str) -> int:
        """Count phonemes in Arteme string (individual characters)"""
        if not arteme_string or not arteme_string.strip():
            return 0
        # Arteme uses single-character phonemes from 12-class system
        return len(arteme_string.replace(' ', ''))
    
    def should_skip_word(self, word: str) -> bool:
        """Check if word should be skipped from cleaning"""
        # Skip exception words
        if word in self.EXCEPTION_WORDS:
            return True
        
        # Skip contractions
        if word.startswith("'"):
            return True
        
        # Skip pure symbols
        if not any(c.isalnum() for c in word):
            return True
        
        # Skip pure numbers
        if word.replace('-', '').replace('.', '').replace(',', '').isdigit():
            return True
        
        return False
    
    def is_truncated(self, word: str, phoneme_string: str, format_type: str = 'ipa') -> Tuple[bool, float]:
        """
        Check if pronunciation is truncated for any format.
        
        Args:
            word: The word being checked
            phoneme_string: The phoneme representation (IPA, ARPA, or Arteme)
            format_type: 'ipa', 'arpa', or 'arteme'
        
        Returns (is_truncated, ratio)
        """
        if not phoneme_string or not phoneme_string.strip():
            return True, 0.0
        
        word_length = len(word)
        
        # Count phonemes based on format
        if format_type == 'ipa':
            phoneme_count = self.count_ipa_phonemes(phoneme_string)
        elif format_type == 'arpa':
            phoneme_count = self.count_arpa_phonemes(phoneme_string)
        elif format_type == 'arteme':
            phoneme_count = self.count_arteme_phonemes(phoneme_string)
        else:
            return False, 0.0
        
        if phoneme_count == 0:
            return True, 0.0
        
        ratio = phoneme_count / word_length
        
        # Critical truncation cases
        if word_length > 7 and phoneme_count <= 2:
            return True, ratio
        
        if word_length > 4 and phoneme_count == 1:
            return True, ratio
        
        # High severity truncation
        if ratio < 0.3:
            return True, ratio
        
        # Medium severity truncation
        if ratio < 0.5:
            return True, ratio
        
        return False, ratio
    
    def scan_database(self) -> List[Dict]:
        """Scan database for truncated pronunciations"""
        print(f"Scanning database: {self.db_path}")
        print("=" * 60)
        
        cursor = self.conn.execute("""
            SELECT word, ipa, arpa, arteme 
            FROM english 
            WHERE ipa IS NOT NULL AND ipa != ''
            ORDER BY word
        """)
        
        for row in cursor:
            word = row['word']
            ipa_string = row['ipa']
            arpa_string = row['arpa'] or ''
            arteme_string = row['arteme'] or ''
            
            self.stats['total_words'] += 1
            
            # Skip exception words
            if self.should_skip_word(word):
                self.stats['words_skipped'] += 1
                continue
            
            self.stats['words_checked'] += 1
            
            # Parse pronunciations (space-delimited)
            ipa_list = [p.strip() for p in ipa_string.split() if p.strip()]
            arpa_list = [p.strip() for p in arpa_string.split('|') if p.strip()]
            arteme_list = [p.strip() for p in arteme_string.split() if p.strip()]
            
            # Validate each format independently (they don't always correspond!)
            # Collect truncated pronunciations for each format separately
            ipa_truncated_indices = set()
            arpa_truncated_indices = set()
            arteme_truncated_indices = set()
            
            for idx, ipa in enumerate(ipa_list):
                ipa_trunc, ipa_ratio = self.is_truncated(word, ipa, 'ipa')
                if ipa_trunc:
                    ipa_truncated_indices.add(idx)
                    self.to_remove.append({
                        'word': word,
                        'format': 'ipa',
                        'ipa': ipa,
                        'arpa': arpa_list[idx] if idx < len(arpa_list) else '',
                        'arteme': arteme_list[idx] if idx < len(arteme_list) else '',
                        'ipa_ratio': ipa_ratio,
                        'ratio': ipa_ratio,
                        'word_length': len(word),
                        'phoneme_count': self.count_ipa_phonemes(ipa)
                    })
            
            for idx, arpa in enumerate(arpa_list):
                arpa_trunc, arpa_ratio = self.is_truncated(word, arpa, 'arpa')
                if arpa_trunc:
                    arpa_truncated_indices.add(idx)
                    self.to_remove.append({
                        'word': word,
                        'format': 'arpa',
                        'ipa': ipa_list[idx] if idx < len(ipa_list) else '',
                        'arpa': arpa,
                        'arteme': arteme_list[idx] if idx < len(arteme_list) else '',
                        'arpa_ratio': arpa_ratio,
                        'ratio': arpa_ratio,
                        'word_length': len(word),
                        'phoneme_count': self.count_arpa_phonemes(arpa)
                    })
            
            for idx, arteme in enumerate(arteme_list):
                arteme_trunc, arteme_ratio = self.is_truncated(word, arteme, 'arteme')
                if arteme_trunc:
                    arteme_truncated_indices.add(idx)
                    self.to_remove.append({
                        'word': word,
                        'format': 'arteme',
                        'ipa': ipa_list[idx] if idx < len(ipa_list) else '',
                        'arpa': arpa_list[idx] if idx < len(arpa_list) else '',
                        'arteme': arteme,
                        'arteme_ratio': arteme_ratio,
                        'ratio': arteme_ratio,
                        'word_length': len(word),
                        'phoneme_count': self.count_arteme_phonemes(arteme)
                    })
            
            # Track if word has any good pronunciations left in IPA
            if ipa_truncated_indices and len(ipa_truncated_indices) < len(ipa_list):
                self.stats['pronunciations_kept'] += (len(ipa_list) - len(ipa_truncated_indices))
        
        print(f"Words in database: {self.stats['total_words']:,}")
        print(f"Words checked: {self.stats['words_checked']:,}")
        print(f"Words skipped (exceptions): {self.stats['words_skipped']:,}")
        print(f"Truncated pronunciations found: {len(self.to_remove):,}")
        print()
        
        return self.to_remove
    
    def clean_database(self, dry_run: bool = False):
        """Remove truncated pronunciations from database"""
        if not self.to_remove:
            print("No truncated pronunciations to remove")
            return
        
        print(f"{'DRY RUN: ' if dry_run else ''}Cleaning database...")
        print("=" * 60)
        
        if dry_run:
            print("\n📋 Pronunciations to be removed:\n")
            print(f"{'Word':<20} {'Format':<8} {'IPA':<25} {'ARPA':<35} {'Arteme':<25} {'Ratio':<10}")
            print("-" * 120)
        
        # Group removals by word and format
        removals_by_word = defaultdict(lambda: {'ipa': set(), 'arpa': set(), 'arteme': set(), 'entries': []})
        for entry in self.to_remove:
            word = entry['word']
            format_type = entry['format']
            removals_by_word[word]['entries'].append(entry)
            
            # Track values to remove for each format
            if format_type == 'ipa' and entry['ipa']:
                removals_by_word[word]['ipa'].add(entry['ipa'])
            elif format_type == 'arpa' and entry['arpa']:
                removals_by_word[word]['arpa'].add(entry['arpa'])
            elif format_type == 'arteme' and entry['arteme']:
                removals_by_word[word]['arteme'].add(entry['arteme'])
        
        words_preserved = 0
        
        for word, removal_data in removals_by_word.items():
            # Get current pronunciations
            cursor = self.conn.execute(
                "SELECT ipa, arpa, arteme FROM english WHERE word = ?", (word,)
            )
            result = cursor.fetchone()
            
            if not result:
                continue
            
            ipa_string = result['ipa']
            arpa_string = result['arpa'] or ''
            arteme_string = result['arteme'] or ''
            
            # Parse into lists
            ipa_list = [p.strip() for p in ipa_string.split() if p.strip()]
            arpa_list = [p.strip() for p in arpa_string.split('|') if p.strip()]
            arteme_list = [p.strip() for p in arteme_string.split() if p.strip()]
            
            # Skip words with only one pronunciation in IPA and no ARPA/Arteme removals
            if len(ipa_list) == 1 and not removal_data['arpa'] and not removal_data['arteme']:
                self.stats['words_skipped'] += 1
                continue
            
            # Remove truncated pronunciations from each format independently
            new_ipa_list = [ipa for ipa in ipa_list if ipa not in removal_data['ipa']]
            new_arpa_list = [arpa for arpa in arpa_list if arpa not in removal_data['arpa']]
            new_arteme_list = [arteme for arteme in arteme_list if arteme not in removal_data['arteme']]
            
            removed_count = len(removal_data['ipa']) + len(removal_data['arpa']) + len(removal_data['arteme'])
            self.stats['pronunciations_removed'] += removed_count
            
            # Show details in dry run
            if dry_run:
                for entry in removal_data['entries']:
                    format_type = entry['format']
                    ipa_display = entry.get('ipa', '')
                    arpa_display = entry.get('arpa', '')
                    arteme_display = entry.get('arteme', '')
                    ratio = entry.get('ratio', 0)
                    
                    print(f"{word:<20} {format_type.upper():<8} {ipa_display:<25} {arpa_display:<35} {arteme_display:<25} {ratio:.2f}")
            
            if not new_ipa_list:
                # NEVER delete words - just skip and preserve them
                words_preserved += 1
                if not dry_run:
                    print(f"  ⚠️  Preserved word '{word}' (would have no pronunciations)")
                continue
            
            # Update with remaining pronunciations
            new_ipa_string = ' '.join(new_ipa_list)
            new_arpa_string = ' | '.join(new_arpa_list) if new_arpa_list else ''
            new_arteme_string = ' '.join(new_arteme_list) if new_arteme_list else ''
            
            if not dry_run:
                self.conn.execute(
                    "UPDATE english SET ipa = ?, arpa = ?, arteme = ? WHERE word = ?",
                    (new_ipa_string, new_arpa_string, new_arteme_string, word)
                )
            self.stats['words_updated'] += 1
        
        if not dry_run:
            self.conn.commit()
            print(f"\n✓ Database updated")
            if words_preserved > 0:
                print(f"✓ {words_preserved} words preserved (would have no pronunciations)")
        else:
            print(f"\n✓ Dry run complete (no changes made)")
            if words_preserved > 0:
                print(f"✓ {words_preserved} words would be preserved (no valid pronunciations)")
        
        print()
    
    def export_report(self, output_path: str):
        """Export detailed report of what will be removed"""
        print(f"Exporting report to: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'word', 'format', 'ipa', 'arpa', 'arteme', 'ratio'
            ])
            
            for entry in sorted(self.to_remove, key=lambda x: (x['word'], x['format'], x['ratio'])):
                writer.writerow([
                    entry['word'],
                    entry['format'],
                    entry.get('ipa', ''),
                    entry.get('arpa', ''),
                    entry.get('arteme', ''),
                    f"{entry['ratio']:.3f}"
                ])
        
        print(f"✓ Exported {len(self.to_remove):,} entries")
        print()
    
    def print_summary(self):
        """Print final statistics"""
        print("=" * 60)
        print("CLEANUP SUMMARY")
        print("=" * 60)
        print(f"Total words in database:         {self.stats['total_words']:6,}")
        print(f"Words checked:                   {self.stats['words_checked']:6,}")
        print(f"Words skipped (exceptions/single): {self.stats['words_skipped']:6,}")
        print(f"Words updated:                   {self.stats['words_updated']:6,}")
        print(f"Pronunciations removed:          {self.stats['pronunciations_removed']:6,}")
        print(f"Pronunciations kept:             {self.stats['pronunciations_kept']:6,}")
        print("=" * 60)
        print("\nNote: Words with only one pronunciation are preserved.")
        print("      Words that would have no pronunciations are never deleted.")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automatically clean truncated pronunciations from lexicon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be removed
  python scripts/clean_lexicon_auto.py --dry-run
  
  # Export report before cleaning
  python scripts/clean_lexicon_auto.py --export data/cleanup_report.csv --dry-run
  
  # Actually clean the database
  python scripts/clean_lexicon_auto.py
  
  # Clean with export
  python scripts/clean_lexicon_auto.py --export data/cleanup_report.csv
        """
    )
    
    parser.add_argument(
        '--db',
        default='data/english_lexicon.db',
        help='Path to lexicon database (default: data/english_lexicon.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying database'
    )
    parser.add_argument(
        '--export',
        help='Export cleanup report to CSV file'
    )
    
    args = parser.parse_args()
    
    # Create cleaner
    cleaner = AutoLexiconCleaner(args.db)
    
    try:
        # Scan for truncated pronunciations
        cleaner.scan_database()
        
        # Export report if requested
        if args.export:
            cleaner.export_report(args.export)
        
        # Clean database
        cleaner.clean_database(dry_run=args.dry_run)
        
        # Print summary
        cleaner.print_summary()
        
    finally:
        cleaner.close()


if __name__ == '__main__':
    main()