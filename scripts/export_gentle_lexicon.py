#!/usr/bin/env python3
"""
Export English Lexicon Database to Gentle Format

Converts the lexicon database to Gentle's align_lexicon.txt format:
- Format: word word phoneme_sequence_with_position_markers
- Position markers: _B (beginning), _I (internal), _E (end), _S (standalone)
- Uses lowercase CMU ARPAbet phones without stress markers
"""

import sqlite3
import sys
import re
from pathlib import Path
from typing import List, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class GentleExporter:
    """Export lexicon database to Gentle format"""
    
    def __init__(self, db_path: str = "data/english_lexicon.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.exported_count = 0
        self.skipped_count = 0
        
    def strip_stress_markers(self, arpa_phone: str) -> str:
        """Remove stress markers (0,1,2) from ARPA phone
        
        Examples:
            AE1 -> AE
            EY0 -> EY
            T -> T
        """
        return re.sub(r'[012]$', '', arpa_phone)
    
    def add_position_markers(self, phones: List[str]) -> str:
        """Add position markers to phone sequence
        
        Args:
            phones: List of ARPA phones (stress markers already removed)
            
        Returns:
            Space-separated phones with position markers
            
        Examples:
            ['HH', 'EH', 'L', 'OW'] -> 'hh_B eh_I l_I ow_E'
            ['AH'] -> 'ah_S'
            ['G', 'OW'] -> 'g_B ow_E'
        """
        if not phones:
            return ""
        
        if len(phones) == 1:
            # Single phone: use _S (standalone)
            return f"{phones[0].lower()}_S"
        
        elif len(phones) == 2:
            # Two phones: _B and _E
            return f"{phones[0].lower()}_B {phones[1].lower()}_E"
        
        else:
            # Three or more: _B, _I (for all middle), _E
            result = [f"{phones[0].lower()}_B"]
            for phone in phones[1:-1]:
                result.append(f"{phone.lower()}_I")
            result.append(f"{phones[-1].lower()}_E")
            return " ".join(result)
    
    def convert_arpa_to_gentle(self, arpa_string: str) -> str:
        """Convert ARPA pronunciation to Gentle format
        
        Args:
            arpa_string: Pipe-delimited ARPA pronunciations from database
                        e.g., "K AE1 T | K AE0 T"
        
        Returns:
            Gentle-formatted phone sequence with position markers
            e.g., "k_B ae_I t_E"
        """
        # Get first pronunciation (most common)
        pronunciations = arpa_string.split("|")
        if not pronunciations:
            return ""
        
        # Take first pronunciation, strip and split
        arpa_phones = pronunciations[0].strip().split()
        
        # Remove stress markers
        phones = [self.strip_stress_markers(phone) for phone in arpa_phones]
        
        # Add position markers
        return self.add_position_markers(phones)
    
    def validate_word(self, word: str) -> bool:
        """Check if word is valid for Gentle lexicon
        
        Args:
            word: Word to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Skip empty or very long words
        if not word or len(word) > 50:
            return False
        
        # Skip words with special characters (except apostrophes, hyphens)
        # Allow: letters, numbers, apostrophes, hyphens
        if not re.match(r"^[a-zA-Z0-9'\-]+$", word):
            return False
        
        return True
    
    def export(self, output_path: str = "data/align_lexicon.txt"):
        """Export database to Gentle format
        
        Args:
            output_path: Output file path
        """
        print(f"Exporting lexicon from {self.db_path}")
        print(f"Output: {output_path}")
        print()
        
        # Query all words with ARPA pronunciations
        cursor = self.conn.execute("""
            SELECT word, arpa 
            FROM english 
            WHERE arpa IS NOT NULL AND arpa != ''
            ORDER BY word
        """)
        
        with open(output_path, 'w') as f:
            # Add header comment
            f.write("# Gentle Lexicon exported from speechline english_lexicon.db\n")
            f.write("# Format: word word phoneme_sequence_with_position_markers\n")
            f.write("# Position markers: _B (begin), _I (internal), _E (end), _S (standalone)\n")
            f.write("#\n\n")
            
            for word, arpa_string in cursor:
                # Validate word
                if not self.validate_word(word):
                    self.skipped_count += 1
                    continue
                
                # Convert ARPA string to list of pronunciations
                arpa_pronunciations = [p.strip() for p in arpa_string.split("|") if p.strip()]
                
                # Export all pronunciation variations
                for arpa_pron in arpa_pronunciations:
                    # Convert to Gentle format
                    phones = arpa_pron.split()
                    phones_no_stress = [self.strip_stress_markers(p) for p in phones]
                    gentle_phones = self.add_position_markers(phones_no_stress)
                    
                    if gentle_phones:
                        # Write entry: word word phoneme_sequence
                        f.write(f"{word.lower()} {word.lower()} {gentle_phones}\n")
                        self.exported_count += 1
        
        print(f"\n{'='*60}")
        print(f"EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"Total entries exported: {self.exported_count:,}")
        print(f"Words skipped (invalid): {self.skipped_count:,}")
        print(f"Output file: {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
        print()
        
    def show_samples(self, output_path: str, n: int = 10):
        """Show sample entries from exported file
        
        Args:
            output_path: Path to exported file
            n: Number of samples to show
        """
        print(f"\nSample entries from {output_path}:")
        print("-" * 60)
        
        with open(output_path, 'r') as f:
            # Skip header comments
            lines = [line for line in f if not line.startswith('#') and line.strip()]
            
            # Show first n entries
            for i, line in enumerate(lines[:n]):
                print(f"{i+1}. {line.rstrip()}")
        
        print("-" * 60)
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export English Lexicon Database to Gentle format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default settings (exports all pronunciation variations)
  python scripts/export_gentle_lexicon.py
  
  # Export with custom output path
  python scripts/export_gentle_lexicon.py -o gentle_lexicon.txt
  
  # Custom database path
  python scripts/export_gentle_lexicon.py --db custom.db
        """
    )
    
    parser.add_argument(
        '--db',
        default='data/english_lexicon.db',
        help='Path to lexicon database (default: data/english_lexicon.db)'
    )
    parser.add_argument(
        '-o', '--output',
        default='data/align_lexicon.txt',
        help='Output file path (default: data/align_lexicon.txt)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of sample entries to display (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create exporter and export
    exporter = GentleExporter(args.db)
    
    try:
        exporter.export(args.output)
        exporter.show_samples(args.output, args.samples)
    finally:
        exporter.close()


if __name__ == '__main__':
    main()