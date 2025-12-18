#!/usr/bin/env python3
"""
Query English Lexicon Database

Utilities for querying and exporting from the English lexicon database.

Usage:
    python scripts/query_lexicon.py lookup hello
    python scripts/query_lexicon.py stats
    python scripts/query_lexicon.py export --format arpa --output output.dict
"""

import sqlite3
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List


class LexiconQuery:
    """Query interface for English lexicon database."""
    
    def __init__(self, db_path: str = "data/english_lexicon.db"):
        self.db_path = db_path
        
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def lookup(self, word: str) -> Optional[Dict]:
        """
        Lookup word pronunciations.
        
        Args:
            word: Word to lookup
        
        Returns:
            Dict with 'ipa', 'arpa', 'arteme' lists, or None if not found
        """
        word = word.lower().strip()
        
        row = self.conn.execute(
            "SELECT ipa, arpa, arteme FROM english WHERE word = ?",
            (word,)
        ).fetchone()
        
        if row:
            return {
                'word': word,
                'ipa': row['ipa'].split() if row['ipa'] else [],
                'arpa': row['arpa'].split(' | ') if row['arpa'] else [],
                'arteme': row['arteme'].split() if row['arteme'] else []
            }
        return None
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        total_words = self.conn.execute("SELECT COUNT(*) FROM english").fetchone()[0]
        
        # Count total IPA pronunciations
        total_ipa = 0
        max_pron = 0
        multi_pron = 0
        
        for row in self.conn.execute("SELECT ipa FROM english"):
            count = len(row[0].split())
            total_ipa += count
            max_pron = max(max_pron, count)
            if count > 1:
                multi_pron += 1
        
        # Get import statistics
        import_stats = []
        for row in self.conn.execute("SELECT * FROM import_stats ORDER BY import_date"):
            import_stats.append(dict(row))
        
        return {
            'total_words': total_words,
            'total_pronunciations': total_ipa,
            'avg_pronunciations': total_ipa / total_words if total_words > 0 else 0,
            'max_pronunciations': max_pron,
            'words_with_multiple_pron': multi_pron,
            'import_stats': import_stats
        }
    
    def search(self, pattern: str, limit: int = 20) -> List[Dict]:
        """
        Search for words matching pattern.
        
        Args:
            pattern: SQL LIKE pattern (use % for wildcard)
            limit: Maximum results to return
        
        Returns:
            List of word dictionaries
        """
        results = []
        
        for row in self.conn.execute(
            "SELECT word, ipa, arpa, arteme FROM english WHERE word LIKE ? LIMIT ?",
            (pattern, limit)
        ):
            results.append({
                'word': row['word'],
                'ipa': row['ipa'].split() if row['ipa'] else [],
                'arpa': row['arpa'].split(' | ') if row['arpa'] else [],
                'arteme': row['arteme'].split() if row['arteme'] else []
            })
        
        return results
    
    def export_arpa_dict(self, output_path: str):
        """Export to ARPA dictionary format (tab-separated)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in self.conn.execute("SELECT word, arpa FROM english ORDER BY word"):
                if row['arpa']:
                    # Each pronunciation on a separate line
                    for arpa in row['arpa'].split(' | '):
                        f.write(f"{row['word']}\t{arpa}\n")
        
        print(f"Exported ARPA dictionary to: {output_path}")
    
    def export_ipa_dict(self, output_path: str):
        """Export to IPA dictionary format (tab-separated)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in self.conn.execute("SELECT word, ipa FROM english ORDER BY word"):
                if row['ipa']:
                    # Each pronunciation on a separate line
                    for ipa in row['ipa'].split():
                        f.write(f"{row['word']}\t{ipa}\n")
        
        print(f"Exported IPA dictionary to: {output_path}")
    
    def export_csv(self, output_path: str):
        """Export to CSV format."""
        import csv
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['word', 'ipa', 'arpa', 'arteme'])
            
            for row in self.conn.execute("SELECT word, ipa, arpa, arteme FROM english ORDER BY word"):
                writer.writerow([row['word'], row['ipa'], row['arpa'], row['arteme']])
        
        print(f"Exported CSV to: {output_path}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def cmd_lookup(args):
    """Handle lookup command."""
    query = LexiconQuery(args.db)
    
    result = query.lookup(args.word)
    
    if result:
        print(f"\nWord: {result['word']}")
        print(f"IPA pronunciations ({len(result['ipa'])}):")
        for ipa in result['ipa']:
            print(f"  {ipa}")
        
        if result['arpa']:
            print(f"\nARPA pronunciations ({len(result['arpa'])}):")
            for arpa in result['arpa']:
                print(f"  {arpa}")
        
        if result['arteme']:
            print(f"\nArteme sequences ({len(result['arteme'])}):")
            for arteme in result['arteme']:
                print(f"  {arteme}")
        print()
    else:
        print(f"Word not found: {args.word}")
        sys.exit(1)
    
    query.close()


def cmd_stats(args):
    """Handle stats command."""
    query = LexiconQuery(args.db)
    
    stats = query.get_stats()
    
    print("\n" + "="*60)
    print("LEXICON DATABASE STATISTICS")
    print("="*60)
    print(f"Total unique words: {stats['total_words']:,}")
    print(f"Total IPA pronunciations: {stats['total_pronunciations']:,}")
    print(f"Average pronunciations per word: {stats['avg_pronunciations']:.2f}")
    print(f"Maximum pronunciations for a word: {stats['max_pronunciations']}")
    print(f"Words with multiple pronunciations: {stats['words_with_multiple_pron']:,}")
    
    if stats['import_stats']:
        print("\n" + "-"*60)
        print("IMPORT SOURCES")
        print("-"*60)
        for source_stat in stats['import_stats']:
            print(f"\n{source_stat['source']}:")
            print(f"  Words added: {source_stat['words_added']:,}")
            print(f"  Pronunciations: {source_stat['pronunciations_added']:,}")
            if source_stat['duplicates_skipped'] > 0:
                print(f"  Duplicates skipped: {source_stat['duplicates_skipped']:,}")
            if source_stat['errors'] > 0:
                print(f"  Errors: {source_stat['errors']:,}")
    
    print("="*60 + "\n")
    
    query.close()


def cmd_search(args):
    """Handle search command."""
    query = LexiconQuery(args.db)
    
    results = query.search(args.pattern, args.limit)
    
    if results:
        print(f"\nFound {len(results)} matches for pattern: {args.pattern}")
        print("-"*60)
        
        for result in results:
            print(f"\n{result['word']}:")
            if result['ipa']:
                print(f"  IPA: {' '.join(result['ipa'])}")
            if result['arpa']:
                print(f"  ARPA: {' | '.join(result['arpa'])}")
        print()
    else:
        print(f"No matches found for pattern: {args.pattern}")
    
    query.close()


def cmd_export(args):
    """Handle export command."""
    query = LexiconQuery(args.db)
    
    if args.format == 'arpa':
        query.export_arpa_dict(args.output)
    elif args.format == 'ipa':
        query.export_ipa_dict(args.output)
    elif args.format == 'csv':
        query.export_csv(args.output)
    else:
        print(f"Unknown format: {args.format}")
        sys.exit(1)
    
    query.close()


def main():
    parser = argparse.ArgumentParser(
        description='Query English Lexicon Database'
    )
    parser.add_argument(
        '--db',
        default='data/english_lexicon.db',
        help='Database path (default: data/english_lexicon.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Lookup command
    lookup_parser = subparsers.add_parser('lookup', help='Lookup word pronunciations')
    lookup_parser.add_argument('word', help='Word to lookup')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for words')
    search_parser.add_argument('pattern', help='SQL LIKE pattern (use % for wildcard)')
    search_parser.add_argument('--limit', type=int, default=20, help='Max results (default: 20)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export database')
    export_parser.add_argument('--format', choices=['arpa', 'ipa', 'csv'], required=True,
                              help='Export format')
    export_parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'lookup':
        cmd_lookup(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'export':
        cmd_export(args)


if __name__ == "__main__":
    main()