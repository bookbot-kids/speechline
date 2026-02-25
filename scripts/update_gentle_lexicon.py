#!/usr/bin/env python3
"""
Update Gentle Lexicon Across All Locations

This script:
1. Backs up existing lexicons
2. Copies speechline lexicon to both Gentle locations
3. Regenerates .int files from the symbol tables
4. Rebuilds FST files using Kaldi tools
5. Validates the update

Usage:
    python scripts/update_gentle_lexicon.py [--dry-run]
"""

import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Paths
GENTLE_DIR = Path("/mnt/Store07/Projects/gentle")
SPEECHLINE_LEXICON = Path("data/align_lexicon.txt")

# Gentle locations for lexicon files
GENTLE_LOCATIONS = [
    {
        "name": "langdir",
        "lexicon_txt": GENTLE_DIR / "exp/langdir/phones/align_lexicon.txt",
        "lexicon_int": GENTLE_DIR / "exp/langdir/phones/align_lexicon.int",
        "words_txt": GENTLE_DIR / "exp/langdir/words.txt",
        "phones_txt": GENTLE_DIR / "exp/langdir/phones.txt",
        "l_fst": GENTLE_DIR / "exp/langdir/L.fst",
        "l_disambig_fst": GENTLE_DIR / "exp/langdir/L_disambig.fst",
    },
    {
        "name": "graph_pp",
        "lexicon_txt": GENTLE_DIR / "exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.txt",
        "lexicon_int": GENTLE_DIR / "exp/tdnn_7b_chain_online/graph_pp/phones/align_lexicon.int",
        "words_txt": GENTLE_DIR / "exp/tdnn_7b_chain_online/graph_pp/words.txt",
        "phones_txt": GENTLE_DIR / "exp/tdnn_7b_chain_online/graph_pp/phones.txt",
    }
]


class GentleLexiconUpdater:
    """Update Gentle lexicons and rebuild FST files"""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_files = []
        
    def log(self, message, level="INFO"):
        """Print log message"""
        prefix = "[DRY-RUN] " if self.dry_run else ""
        print(f"{prefix}{level}: {message}")
    
    def backup_file(self, file_path: Path):
        """Create backup of a file"""
        if not file_path.exists():
            self.log(f"File does not exist, skipping backup: {file_path}", "WARN")
            return None
        
        backup_path = file_path.with_suffix(f".backup.{self.timestamp}")
        
        if self.dry_run:
            self.log(f"Would backup: {file_path} -> {backup_path}")
            return backup_path
        
        try:
            shutil.copy2(file_path, backup_path)
            self.backup_files.append((file_path, backup_path))
            self.log(f"Backed up: {backup_path}")
            return backup_path
        except Exception as e:
            self.log(f"Failed to backup {file_path}: {e}", "ERROR")
            return None
    
    def copy_lexicon(self, source: Path, dest: Path):
        """Copy lexicon file"""
        if self.dry_run:
            self.log(f"Would copy: {source} -> {dest}")
            return True
        
        try:
            # Ensure parent directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            self.log(f"Copied lexicon to: {dest}")
            return True
        except Exception as e:
            self.log(f"Failed to copy {source} to {dest}: {e}", "ERROR")
            return False
    
    def regenerate_int_file(self, location: dict):
        """Regenerate .int file from .txt using symbol tables"""
        lexicon_txt = location["lexicon_txt"]
        lexicon_int = location["lexicon_int"]
        words_txt = location["words_txt"]
        phones_txt = location["phones_txt"]
        
        if not lexicon_txt.exists():
            self.log(f"Lexicon .txt not found: {lexicon_txt}", "ERROR")
            return False
        
        if not words_txt.exists() or not phones_txt.exists():
            self.log(f"Symbol tables not found for {location['name']}", "ERROR")
            return False
        
        self.log(f"Regenerating .int file for {location['name']}...")
        
        if self.dry_run:
            self.log(f"Would regenerate: {lexicon_int}")
            return True
        
        try:
            # Load symbol tables
            word_to_id = self._load_symbol_table(words_txt)
            phone_to_id = self._load_symbol_table(phones_txt)
            
            entries_written = 0
            entries_skipped = 0
            
            with open(lexicon_txt, 'r', encoding='utf-8') as fin, \
                 open(lexicon_int, 'w', encoding='utf-8') as fout:
                
                for line_num, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    
                    word = parts[0]
                    phones = parts[2:]
                    
                    # Map word to ID
                    if word not in word_to_id:
                        if entries_skipped < 10:
                            self.log(f"Word '{word}' not in words.txt (line {line_num})", "WARN")
                        entries_skipped += 1
                        continue
                    
                    word_id = word_to_id[word]
                    
                    # Map phones to IDs
                    phone_ids = []
                    skip_entry = False
                    for phone in phones:
                        if phone not in phone_to_id:
                            if entries_skipped < 10:
                                self.log(f"Phone '{phone}' not in phones.txt (line {line_num})", "WARN")
                            skip_entry = True
                            break
                        phone_ids.append(phone_to_id[phone])
                    
                    if skip_entry:
                        entries_skipped += 1
                        continue
                    
                    # Write integer format: word_id word_id phone_id1 phone_id2 ...
                    fout.write(f"{word_id} {word_id} {' '.join(phone_ids)}\n")
                    entries_written += 1
            
            self.log(f"✓ Wrote {entries_written:,} entries to {lexicon_int.name}")
            if entries_skipped > 0:
                self.log(f"⚠ Skipped {entries_skipped:,} entries (not in symbol tables)", "WARN")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to regenerate .int file: {e}", "ERROR")
            return False
    
    def _load_symbol_table(self, symbol_file: Path) -> dict:
        """Load a Kaldi symbol table (symbol -> ID mapping)"""
        symbols = {}
        with open(symbol_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    symbol, idx = parts
                    symbols[symbol] = idx
        return symbols
    
    def rebuild_fst_files(self):
        """Rebuild FST files for langdir location"""
        self.log("Rebuilding FST files for langdir...")
        
        if self.dry_run:
            self.log("Would rebuild L.fst and L_disambig.fst")
            return True
        
        script_path = Path("scripts/rebuild_gentle_lexicon.sh")
        if not script_path.exists():
            self.log("rebuild_gentle_lexicon.sh not found, skipping FST rebuild", "WARN")
            return False
        
        try:
            result = subprocess.run(
                ["bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.log("✓ FST files rebuilt successfully")
                return True
            else:
                self.log(f"FST rebuild failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("FST rebuild timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Failed to rebuild FST files: {e}", "ERROR")
            return False
    
    def validate_update(self):
        """Validate that updates were successful"""
        self.log("\nValidating updates...")
        
        all_valid = True
        for location in GENTLE_LOCATIONS:
            lexicon_txt = location["lexicon_txt"]
            lexicon_int = location["lexicon_int"]
            
            if not lexicon_txt.exists():
                self.log(f"✗ Missing: {lexicon_txt}", "ERROR")
                all_valid = False
            else:
                self.log(f"✓ Found: {lexicon_txt}")
            
            if not lexicon_int.exists():
                self.log(f"✗ Missing: {lexicon_int}", "ERROR")
                all_valid = False
            else:
                self.log(f"✓ Found: {lexicon_int}")
        
        # Check FST files for langdir
        l_fst = GENTLE_LOCATIONS[0]["l_fst"]
        l_disambig_fst = GENTLE_LOCATIONS[0]["l_disambig_fst"]
        
        if l_fst.exists():
            self.log(f"✓ Found: {l_fst}")
        else:
            self.log(f"✗ Missing: {l_fst}", "WARN")
        
        if l_disambig_fst.exists():
            self.log(f"✓ Found: {l_disambig_fst}")
        else:
            self.log(f"✗ Missing: {l_disambig_fst}", "WARN")
        
        return all_valid
    
    def rollback(self):
        """Rollback changes using backups"""
        self.log("\n🔄 Rolling back changes...")
        
        for original, backup in self.backup_files:
            try:
                shutil.copy2(backup, original)
                self.log(f"Restored: {original}")
            except Exception as e:
                self.log(f"Failed to restore {original}: {e}", "ERROR")
    
    def run(self):
        """Execute the complete update process"""
        print("=" * 70)
        print("GENTLE LEXICON UPDATE")
        print("=" * 70)
        print()
        
        if self.dry_run:
            print("🔍 DRY-RUN MODE: No changes will be made")
            print()
        
        # Verify source lexicon exists
        if not SPEECHLINE_LEXICON.exists():
            self.log(f"Source lexicon not found: {SPEECHLINE_LEXICON}", "ERROR")
            return False
        
        self.log(f"Source lexicon: {SPEECHLINE_LEXICON}")
        self.log(f"Size: {SPEECHLINE_LEXICON.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        
        try:
            # Step 1: Backup all existing files
            print("STEP 1: Creating backups")
            print("-" * 70)
            for location in GENTLE_LOCATIONS:
                self.backup_file(location["lexicon_txt"])
                self.backup_file(location["lexicon_int"])
            
            # Backup FST files for langdir
            self.backup_file(GENTLE_LOCATIONS[0]["l_fst"])
            self.backup_file(GENTLE_LOCATIONS[0]["l_disambig_fst"])
            print()
            
            # Step 2: Copy lexicon to both locations
            print("STEP 2: Copying lexicon files")
            print("-" * 70)
            for location in GENTLE_LOCATIONS:
                if not self.copy_lexicon(SPEECHLINE_LEXICON, location["lexicon_txt"]):
                    raise Exception(f"Failed to copy to {location['name']}")
            print()
            
            # Step 3: Regenerate .int files
            print("STEP 3: Regenerating .int files")
            print("-" * 70)
            for location in GENTLE_LOCATIONS:
                if not self.regenerate_int_file(location):
                    raise Exception(f"Failed to regenerate .int for {location['name']}")
            print()
            
            # Step 4: Rebuild FST files
            print("STEP 4: Rebuilding FST files")
            print("-" * 70)
            if not self.rebuild_fst_files():
                self.log("FST rebuild failed, but continuing...", "WARN")
            print()
            
            # Step 5: Validate
            print("STEP 5: Validation")
            print("-" * 70)
            if not self.validate_update():
                raise Exception("Validation failed")
            
            print()
            print("=" * 70)
            print("✅ UPDATE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print()
            print("Backup files created:")
            for _, backup in self.backup_files:
                print(f"  - {backup}")
            print()
            print("Next steps:")
            print("1. Run: python scripts/validate_gentle_lexicon.py")
            print("2. Test alignment with sample files")
            print("3. Re-process error files to verify improvements")
            print()
            
            return True
            
        except Exception as e:
            self.log(f"\n❌ UPDATE FAILED: {e}", "ERROR")
            
            if not self.dry_run and self.backup_files:
                response = input("\nRollback changes? (y/n): ")
                if response.lower() == 'y':
                    self.rollback()
            
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Update Gentle lexicon across all locations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run (no changes)
  python scripts/update_gentle_lexicon.py --dry-run
  
  # Execute update
  python scripts/update_gentle_lexicon.py
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test run without making changes'
    )
    
    args = parser.parse_args()
    
    updater = GentleLexiconUpdater(dry_run=args.dry_run)
    success = updater.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()